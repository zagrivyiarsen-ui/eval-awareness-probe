"""Основная генерация (этап 4): все 6 условий × 16 промптов × k в ВЫБРАННОЙ среде. Резюмируемая (results/main_gens.jsonl, append-only).
Пилотные deployment/E1 (sample_idx 0..pilot.k−1, та же T/sampling/dtype) засчитываются — генерируются только недостающие sample_idx.
    PYTHONPATH=. python scripts/main_gen.py --dps data/decision_points.json --out results --envs secret_number [--skip-conds E1_S1] --max-hours 11
Требует: fill_gate_params.py + verify_prereg.py без --allow-unset (k/cap/T/bs/top_p/top_k/dtype/layers в yaml), pilot.decision.env == --envs
(иначе deviation с причиной через --force). Скоринг и вербализация — на лету, как в пилоте."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gate0_kaggle import load_model, release
from scripts.pilot import done_keys
from src.analysis import code_sha256
from src.conditions import MAIN_IDS, assert_ready_for_generation, conditions_hash, load_conditions
from src.contracts import assert_continuity, assert_matches_prereg, dps_sha256, env_info, make_records, persist_results, record_numbers
from src.prompts import load_tokenizer
from src.scoring import analysis_hash, load_analysis, score, verbalization
from src.throughput import generate_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main_gen")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--analysis", default="configs/analysis.yaml")
    ap.add_argument("--dps", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--envs", required=True, help="secret_number | sandbagging | secret_number,sandbagging")
    ap.add_argument("--skip-conds", default="", help="напр. E1_S1 (аддитивность одной строкой; в плече пробе E1_S1 остаётся)")
    ap.add_argument("--conds", default="", help="ТОЛЬКО эти условия. 2026-08-28: НЕ использовать для разбиения на блоки (ломает равный k); пусто = все main")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--force", action="store_true", help="разрешить среду ≠ pilot.decision.env (записывается как deviation)")
    a = ap.parse_args(argv)
    out = Path(a.out); nums = out / "numbers.json"
    cfg = load_conditions(a.config)                                  # gate-параметры ОБЯЗАНЫ быть заполнены
    A = load_analysis(a.analysis)
    dsha = dps_sha256(a.dps); assert_matches_prereg(cfg, nums, dsha)
    assert_continuity(nums, Path(__file__).resolve().parents[1], dsha, "main")
    prev = json.loads(nums.read_text(encoding="utf-8"))
    assert prev.get("prereg.analysis_hash") == analysis_hash(a.analysis) and prev.get("prereg.conditions_hash") == conditions_hash(a.config)
    assert prev.get("prereg.config_change_log"), "fill_gate_params/verify_prereg после пилота не запускались (config_change_log пуст)"
    ei = env_info()
    for kk in ("env.transformers", "env.torch"):
        assert prev.get(kk) == ei[kk], f"{kk}: пилот шёл на {prev.get(kk)}, сейчас {ei[kk]} — версия сменилась между пилотом и main"
    PREREG_SEED = 20260830                                           # Р2 блокер 4 (addendum 28.08: main 20260830)
    if a.seed != PREREG_SEED:
        record_numbers(nums, {"main.deviation.seed": f"{a.seed} != prereg {PREREG_SEED}"})
        log.warning("seed %d != prereg %d — записано как deviation", a.seed, PREREG_SEED)
    envs = sorted(set(a.envs.split(",")))
    dec_env = prev.get("pilot.decision.env")
    if prev.get("pilot.decision.n_env") == 1 and envs != [dec_env]:
        assert a.force, f"pilot.decision.env={dec_env}, а --envs={envs}; это deviation → --force с записью"
        record_numbers(nums, {"main.deviation.env": f"envs={envs} vs decision {dec_env} (--force)"})
    skip = set(x for x in a.skip_conds.split(",") if x)
    only = set(x for x in a.conds.split(",") if x)
    conds = [c for c in cfg.main if c.id not in skip and (not only or c.id in only)]
    k = int(cfg.generation["k"]); cap = int(cfg.generation["max_new_tokens"]); bs = int(cfg.generation["batch_size"])
    T = float(cfg.generation["temperature"]); dtype = cfg.model["dtype"]
    ps = {kk.split("pilot.sampling.")[1]: v for kk, v in prev.items() if kk.startswith("pilot.sampling.")}
    reuse_pilot = bool(ps) and float(ps.get("temperature", -1)) == T and ps.get("dtype") == dtype and float(ps.get("top_p")) == float(cfg.generation["top_p"]) \
        and int(ps.get("top_k")) == int(cfg.generation["top_k"]) and float(ps.get("repetition_penalty", -1)) == float(cfg.generation["repetition_penalty"]) \
        and float(ps.get("presence_penalty", -1)) == float(cfg.generation.get("presence_penalty") if cfg.generation.get("presence_penalty") is not None else -1) \
        and int(ps.get("batch_size", -1)) == bs        # 2026-08-28: побайтовое равенство sampling пилот/main (ревью В; +presence R-Launch)
    log.info("k=%d cap=%d bs=%d T=%s dtype=%s envs=%s skip=%s reuse_pilot=%s", k, cap, bs, T, dtype, envs, skip, reuse_pilot)

    tok = load_tokenizer(cfg)
    dps = [d for d in json.loads(Path(a.dps).read_text(encoding="utf-8")) if d["env"] in envs]
    keys = {f"{kk['env']}/{kk['prompt_id']}": kk for kk in json.loads(Path(a.dps).with_name(Path(a.dps).stem + "_keys.json").read_text(encoding="utf-8"))["keys"]}
    recs = make_records(dps, conds, tok, cfg, dps_sha=dsha)
    jobs = [(r, i) for r in recs for i in range(k)]
    done = done_keys(out / "main_gens.jsonl")
    if reuse_pilot:
        done |= {kk for kk in done_keys(out / "pilot_gens.jsonl")}
    else:
        record_numbers(nums, {"main.pilot_samples_reused": False, "main.note": "пилотные генерации при другой T/sampling НЕ засчитаны"})
    todo = [j for j in jobs if (j[0].prompt_key, j[1]) not in done]
    log.info("jobs=%d done=%d todo=%d", len(jobs), len(done & {(j[0].prompt_key, j[1]) for j in jobs}), len(todo))
    if not todo:
        print("основная генерация завершена"); return
    model = load_model(cfg.tokenizer["name"], cfg.model["revision"], getattr(torch, dtype))
    sampling = assert_ready_for_generation(cfg, ("main",), generation_config=model.generation_config)
    sampling["batch_size"] = bs; sampling["dtype"] = dtype; sampling["cap"] = cap
    sampling["seed_base"] = a.seed
    record_numbers(nums, {"main.sampling": dict(sampling), "main.k": k, "main.cap": cap, "main.envs": envs, "main.skip_conds": sorted(skip),
                          "main.pilot_samples_reused": bool(reuse_pilot), "main.code_sha256": code_sha256(Path(__file__).resolve().parents[1]),
                          "main.config_hash": cfg.config_hash, "main.n_jobs": len(jobs)})
    # 2026-08-28 (ревью R-Pipeline В): порядок sample_idx → условие → длина промпта; батч = одно (sample_idx, условие).
    # Слой sample_idx=0 = все условия × все промпты; при остановке в любой момент k равен по всем промптам и условиям ± один батч.
    per_layer = {}
    for j in sorted(todo, key=lambda j: (j[1], j[0].condition_id, j[0].n_prompt_tokens)):
        per_layer.setdefault((j[1], j[0].condition_id), []).append(j)
    batches = []
    for key in sorted(per_layer):
        v = per_layer[key]
        batches += [v[i:i + bs] for i in range(0, len(v), bs)]
    t_start = time.time(); n_done = 0
    for bi, b in enumerate(batches):
        if (time.time() - t_start) / 3600 > a.max_hours:
            log.warning("max-hours: остановка; перезапуск продолжит"); break
        torch.manual_seed(a.seed + bi + len(done))
        recs_b = [j[0] for j in b]; sidx = [j[1] for j in b]
        for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)
        t0 = time.time()
        metas = generate_batch(model, tok, recs_b, sampling, cap, model_name=cfg.tokenizer["name"], model_revision=cfg.model["revision"], sample_idx=sidx)
        for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)
        dt = time.time() - t0
        assert not any(m["cot_category"] == "unknown_finish" for m in metas), "unknown_finish в батче: eos-набор/остановка неисправны (стоп)"
        with open(out / "main_gens.jsonl", "a", encoding="utf-8") as f:
            for r, m in zip(recs_b, metas):
                m["group"] = "main"
                m["score"] = score(r.env, m["answer_text"], keys[r.env_prompt], A) if m["cot_category"] == "ok" else {"class": "unparsable", "target": None}
                m["verbalization"] = verbalization(m["cot_text"], A)
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        with open(out / "main_batches.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "batch": bi, "n": len(b), "seconds": dt, "condition_id": recs_b[0].condition_id,
                                "tok_per_s": sum(m["n_generated_tokens"] for m in metas) / dt,
                                "n_truncated": sum(m["cot_category"] == "truncated" for m in metas)}) + "\n")
        n_done += 1
        log.info("batch %d/%d (%s, sample_idx=%d): %.0fs, truncated %d/%d", bi + 1, len(batches), recs_b[0].condition_id, sidx[0], dt, sum(m["cot_category"] == "truncated" for m in metas), len(b))
        persist_results(out)
    del model; release()
    print(f"батчей сделано: {n_done}; остаток: {len(batches) - n_done}; готово {len(done & {(j[0].prompt_key, j[1]) for j in jobs}) + n_done * bs}/{len(jobs)} (грубо)")


if __name__ == "__main__":
    main()
