"""Пилот (Gate 1) по configs/analysis.yaml §pilot: deployment + E1 × 32 DP × k=6, позитивный контроль 10 промптов × k=6, cap 4096,
плюс S-parse-probe (вариант (ii) ревью: 4 DP/среда × S1/S2/Dstr × k=1 — на диск пишется ТОЛЬКО parse_ok/cot_category, текст не сохраняется,
метки поведения не считаются). Резюмируемый: append-only results/pilot_gens.jsonl, пропуск готовых (prompt_key, sample_idx).
    PYTHONPATH=. python scripts/pilot.py --dps data/decision_points.json --out results --temperature 0.7 --top-p 0.95 --top-k 20 \\
        --batch-size 24 --dtype float16 --max-hours 11
Gate-параметры yaml остаются null: пилот получает их через CLI и ЗАПИСЫВАЕТ (pilot.sampling.*); fill_gate_params.py после пилота
вписывает те же значения в yaml (ассерт равенства) — одна правка config_hash."""
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
from src.analysis import code_sha256
from src.conditions import S_IDS, assert_ready_for_generation, conditions_hash, load_conditions
from src.contracts import assert_continuity, assert_matches_prereg, dps_sha256, make_records, persist_results, record_numbers
from src.prompts import load_tokenizer
from src.scoring import analysis_hash, load_analysis, score, verbalization
from src.throughput import generate_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pilot")


def jobs_for(recs, A, dps, escalate_pc=False):
    P = A["pilot"]; PC = A["positive_control"]
    ids_by_env = {}
    for dp in dps:
        ids_by_env.setdefault(dp["env"], []).append(dp["prompt_id"])
    main = [(r, i, "main") for r in recs if r.condition_id in P["conditions"] for i in range(P["k"])]
    pc = [(r, i, "pc") for r in recs if r.condition_id == "positive_control" and r.prompt_id in PC["prompt_ids"] for i in range(PC["k"])]
    if escalate_pc:   # D4: одна попытка усиления, те же 10 промптов, тот же k
        pc += [(r, i, "pc") for r in recs if r.condition_id == PC["escalation_condition"] and r.prompt_id in PC["prompt_ids"] for i in range(PC["k"])]
    sp = []
    if P["s_parse_probe"]["enabled"]:
        first = {e: set(v[:P["s_parse_probe"]["prompt_ids_per_env"]]) for e, v in ids_by_env.items()}
        sp = [(r, i, "s_parse") for r in recs if r.condition_id in P["s_parse_probe"]["conditions"] and r.prompt_id in first[r.env]
              for i in range(P["s_parse_probe"]["k"])]
    return main + pc + sp


def done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {(json.loads(l)["prompt_key"], json.loads(l)["sample_idx"]) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--analysis", default="configs/analysis.yaml")
    ap.add_argument("--dps", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float, required=True)
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], required=True)
    ap.add_argument("--cap", type=int, default=None, help="default: analysis.yaml pilot.cap")
    ap.add_argument("--max-hours", type=float, default=11.0, help="остановиться до конца сессии Kaggle")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--stop-after", default=None, choices=["pc", "main_deployment", "s_parse", "main_E1"],
                    help="остановиться после группы (стоп-точка «выбор среды» = main_deployment/s_parse)")
    ap.add_argument("--envs", default=None, help="ограничить main-генерации средой(ами), напр. secret_number (после выбора среды)")
    ap.add_argument("--max-sample-idx", type=int, default=None,
                    help="только sample_idx < N для main-групп (напр. 3: deployment k=3 в обеих средах для выбора среды; добор до k=6 — в выбранной)")
    ap.add_argument("--escalate-pc", action="store_true", help="D4: positive_control_strong на тех же 10 промптах (только если PC мёртв; один раз)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert "fake" not in out.name and "fake" not in Path(a.dps).name
    nums = out / "numbers.json"
    cfg = load_conditions(a.config, allow_unset_gate_params=True)
    A = load_analysis(a.analysis)
    dsha = dps_sha256(a.dps); assert_matches_prereg(cfg, nums, dsha)
    assert_continuity(nums, Path(__file__).resolve().parents[1], dsha, "pilot")
    P = A["pilot"]                                           # 2026-08-28: был NameError в main() (ревью R-Pipeline Г-1)
    prev = json.loads(nums.read_text(encoding="utf-8"))
    assert prev.get("prereg.analysis_hash") == analysis_hash(a.analysis), "analysis.yaml изменён после verify_prereg"
    assert prev.get("prereg.conditions_hash") == conditions_hash(a.config), "тексты условий изменены после verify_prereg"
    assert prev.get("gate0.fp16.accepted") is True or a.dtype == "bfloat16", "fp16 не принят gate0_fp16_checks.py → --dtype bfloat16"
    cap = a.cap or int(A["pilot"]["cap"])
    T = a.temperature
    assert T in (A["temperature"]["default"], A["temperature"]["alternative"]), f"temperature {T} не из prereg {A['temperature']}"

    tok = load_tokenizer(cfg)
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8"))
    keys = {f"{k['env']}/{k['prompt_id']}": k for k in json.loads(Path(a.dps).with_name(Path(a.dps).stem + "_keys.json").read_text(encoding="utf-8"))["keys"]}
    conds = cfg.main + cfg.group("positive_control") + (cfg.group("positive_control_strong") if a.escalate_pc else [])
    recs = make_records(dps, conds, tok, cfg, dps_sha=dsha)
    if a.escalate_pc:
        assert prev.get("pilot.positive_control.alive") is False, "усиление PC разрешено только после gate1_checks с positive_control.alive=False"
        assert not prev.get("pilot.pc_escalated"), "усиление PC уже выполнялось — второго повтора план не даёт"
        # 2026-08-28 (Р2 блокер 3): pc_escalated ставится ПОСЛЕ фактического завершения strong-группы (см. конец main) —
        # иначе сорванный B3b блокировал бы повтор навсегда; резюме идёт по done_keys, второй ПОЛНЫЙ замер невозможен.
    jobs = jobs_for(recs, A, dps, escalate_pc=a.escalate_pc)
    # Р2 блокер 4: сид — prereg-величина (addendum 28.08: пилот 20260827); отклонение = deviation с записью
    PREREG_SEED = 20260827
    if a.seed != PREREG_SEED:
        record_numbers(nums, {"pilot.deviation.seed": f"{a.seed} != prereg {PREREG_SEED}"})
        log.warning("seed %d != prereg %d — записано как deviation", a.seed, PREREG_SEED)
    if a.max_sample_idx is not None:
        jobs = [j for j in jobs if j[2] != "main" or j[1] < a.max_sample_idx]
    if a.envs:
        keep = set(a.envs.split(","))
        jobs = [j for j in jobs if j[2] != "main" or j[0].env in keep]
        record_numbers(nums, {"pilot.envs_restricted_to": sorted(keep)})
    gens_path, sp_path = out / "pilot_gens.jsonl", out / "pilot_s_parse.jsonl"
    done = done_keys(gens_path) | done_keys(sp_path)
    todo = [j for j in jobs if (j[0].prompt_key, j[1]) not in done]
    log.info("jobs=%d done=%d todo=%d cap=%d", len(jobs), len(done), len(todo), cap)
    if not todo:
        print("пилот завершён: все генерации на диске"); return

    model = load_model(cfg.tokenizer["name"], cfg.model["revision"], getattr(torch, a.dtype))
    cfg.generation.update({"top_p": a.top_p, "top_k": a.top_k, "max_new_tokens": cap})
    sampling = assert_ready_for_generation(cfg, ("main", "positive_control") + (("positive_control_strong",) if a.escalate_pc else ()), temperature_override=T, generation_config=model.generation_config)
    sampling["batch_size"] = a.batch_size; sampling["dtype"] = a.dtype; sampling["cap"] = cap; sampling["seed_base"] = a.seed
    record_numbers(nums, {"pilot.sampling": {k: v for k, v in sampling.items()}, "pilot.code_sha256": code_sha256(Path(__file__).resolve().parents[1]),
                          "pilot.analysis_hash": analysis_hash(a.analysis), "pilot.n_jobs": len(jobs)})
    # батчи: одна группа (main / pc / s_parse), сортировка по длине промпта → меньше паддинга
    order = {g: i for i, g in enumerate(P.get("order", ["pc", "main_deployment", "s_parse", "main_E1"]))}
    def og(j):                      # группа для порядка: main расщепляется по условию
        return f"main_{j[0].condition_id}" if j[2] == "main" else j[2]
    # main: чередуем среды по DP (sb_001, sn_001, sb_002, …), внутри DP — deployment, E1, sample_idx; батч = 8 → одна среда на батч
    # (без смешения длин промптов), но обе среды продвигаются параллельно
    def key(j):
        r, i, g = j
        pid_n = int(r.prompt_id.split("_")[-1])
        return (order.get(og(j), 9), pid_n, r.env, i) if g == "main" else (order.get(g, 9), r.env, r.n_prompt_tokens, i)
    todo.sort(key=key)
    batches = []
    for g in sorted({og(j) for j in todo}, key=lambda x: order.get(x, 9)):
        per_env = []
        for e in sorted({j[0].env for j in todo}):
            part = [j for j in todo if og(j) == g and j[0].env == e]
            per_env.append([part[i:i + a.batch_size] for i in range(0, len(part), a.batch_size)])
        while any(per_env):                                  # чередуем среды внутри группы
            for lst in per_env:
                if lst: batches.append(lst.pop(0))
    if a.stop_after:
        cut = order[a.stop_after]
        batches = [b for b in batches if order.get(og(b[0]), 9) <= cut]
        log.info("stop-after=%s: %d батчей в этом запуске", a.stop_after, len(batches))
    t_start = time.time(); tok_s_log = []
    for bi, b in enumerate(batches):
        if (time.time() - t_start) / 3600 > a.max_hours:
            log.warning("max-hours достигнут: остановка до конца сессии; перезапуск продолжит с места"); break
        torch.manual_seed(a.seed + bi + len(done))
        recs_b = [j[0] for j in b]; sidx = [j[1] for j in b]; grp = b[0][2]
        assert len({og(j) for j in b}) == 1 and len({j[0].env for j in b}) == 1
        for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)
        t0 = time.time()
        metas = generate_batch(model, tok, recs_b, sampling, cap, model_name=cfg.tokenizer["name"], model_revision=cfg.model["revision"], sample_idx=sidx)
        for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)
        dt = time.time() - t0
        assert not any(m["cot_category"] == "unknown_finish" for m in metas), "unknown_finish в батче: eos-набор/остановка неисправны (стоп)"
        n_tok = sum(m["n_generated_tokens"] for m in metas)
        tok_s_log.append({"batch": bi, "n": len(b), "seconds": dt, "tok_per_s": n_tok / dt, "group": grp, "subgroup": og(b[0]), "env": recs_b[0].env,
                          "max_len": max(m["n_generated_tokens"] for m in metas), "n_truncated": sum(m["cot_category"] == "truncated" for m in metas)})
        with open(gens_path if grp != "s_parse" else sp_path, "a", encoding="utf-8") as f:
            for r, m in zip(recs_b, metas):
                if grp == "s_parse":
                    # (ii): только parse_ok / категория — ни текста, ни метки поведения
                    sc = score(r.env, m["answer_text"], keys[r.env_prompt], A) if m["cot_category"] == "ok" else {"class": "unparsable"}
                    row = {"prompt_key": r.prompt_key, "sample_idx": m["sample_idx"], "env": r.env, "condition_id": r.condition_id,
                           "cot_category": m["cot_category"], "parse_ok": sc["class"] != "unparsable", "n_generated_tokens": m["n_generated_tokens"]}
                else:
                    assert r.condition_id not in S_IDS
                    m["score"] = score(r.env, m["answer_text"], keys[r.env_prompt], A) if m["cot_category"] == "ok" else {"class": "unparsable", "target": None}
                    m["verbalization"] = verbalization(m["cot_text"], A)
                    row = m
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(out / "pilot_batches.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **tok_s_log[-1]}) + "\n")
        log.info("batch %d/%d (%s): %.0fs, %.1f tok/s, max_len=%d", bi + 1, len(batches), grp, dt, tok_s_log[-1]["tok_per_s"], tok_s_log[-1]["max_len"])
        persist_results(out)                                  # инкрементальный бэкап после каждого батча
    del model; release()
    if a.escalate_pc:                    # Р2 блокер 3: флаг — только когда ВСЯ strong-группа на диске (полный «один повтор» состоялся)
        want = {(j[0].prompt_key, j[1]) for j in jobs if j[0].condition_id == A["positive_control"]["escalation_condition"]}
        have = done_keys(gens_path)
        if want and want <= have:
            record_numbers(nums, {"pilot.pc_escalated": True})
            log.info("positive_control_strong: все %d генераций на диске — pilot.pc_escalated=True", len(want))
        else:
            log.warning("strong-группа не завершена (%d/%d) — pc_escalated НЕ ставится; перезапуск B3b доделает", len(want & have), len(want))
    rem = len(batches) - len(tok_s_log)
    print(f"сделано батчей: {len(tok_s_log)}; остаток в этом запуске: {rem}")
    if rem == 0 and a.stop_after in ("main_deployment", "s_parse"):
        print("СТОП-ТОЧКА «выбор среды»: PYTHONPATH=. python scripts/gate1_checks.py --dps ... --out results --today YYYY-MM-DD --hours-used-this-week H"
              "\n  затем: pilot.py ... --envs <выбранная среда>   (E1 k=6 только там; затем fill_gate_params → основная генерация)")


if __name__ == "__main__":
    main()
