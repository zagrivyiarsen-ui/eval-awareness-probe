"""Gate 0 на реальной модели. Проверки 1–3, 5–8, G + toy-probe диагностика + априорный MDE.
    PYTHONPATH=. python scripts/gate0_real_model.py --dps data/decision_points.json --temperature 0.7 --revision <sha>
Dry-run ДО заморозки (только fake-DP, только каталог *fake*):
    PYTHONPATH=. python scripts/gate0_real_model.py --fake-config --dps data/decision_points_fake.json --out results_fake --temperature 0.7
S-условия участвуют только в check 1/2 (паддинг, последний токен); их активации НЕ сохраняются и не проецируются."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.activations import (check_hook_matches_hidden_states, check_last_token_uniform, check_naive_last_index_breaks_on_right_pad,
                             check_norm_vs_length, check_padding_invariance, check_repeatability_and_memory,
                             default_candidate_layers, extract, get_layers)
from src.conditions import S_IDS, assert_ready_for_generation, load_conditions
from src.contracts import (acts_payload, atomic_write_text, dps_sha256, env_info, make_records, mde_apriori, record_numbers,
                           validate_acts_payload)
from src.probe_toy import position_diagnostic
from src.prompts import load_tokenizer, render_text
from src.throughput import check_batch_generation_matches_single, eos_ids, measure_throughput

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gate0")


def load_model(name, revision, dtype):
    from transformers import AutoModelForCausalLM
    kw = dict(device_map="auto" if torch.cuda.is_available() else None)
    if revision:
        kw["revision"] = revision
    last = None
    for attn in ("flash_attention_2", "sdpa"):
        for dkey in ("dtype", "torch_dtype"):
            try:
                m = AutoModelForCausalLM.from_pretrained(name, attn_implementation=attn, **{dkey: dtype}, **kw)
                log.info("loaded %s attn=%s (%s=%s)", type(m).__name__, attn, dkey, dtype)
                return m, attn
            except TypeError as e:          # dtype vs torch_dtype
                last = e; continue
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("%s unavailable (%s)", attn, str(e)[:120])
                break
    raise RuntimeError(f"cannot load {name}: {last}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--dps", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None, help="ревизия весов; в реальном Gate 0 должна совпадать с model.revision в yaml")
    ap.add_argument("--layers", default=None, help="ТОЛЬКО для отладки; в реальном Gate 0 не задавать (кандидаты по правилу)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--fake-config", action="store_true")
    ap.add_argument("--n-gen", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.fake_config:
        assert "fake" in out.name and "fake" in Path(a.dps).name, "--fake-config: и --out, и --dps обязаны содержать 'fake' (ревью 1.1)"
        from tests.fake_config import load_fake
        cfg = load_fake(out, fill_gate_params=False)
    else:
        assert "fake" not in out.name and "fake" not in Path(a.dps).name, "реальный Gate 0 с fake-входами"
        cfg = load_conditions(a.config, allow_unset_gate_params=True)
        if cfg.model.get("revision") and a.revision:
            assert cfg.model["revision"] == a.revision, "--revision != model.revision в yaml"
        a.revision = a.revision or cfg.model.get("revision")
    out.mkdir(parents=True, exist_ok=True)
    nums_path = out / "numbers.json"
    model_name = a.model or cfg.tokenizer["name"]
    tok = load_tokenizer(cfg, name_override=model_name if model_name != cfg.tokenizer["name"] else None)
    if not torch.cuda.is_available():
        dtype = torch.float32
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16                                # T4/P100 (Kaggle): dry-run только на корректность, не на throughput
        log.warning("GPU без bf16 → fp16; числа throughput НЕ переносимы на A40/L40S, риск переполнения в fp16")
    model, attn = load_model(model_name, a.revision, dtype)
    model.eval()
    n_layers = len(get_layers(model))
    if a.layers:
        log.warning("--layers задан руками — НЕ prereg-выбор")
    layers = [int(x) for x in a.layers.split(",")] if a.layers else (cfg.probe.get("candidate_layers") or default_candidate_layers(n_layers))
    assert len(layers) <= 3
    positions = list(cfg.probe.get("positions_diagnostic") or ["last"])
    thr = cfg.thresholds
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8"))
    dsha = dps_sha256(a.dps)
    conds_all = cfg.main + cfg.group("e2") + cfg.group("positive_control")
    recs_all = make_records(dps, conds_all, tok, cfg, tokenizer_name=model_name, dps_sha=dsha)
    recs_keep = [r for r in recs_all if r.condition_id not in S_IDS]           # сохраняемое/проецируемое
    log.info("model=%s attn=%s n_layers=%d layers=%s positions=%s records=%d (kept for acts: %d)",
             model_name, attn, n_layers, layers, positions, len(recs_all), len(recs_keep))
    record_numbers(nums_path, {"gate0.fake_config": a.fake_config, "gate0.model_name": model_name, "gate0.model_revision": a.revision,
                               "gate0.model_class": type(model).__name__, "gate0.attn_impl": attn, "gate0.n_layers": n_layers,
                               "gate0.candidate_layers": layers, "gate0.layers_from_cli": bool(a.layers), "gate0.positions": positions,
                               "gate0.compute_dtype": str(dtype), "gate0.config_hash": cfg.config_hash, "gate0.dps_sha256": dsha,
                               "gate0.eos_ids": eos_ids(tok, model),
                               "gate0.think_close_special": "</think>" in (tok.all_special_tokens or []), **env_info()})

    preopens = render_text(recs_all[0].messages, tok, cfg).rstrip().endswith("<think>")
    record_numbers(nums_path, {"gate0.template_preopens_think": preopens})
    assert preopens, "шаблон не открывает <think> в промпте — cot_rule и позиция last описаны для другого шаблона"
    u = check_last_token_uniform(recs_all)
    log.info("check2: %s", u)
    record_numbers(nums_path, {"gate0.check2": u})
    assert u["uniform"], "последний токен различается между версиями"

    sub = [r for r in recs_all if r.prompt_id == dps[0]["prompt_id"] and r.group == "main"]   # S только здесь, не сохраняются
    longest = max(recs_all, key=lambda r: r.n_prompt_tokens)
    if longest not in sub:
        sub.append(longest)
    assert len({r.n_prompt_tokens for r in sub}) >= 3, "тривиально: промпты одной длины"
    p1 = check_padding_invariance(model, tok, sub, layers, positions, rel_l2_max=thr["check1_rel_l2_max"], cos_min=thr["check1_cos_min"],
                                  assert_sides=("right", "left") if thr.get("check1_assert_left") else ("right",))
    naive = check_naive_last_index_breaks_on_right_pad(model, tok, sub, layers[0])
    log.info("check1: %s ; naive: %s", p1, naive)
    record_numbers(nums_path, {"gate0.check1": p1, "gate0.check1.naive_last_right_pad": naive,
                               "notes.gate0.check1": "rel_l2 в bf16 ~1e-3..1e-2, 0.0 = сравнили с собой; naive должен быть ≫ порога"})
    assert naive["rel_l2_max"] >= thr["check1_naive_rel_l2_min"], "тест не кусается: наивный [-1] при right pad совпал с поштучным"

    p3 = check_hook_matches_hidden_states(model, tok, recs_all[0], layers, max_abs_diff=thr["check3_max_abs_diff"])
    record_numbers(nums_path, {"gate0.check3": p3})
    p6 = check_repeatability_and_memory(model, tok, sub, layers, positions)
    log.info("check6: %s", p6)
    record_numbers(nums_path, {"gate0.check6": p6})
    assert p6["hooks_clean"] and (p6["bitwise_repeatable"] or p6["cos_min"] >= thr["check6_cos_min"]), p6
    if not p6["bitwise_repeatable"]:
        log.warning("check6: не побитово (ядра недетерминированы), cos_min=%.6f", p6["cos_min"])

    acts = extract(model, tok, recs_keep, layers, positions, batch_size=8, padding_side="right")
    payload = acts_payload(recs_keep, acts, model_name, a.revision, str(dtype), "right")
    validate_acts_payload(payload)
    torch.save(payload, out / "gate0_acts.pt")
    for L in layers:
        for p in positions:
            n5 = check_norm_vs_length(acts[p][L], recs_keep)
            record_numbers(nums_path, {f"gate0.check5.{p}.L{L}": n5})
        diag = position_diagnostic({p: acts[p][L] for p in positions}, [r.condition_id for r in recs_keep], [r.env_prompt for r in recs_keep])
        log.info("toy-probe L=%d: %s", L, diag)
        record_numbers(nums_path, {f"gate0.toyprobe.L{L}": diag})
    log.info("position=%s зафиксирована в yaml (position_change_rule=%s); диагностика — secondary analysis",
             cfg.probe["position"], cfg.probe["position_change_rule"])

    g = check_batch_generation_matches_single(model, tok, [r for r in recs_all if r.condition_id == "deployment"][:4])
    record_numbers(nums_path, {"gate0.checkG": g})
    if g["prefix_match_min"] < thr["checkG_prefix_tokens_min"]:
        log.warning("checkG: greedy в left-pad батче расходится с поштучным на токене %d — left pad в generate неисправен", g["prefix_match_min"])

    sampling = assert_ready_for_generation(cfg, ("main",), temperature_override=a.temperature,
                                           generation_config=getattr(model, "generation_config", None), allow_smoke=a.fake_config)
    gen_recs = [r for r in recs_all if r.condition_id in ("deployment", "E1")]
    need = a.n_gen + 2 * a.batch_size
    gen_recs = (gen_recs * (need // max(1, len(gen_recs)) + 1))[:need]
    th = measure_throughput(model, tok, gen_recs, sampling, sampling["max_new_tokens"], batch_size=a.batch_size,
                            model_name=model_name, model_revision=a.revision)
    metas = th.pop("metas")
    atomic_write_text(out / "gate0_gens.jsonl", "\n".join(json.dumps(m, ensure_ascii=False) for m in metas) + "\n")
    record_numbers(nums_path, {"gate0.throughput": th, "gate0.sampling": sampling,
                               "gate0.mde_apriori": mde_apriori(thr["mde_sd_delta_apriori"], len(dps)),
                               "gate0.mde_apriori_sd_delta": thr["mde_sd_delta_apriori"], "gate0.mde_apriori_n_prompts": len(dps)})
    assert th["unknown_finish_count"] == 0, "есть генерации без eos короче max_new_tokens — eos-набор/остановка неисправны"
    if th["reopened_think_count"]:
        log.warning("check8: %d генераций содержат <think> в output", th["reopened_think_count"])
    if th["wallclock_est_hours_planned"] > thr["wallclock_hours_max"]:
        log.warning("wall-clock %.1f ч > %.0f ч → k=20 по §4.4", th["wallclock_est_hours_planned"], thr["wallclock_hours_max"])
    if th["truncated_frac"] > thr["truncated_frac_max"]:
        log.warning("обрыв %.2f > %.2f → поднять max_new_tokens ДО Gate 1", th["truncated_frac"], thr["truncated_frac_max"])
    print(json.dumps({k: v for k, v in json.loads(nums_path.read_text()).items() if k.startswith(("gate0.", "env."))}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
