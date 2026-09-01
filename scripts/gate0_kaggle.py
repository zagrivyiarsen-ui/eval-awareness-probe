"""Gate 0 на РЕАЛЬНЫХ DP в режиме Kaggle T4x2 (device_map=auto, без flash-attn). Пишет gate0.* в results/numbers.json.
    PYTHONPATH=. python scripts/gate0_kaggle.py --dps data/decision_points.json --temperature 0.7 --out results
Что делает сверх gate0_real_model.py:
  - prereg.conditions_hash: хеш ТОЛЬКО текстов условий + лексикона (заморожены сейчас); config_hash меняется при
    заполнении gate-параметров в Gate 0/1 — это не deviation, а правка текстов — deviation.
  - dtype-решение по ЗАПИСАННОМУ правилу: fp16 принимается, если (а) greedy-префикс fp16 == bf16 >= thr['checkG_prefix_tokens_min']
    токенов на 4 реальных промптах, (б) активации fp16 конечны, (в) check1 right pad в fp16 проходит. Иначе bf16.
  - скан батча {8,16,32}: tok/s декода (короткая генерация 384 токенов, после прогрева), OOM записывается.
  - таблица часов для k ∈ {25,20,15,10} × max_new_tokens ∈ {2048,3072,4096} против бюджета Kaggle.
  - toy-probe/check5 на реальных DP без S-условий (S только в check1/2), MDE при n=16 (на среду) и n=32 (пул).
Полная генерация 2048 токенов здесь НЕ делается (dry-run уже дал truncated_frac=0.75@2048 на fake-DP; длины на реальных DP
даст пилот Gate 1)."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.activations import (check_hook_matches_hidden_states, check_last_token_uniform, check_naive_last_index_breaks_on_right_pad,
                             check_norm_vs_length, check_padding_invariance, check_repeatability_and_memory,
                             default_candidate_layers, extract, get_layers)
from src.conditions import S_IDS, assert_ready_for_generation, conditions_hash, load_conditions
from src.contracts import acts_payload, dps_sha256, env_info, make_records, mde_apriori, record_numbers, validate_acts_payload
from src.probe_toy import position_diagnostic
from src.prompts import load_tokenizer, render_text
from src.throughput import eos_ids, generate_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gate0k")
KAGGLE_GPU_HOURS_PER_WEEK = 30.0
KAGGLE_SESSION_HOURS = 12.0



def load_model(name, revision, dtype):
    from transformers import AutoModelForCausalLM
    kw = dict(device_map="auto", revision=revision) if revision else dict(device_map="auto")
    try:
        m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", dtype=dtype, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", torch_dtype=dtype, **kw)
    m.eval()
    return m


def release():
    """Освобождает GPU после `del model` в вызывающем коде (del внутри функции удалял бы только локальную ссылку)."""
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def _sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


@torch.no_grad()
def throughput_scan(model, tok, recs, sampling, sizes, n_tokens=1024):   # Г-4: 384 завышало tok/s
    out = {}
    pool = recs
    for bs in sizes:
        batch = (pool * (bs // len(pool) + 1))[:bs]
        try:
            generate_ids(model, tok, batch[:min(4, bs)], sampling, 32)              # прогрев
            _sync(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            gens = generate_ids(model, tok, batch, sampling, n_tokens)
            _sync(); dt = time.perf_counter() - t0
            n_tok = sum(len(g) for g in gens)
            peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
            out[bs] = {"tok_per_s": n_tok / dt, "seconds": dt, "n_tokens": n_tok, "peak_mem_gb": peak / 2**30,
                       "finished_early_frac": sum(len(g) < n_tokens for g in gens) / len(gens)}
            log.info("bs=%d: %.1f tok/s, peak %.1f GB", bs, out[bs]["tok_per_s"], out[bs]["peak_mem_gb"])
        except torch.cuda.OutOfMemoryError:
            out[bs] = {"oom": True}
            log.warning("bs=%d: OOM", bs)
            gc.collect(); torch.cuda.empty_cache()
    return out


def batch_for_cap(scan: dict, cap: int, p_max: int, mem_limit_gb: float) -> dict:
    """Линейная модель памяти: peak(bs, L) ≈ W + bs·(P+L)·c по двум точкам скана (одинаковое L, разные bs) → max bs при L=cap.
    Ревью В(4): при 1024 токенах bs=24 уже OOM — экстраполяция на 4096 обязательна ДО пилота."""
    pts = sorted((int(k), v) for k, v in scan.items() if isinstance(v, dict) and "peak_mem_gb" in v)
    if len(pts) < 2:
        return {"batch_size_for_cap": max(1, pts[0][0] // 2) if pts else 4, "model": "insufficient points (fallback)"}
    (b1, v1), (b2, v2) = pts[0], pts[-1]
    L = int(v1["n_tokens"] / b1)
    slope = (v2["peak_mem_gb"] - v1["peak_mem_gb"]) / (b2 - b1)          # GB per batch row at length L
    c = slope / (p_max + L); W = v1["peak_mem_gb"] - b1 * slope
    bs = int((mem_limit_gb - W) / (c * (p_max + cap))) if c > 0 else b2
    bs = max(1, min(bs, 64)); bs = bs - bs % 2 if bs > 2 else bs
    return {"batch_size_for_cap": bs, "W_gb": W, "c_gb_per_token_row": c, "L_scan": L, "p_max": p_max, "cap": cap,
            "mem_limit_gb": mem_limit_gb, "model": "peak = W + bs*(P+L)*c"}


@torch.no_grad()
def verify_batch_at_cap(model, tok, recs, sampling, bs, cap):
    """Один реальный батч bs×cap (самые длинные промпты обеих сред): подтверждает, что не OOM, и даёт tok_s_eff для бюджета.
    OOM → bs//2 (до 2 попыток). Это ~10 мин GPU, но именно это число (tok/s при длине cap) и есть tok_s_eff правила урезания."""
    pool = sorted(recs, key=lambda r: -r.n_prompt_tokens)
    for attempt in range(3):
        batch = (pool * (bs // len(pool) + 1))[:bs]
        try:
            _sync(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
            gens = generate_ids(model, tok, batch, sampling, cap)
            _sync(); dt = time.perf_counter() - t0
            peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
            n_tok = sum(len(g) for g in gens)
            return {"batch_size": bs, "cap": cap, "seconds": dt, "tok_per_s": n_tok / dt, "peak_mem_gb": peak,
                    "batch_seconds": dt, "finished_early_frac": sum(len(g) < cap for g in gens) / len(gens), "attempts": attempt + 1}
        except torch.cuda.OutOfMemoryError:
            log.warning("verify_batch_at_cap: bs=%d cap=%d OOM → bs=%d", bs, cap, bs // 2)
            gc.collect(); torch.cuda.empty_cache(); bs = max(1, bs // 2)
    return {"batch_size": bs, "cap": cap, "oom": True}


@torch.no_grad()
def finite_check(model, tok, recs, layers):
    acts = extract(model, tok, recs, layers, ["last"], batch_size=8)
    return all(torch.isfinite(acts["last"][L]).all().item() for L in layers)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--dps", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--bs-scan", default="8,16")
    ap.add_argument("--cap", type=int, default=4096, help="cap пилота: batch подбирается так, чтобы bs×cap не OOM")
    ap.add_argument("--mem-limit-gb", type=float, default=13.6, help="≈0.93 × 14.6 GB T4")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--only-fp16", action="store_true", help="bf16-стадия уже в numbers.json: только greedy-эталон bf16 → fp16")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert "fake" not in out.name and "fake" not in Path(a.dps).name
    nums_path = out / "numbers.json"
    cfg = load_conditions(a.config, allow_unset_gate_params=True)
    thr = cfg.thresholds
    prereg = json.loads(nums_path.read_text(encoding="utf-8")) if nums_path.exists() else {}
    assert prereg.get("prereg.config_hash") == cfg.config_hash, "yaml изменён после verify_prereg — перезапустите verify"
    dsha = dps_sha256(a.dps)
    assert prereg.get("prereg.dps_sha256") == dsha, "DP изменены после verify_prereg"
    chash = conditions_hash(a.config)
    record_numbers(nums_path, {"prereg.conditions_hash": chash})

    tok = load_tokenizer(cfg)
    name, rev = cfg.tokenizer["name"], cfg.model["revision"]
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8"))
    conds_all = cfg.main + cfg.group("e2") + cfg.group("positive_control")
    recs_all = make_records(dps, conds_all, tok, cfg, dps_sha=dsha)
    recs_keep = [r for r in recs_all if r.condition_id not in S_IDS]
    envs = sorted({r.env for r in recs_all})
    n_per_env = {e: len({r.prompt_id for r in recs_all if r.env == e}) for e in envs}
    gen_pool = [r for r in recs_all if r.condition_id in ("deployment", "E1")]
    gen_pool = [r for pair in zip(gen_pool[:len(gen_pool) // 2], gen_pool[len(gen_pool) // 2:]) for r in pair]   # Г-4: чередование сред
    ref4 = [r for r in recs_all if r.condition_id == "deployment"][::max(1, len(dps) // 4)][:4]
    sub = [r for r in recs_all if r.prompt_id == dps[0]["prompt_id"] and r.group == "main"]
    longest = max(recs_all, key=lambda r: r.n_prompt_tokens)
    if longest not in sub: sub.append(longest)
    sizes = [int(x) for x in a.bs_scan.split(",")]
    greedy = {"temperature": 0.0, "do_sample": False, "top_p": None, "top_k": None, "repetition_penalty": None}

    # ---------------- bf16: эталон
    model = load_model(name, rev, torch.bfloat16)
    n_layers = len(get_layers(model))
    layers = cfg.probe.get("candidate_layers") or default_candidate_layers(n_layers)
    positions = list(cfg.probe.get("positions_diagnostic") or ["last"])
    sampling = assert_ready_for_generation(cfg, ("main",), temperature_override=a.temperature, generation_config=model.generation_config)
    if a.only_fp16:
        assert "gate0.check6.hooks_clean" in prereg, "--only-fp16 требует уже записанной bf16-стадии"
        ref_ids = generate_ids(model, tok, ref4, greedy, 32)
        scan_bf16 = {int(k.split(".")[3]): {"tok_per_s": v} for k, v in prereg.items()
                     if k.startswith("gate0.throughput_scan.bf16.") and k.endswith(".tok_per_s")}
    else:
        record_numbers(nums_path, {"gate0.fake_config": False, "gate0.model_name": name, "gate0.model_revision": rev, "gate0.model_class": type(model).__name__,
                                   "gate0.attn_impl": "sdpa", "gate0.n_layers": n_layers, "gate0.candidate_layers": layers, "gate0.positions": positions,
                                   "gate0.config_hash": cfg.config_hash, "gate0.dps_sha256": dsha, "gate0.conditions_hash": chash,
                                   "gate0.eos_ids": eos_ids(tok, model), "gate0.n_prompts_per_env": n_per_env, "gate0.kaggle_mode": True, **env_info()})
        preopens = render_text(recs_all[0].messages, tok, cfg).rstrip().endswith("<think>")
        record_numbers(nums_path, {"gate0.template_preopens_think": preopens}); assert preopens
        u = check_last_token_uniform(recs_all); record_numbers(nums_path, {"gate0.check2": u}); assert u["uniform"]
        p1 = check_padding_invariance(model, tok, sub, layers, positions, rel_l2_max=thr["check1_rel_l2_max"], cos_min=thr["check1_cos_min"])
        naive = check_naive_last_index_breaks_on_right_pad(model, tok, sub, layers[0])
        assert naive["rel_l2_max"] >= thr["check1_naive_rel_l2_min"]
        record_numbers(nums_path, {"gate0.check1": p1, "gate0.check1.naive_last_right_pad": naive})
        record_numbers(nums_path, {"gate0.check3": check_hook_matches_hidden_states(model, tok, recs_all[0], layers, max_abs_diff=thr["check3_max_abs_diff"])})
        p6 = check_repeatability_and_memory(model, tok, sub, layers, positions, n_repeats=2)
        assert p6["hooks_clean"] and (p6["bitwise_repeatable"] or p6["cos_min"] >= thr["check6_cos_min"])
        record_numbers(nums_path, {"gate0.check6": p6})
        acts = extract(model, tok, recs_keep, layers, positions, batch_size=8, padding_side="right")
        payload = acts_payload(recs_keep, acts, name, rev, "torch.bfloat16", "right"); validate_acts_payload(payload)
        torch.save(payload, out / "gate0_acts.pt")
        for L in layers:
            for p in positions:
                record_numbers(nums_path, {f"gate0.check5.{p}.L{L}": check_norm_vs_length(acts[p][L], recs_keep)})
            diag = position_diagnostic({p: acts[p][L] for p in positions}, [r.condition_id for r in recs_keep], [r.env_prompt for r in recs_keep])
            record_numbers(nums_path, {f"gate0.toyprobe.L{L}": diag})
            for e in envs:
                m = [i for i, r in enumerate(recs_keep) if r.env == e]
                d_e = position_diagnostic({p: acts[p][L][m] for p in positions}, [recs_keep[i].condition_id for i in m], [recs_keep[i].env_prompt for i in m])
                record_numbers(nums_path, {f"gate0.toyprobe_by_env.{e}.L{L}": {p: {k: v for k, v in d.items() if k != "heldout_aucs"} for p, d in d_e.items()}})
            log.info("toy-probe L=%d: %s", L, {p: d["auc_e2_vs_deploy_along_e1_dir"] for p, d in diag.items()})
        ref_ids = generate_ids(model, tok, ref4, greedy, 32)
        scan_bf16 = throughput_scan(model, tok, gen_pool, sampling, sizes)
        record_numbers(nums_path, {"gate0.throughput_scan.bf16": {str(k): v for k, v in scan_bf16.items()}, "gate0.sampling": sampling})
    record_numbers(nums_path, {f"gate0.greedy_ref_ids_bf16.row{i}": list(map(int, x)) for i, x in enumerate(ref_ids)})
    del model; release()

    # ---------------- fp16: кандидат
    dtype_decision = {"rule": "fp16 if greedy prefix vs bf16 >= checkG_prefix_tokens_min on 4 real prompts AND finite acts AND check1 right ok; else bf16"}
    scan_fp16 = {}
    if not a.no_fp16:
        m16 = load_model(name, rev, torch.float16)
        assert not any(p.device.type == "meta" for p in m16.parameters()), "fp16-модель не поместилась на GPU (bf16 не освобождена?)"
        ids16 = generate_ids(m16, tok, ref4, greedy, 32)
        match = []
        for x, y in zip(ref_ids, ids16):
            k = 0
            while k < min(len(x), len(y)) and x[k] == y[k]: k += 1
            match.append(k)
        fin = finite_check(m16, tok, sub, layers)
        try:
            p1_16 = check_padding_invariance(m16, tok, sub, layers, ["last"], rel_l2_max=thr["check1_rel_l2_max"], cos_min=thr["check1_cos_min"])
            c1_ok = True
        except AssertionError as e:
            p1_16, c1_ok = {"error": str(e)[:200]}, False
        scan_fp16 = throughput_scan(m16, tok, gen_pool, sampling, sizes)
        p_max = max(r.n_prompt_tokens for r in gen_pool)
        bfc = batch_for_cap(scan_fp16, a.cap, p_max, a.mem_limit_gb)
        log.info("batch_for_cap(fp16, cap=%d): %s", a.cap, bfc)
        vac = verify_batch_at_cap(m16, tok, gen_pool, sampling, bfc["batch_size_for_cap"], a.cap)
        log.info("verify_batch_at_cap(fp16): %s", vac)
        record_numbers(nums_path, {"gate0.batch_for_cap.fp16": bfc, "gate0.verify_at_cap.fp16": vac})
        ok = min(match) >= thr["checkG_prefix_tokens_min"] and fin and c1_ok
        dtype_decision.update({"prefix_match_fp16_vs_bf16": match, "acts_finite": fin, "check1_right_ok": c1_ok, "fp16_accepted": ok,
                               "decision": "float16" if ok else "bfloat16"})
        record_numbers(nums_path, {"gate0.check1_fp16": p1_16, "gate0.throughput_scan.fp16": {str(k): v for k, v in scan_fp16.items()}})
        del m16; release()
    else:
        dtype_decision.update({"decision": "bfloat16"})
    record_numbers(nums_path, {"gate0.dtype_decision": dtype_decision})

    # ---------------- бюджет
    chosen = "fp16" if dtype_decision.get("fp16_accepted") else "bf16"
    scan = scan_fp16 if chosen == "fp16" else scan_bf16
    nums_now = json.loads(nums_path.read_text(encoding="utf-8"))
    vac_key = f"gate0.verify_at_cap.{chosen}"
    if nums_now.get(vac_key + ".tok_per_s"):
        best_bs, best = int(nums_now[vac_key + ".batch_size"]), {"tok_per_s": float(nums_now[vac_key + ".tok_per_s"]), "source": f"verify_at_cap cap={a.cap}"}
    else:
        cands = [(k, v) for k, v in scan.items() if isinstance(v, dict) and "tok_per_s" in v]
        if not cands:
            log.error("скан батча пуст (все OOM) — бюджет не считается; уменьшите --bs-scan"); best_bs, best = 0, {"tok_per_s": float("nan"), "source": "none"}
        else:
            best_bs, best = max(cands, key=lambda kv: kv[1]["tok_per_s"]); best = dict(best, source="scan_1024")
    if not best_bs:
        record_numbers(nums_path, {"gate0.budget": {"dtype": chosen, "error": "no valid batch size"}}); return
    n_prompts = sum(n_per_env.values())
    table = {}
    # 2026-08-27 (ревью В): батч ждёт самую длинную генерацию → часы = ⌈n/bs⌉·bs·cap/tok_s (+overhead); «нижней границы» нет
    from src.analysis import hours_for
    for k in (25, 20, 15, 10):
        n_gen = n_prompts * 6 * k + 100
        for cap in (2048, 3072, 4096):
            h = hours_for(n_gen, cap, best["tok_per_s"], best_bs, overhead_per_session=1.0)
            table[f"k{k}.cap{cap}"] = {"n_gen": n_gen, "hours_lower": h, "hours_upper": h}
    n1 = max(n_per_env.values())
    for k in (15, 10):
        n_gen = n1 * 6 * k + 100
        h = hours_for(n_gen, 2048, best["tok_per_s"], best_bs, overhead_per_session=1.0)
        table[f"ONE_ENV.k{k}.cap2048"] = {"n_gen": n_gen, "hours_lower": h, "hours_upper": h}
    record_numbers(nums_path, {"gate0.budget": {"dtype": chosen, "best_batch_size": best_bs, "tok_per_s": best["tok_per_s"], "tok_per_s_source": best["source"], "cap_verified": a.cap,
                                                "kaggle_gpu_hours_per_week": KAGGLE_GPU_HOURS_PER_WEEK, "kaggle_session_hours": KAGGLE_SESSION_HOURS,
                                                "n_prompts": n_prompts, "table": table},
                               "gate0.mde_apriori_n16": mde_apriori(thr["mde_sd_delta_apriori"], 16),
                               "gate0.mde_apriori_n32_pooled": mde_apriori(thr["mde_sd_delta_apriori"], n_prompts)})
    print("\n" + "=" * 78 + f"\nБЮДЖЕТ: dtype={chosen} bs={best_bs} tok/s={best['tok_per_s']:.1f}  (Kaggle: {KAGGLE_GPU_HOURS_PER_WEEK:.0f} GPU-ч/нед, сессия {KAGGLE_SESSION_HOURS:.0f} ч)")
    print(f"{'вариант':>22} {'n_gen':>6} {'часы(min..max)':>18} {'сессий по 11ч':>14}")
    for key, v in table.items():
        print(f"{key:>22} {v['n_gen']:>6} {v['hours_lower']:>8.1f}..{v['hours_upper']:<8.1f} {math.ceil(v['hours_upper'] / 11):>10}")
    print("Правило урезания — configs/analysis.yaml cut_rule; batch_size пилота = gate0.budget.best_batch_size (bs×cap проверен без OOM).")
    print(json.dumps({k: v for k, v in json.loads(nums_path.read_text()).items() if k.startswith(("gate0.dtype", "gate0.throughput_scan"))}, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
