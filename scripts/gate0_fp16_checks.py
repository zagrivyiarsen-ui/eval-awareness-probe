"""Gate 0 — дополнение после ревью R-Gate0 (Г-1, Г-5). ~15 мин на T4×2. Пишет gate0.fp16.* в results/numbers.json.
    PYTHONPATH=. python scripts/gate0_fp16_checks.py --dps data/decision_points.json --out results --batch-size 24
Г-1: checkG в fp16 при bs=24 на РЕАЛЬНЫХ промптах (8 промптов × 3 копии), 64 токена, ассерт ≥ thresholds.checkG_prefix_tokens_min.
     Поштучный (не батчевый) greedy fp16 vs bf16 на 8 промптах — расхождение dtype, а не батча.
Г-5: overflow-скан fp16: max|h| по всем 32 слоям, всем токенам всех промпт-версий (порог thresholds.fp16_max_abs_h);
     KL(bf16‖fp16) следующего токена на позиции last по 32 deployment-промптам (медиана ≤ thresholds.fp16_kl_last_median_max);
     8 семплированных генераций по 1024 токена в fp16: unknown_finish == 0, reopened == 0, все id конечны."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gate0_kaggle import load_model, release
from src.activations import _pad_batch, get_layers
from src.conditions import assert_ready_for_generation, load_conditions
from src.contracts import assert_matches_prereg, classify_cot, dps_sha256, finish_reason, make_records, record_numbers
from src.prompts import load_tokenizer
from src.throughput import check_batch_generation_matches_single, eos_ids, generate_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gate0fp16")
GREEDY = {"temperature": 0.0, "do_sample": False, "top_p": None, "top_k": None, "repetition_penalty": None}


@torch.no_grad()
def last_logits(model, tok, recs, batch_size=1):
    """Логиты следующего токена на позиции last. Поштучно (без паддинга) с logits_to_keep=1: полный [B,T,V] в fp16 — гигабайты (OOM на T4)."""
    dev = next(model.parameters()).device
    outs = []
    for r in recs:
        ids = torch.tensor([r.input_ids], device=dev)
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        outs.append(out.logits[0, -1].float().cpu())
        del out
    return torch.stack(outs)


@torch.no_grad()
def overflow_scan(model, tok, recs, batch_size=8):
    mods = get_layers(model)
    mx = torch.zeros(len(mods)); nonfinite = 0
    def mk(L):
        def hook(_m, _i, out):
            nonlocal nonfinite
            h = out[0] if isinstance(out, (tuple, list)) else out
            hf = h.float()
            nonfinite += int((~torch.isfinite(hf)).sum().item())
            mx[L] = max(mx[L].item(), hf.abs().nan_to_num(0).max().item())
        return hook
    hs = [mods[L].register_forward_hook(mk(L)) for L in range(len(mods))]
    try:
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        dev = next(model.parameters()).device
        for b in range(0, len(recs), batch_size):
            chunk = recs[b:b + batch_size]
            ids, mask = _pad_batch(chunk, pad, "right", dev)
            model(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(1) - 1).clamp_min(0), use_cache=False, logits_to_keep=1)
    finally:
        for h in hs: h.remove()
    return {"max_abs_h_by_layer": [round(x, 1) for x in mx.tolist()], "max_abs_h": mx.max().item(), "argmax_layer": int(mx.argmax()),
            "nonfinite_count": nonfinite}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--dps", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args(argv)
    out = Path(a.out); nums = out / "numbers.json"
    cfg = load_conditions(a.config, allow_unset_gate_params=True)
    thr = cfg.thresholds
    dsha = dps_sha256(a.dps); assert_matches_prereg(cfg, nums, dsha)
    tok = load_tokenizer(cfg)
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8"))
    recs_all = make_records(dps, cfg.conditions, tok, cfg, dps_sha=dsha)
    dep = [r for r in recs_all if r.condition_id == "deployment"]
    n_env = len(dep) // 2
    ref8 = dep[:4] + dep[n_env:n_env + 4]                                   # 4 sb + 4 sn
    ref8[-1] = max(recs_all, key=lambda r: r.n_prompt_tokens)                # включая самый длинный
    name, rev = cfg.tokenizer["name"], cfg.model["revision"]

    # ---------------- bf16 эталон (поштучно)
    m = load_model(name, rev, torch.bfloat16)
    sampling = assert_ready_for_generation(cfg, ("main",), temperature_override=a.temperature, generation_config=m.generation_config)
    ref_single = [generate_ids(m, tok, [r], GREEDY, 32)[0] for r in ref8]
    lg_bf16 = last_logits(m, tok, dep)
    del m; release()

    # ---------------- fp16
    m16 = load_model(name, rev, torch.float16)
    assert not any(p.device.type == "meta" for p in m16.parameters())
    ov = overflow_scan(m16, tok, recs_all)
    lg_fp16 = last_logits(m16, tok, dep)
    kl = F.kl_div(F.log_softmax(lg_fp16, -1), F.log_softmax(lg_bf16, -1), log_target=True, reduction="none").sum(-1)
    kl_stats = {"median": kl.median().item(), "max": kl.max().item(), "mean": kl.mean().item(),
                "argmax_top1_agree": (lg_fp16.argmax(-1) == lg_bf16.argmax(-1)).float().mean().item()}
    single16 = [generate_ids(m16, tok, [r], GREEDY, 32)[0] for r in ref8]
    match = []
    for x, y in zip(ref_single, single16):
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]: k += 1
        match.append(k)
    # Г-1: checkG fp16 bs=24, реальные промпты ×3
    batch = (ref8 * 3)[:a.batch_size]
    g = check_batch_generation_matches_single(m16, tok, batch, n_tokens=64)
    # семплированная генерация 1024 в fp16
    eos = eos_ids(tok, m16)
    gens = generate_ids(m16, tok, ref8, sampling, 1024)
    cats = []
    for r, gg in zip(ref8, gens):
        fr = finish_reason(gg, 1024, eos)
        txt = tok.decode(gg, skip_special_tokens=False)
        c = classify_cot(txt, fr); cats.append({"finish": fr, "cat": c["cot_category"], "reopened": c["reopened_think"], "len": len(gg)})
    del m16; release()

    res = {"gate0.fp16.overflow": ov, "gate0.fp16.kl_last_bf16_vs_fp16": kl_stats, "gate0.fp16.greedy_prefix_single_match": match,
           "gate0.fp16.checkG_bs24": {k: v for k, v in g.items()}, "gate0.fp16.sample1024": {
               "unknown_finish": sum(c["finish"] == "unknown" for c in cats), "reopened": sum(c["reopened"] > 0 for c in cats),
               "closed_frac": sum(c["cat"] in ("ok", "empty_answer") for c in cats) / len(cats), "len_mean": sum(c["len"] for c in cats) / len(cats)},
           "gate0.fp16.batch_size": a.batch_size}
    record_numbers(nums, res)
    ok = {"overflow": ov["max_abs_h"] < thr["fp16_max_abs_h"] and ov["nonfinite_count"] == 0,
          "kl_median": kl_stats["median"] <= thr["fp16_kl_last_median_max"],
          "prefix_single": min(match) >= thr["checkG_prefix_tokens_min"],
          "checkG_fp16_bs24": g["prefix_match_min"] >= thr["checkG_prefix_tokens_min"],
          "sample1024": res["gate0.fp16.sample1024"]["unknown_finish"] == 0 and res["gate0.fp16.sample1024"]["reopened"] == 0}
    record_numbers(nums, {"gate0.fp16.verdict": ok, "gate0.fp16.accepted": all(ok.values())})
    print(json.dumps({k: v for k, v in res.items() if "by_layer" not in k}, indent=1, default=str)[:3000])
    print("\nВЕРДИКТ fp16:", ok, "\n→", "fp16 ПРИНЯТ" if all(ok.values()) else "fp16 НЕ ПРИНЯТ → dtype=bfloat16 (бюджет ×4), правило записано")
    if not ok["checkG_fp16_bs24"]:
        print("!! left pad в generate при bs=24 неисправен — генерация только поштучно/меньшим батчем; пересчитать бюджет")


if __name__ == "__main__":
    main()
