"""Модуль c: residual stream на промптовых токенах. Prefill — RIGHT pad по умолчанию (каузальность:
пад после реальных токенов не влияет ни на attention, ни на рекурренцию DeltaNet). Индексы по маске."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from src.contracts import PromptRecord

log = logging.getLogger(__name__)


def get_layers(model) -> torch.nn.ModuleList:
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers", "transformer.h"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        except AttributeError:
            continue
    raise RuntimeError("cannot find decoder layer list; set explicitly for this architecture")


def default_candidate_layers(n_layers: int) -> list[int]:
    c = round(2 * n_layers / 3)
    step = max(1, n_layers // 12)
    return sorted({max(0, min(n_layers - 2, c - step)), min(n_layers - 2, c), min(n_layers - 2, c + step)})


_OUR_HOOK_IDS: set[int] = set()     # id всех когда-либо навешанных нами хуков; transformers 5.x вешает свои — они не наши


@contextmanager
def residual_hooks(model, layers: Sequence[int]):
    store: dict[int, torch.Tensor] = {}
    mods = get_layers(model)
    handles = []

    def mk(L):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            assert torch.is_tensor(h) and h.dim() == 3, f"layer {L}: unexpected output {type(out)}"
            assert L not in store, f"layer {L}: hook fired twice in one forward"
            store[L] = h
        return hook

    try:
        for L in layers:
            assert 0 <= L < len(mods), f"layer {L} out of range [0,{len(mods)})"
            h = mods[L].register_forward_hook(mk(L))
            handles.append(h); _OUR_HOOK_IDS.add(h.id)
        yield store
    finally:
        for h in handles:
            h.remove()
        store.clear()


def hooks_are_clean(model, layers: Sequence[int]) -> bool:
    """Ни один из НАШИХ хуков не остался. Чужие (transformers/accelerate) не считаются — их число логируется."""
    mods = get_layers(model)
    foreign = sum(len(mods[L]._forward_hooks) for L in layers)
    if foreign:
        log.info("foreign forward hooks on candidate layers: %d (transformers/accelerate) — not ours", foreign)
    return not any(hid in mods[L]._forward_hooks for L in layers for hid in _OUR_HOOK_IDS)


def forward_no_logits(model, **kw):
    """Prefill для активаций: логиты не нужны. lm_head на [B,T,248k] в bf16 — гигабайты (OOM на T4);
    logits_to_keep=1 оставляет только последний токен. Активации residual stream от этого не зависят."""
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError:
        return model(**kw)


def _pad_batch(recs: Sequence[PromptRecord], pad_id: int, side: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    assert side in ("left", "right")
    T = max(r.n_prompt_tokens for r in recs)
    ids = torch.full((len(recs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(recs), T), dtype=torch.long)
    for i, r in enumerate(recs):
        n = r.n_prompt_tokens
        sl = slice(T - n, T) if side == "left" else slice(0, n)
        ids[i, sl] = torch.tensor(r.input_ids)
        mask[i, sl] = 1
    return ids.to(device), mask.to(device)


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Как в generate: позиции считаются по реальным токенам, пад слева не сдвигает RoPE
    (без этого left pad в bf16 давал rel_l2≈0.026 на 0.6B — не баг индекса, а сдвиг углов RoPE)."""
    return (mask.cumsum(dim=1) - 1).clamp_min(0)


def gather_positions(h: torch.Tensor, mask: torch.Tensor, recs: Sequence[PromptRecord], position: str) -> torch.Tensor:
    B, T, _ = h.shape
    assert mask.shape == (B, T)
    mask = mask.to(h.device)                                 # device_map=auto: слой может жить не на карте входа
    offset = mask.argmax(dim=1)
    n_real = mask.sum(dim=1)
    for i, r in enumerate(recs):
        assert int(n_real[i]) == r.n_prompt_tokens, f"row {i}: mask {int(n_real[i])} != record {r.n_prompt_tokens}"
    if position == "mean_user":
        return torch.stack([h[i, int(offset[i]) + r.user_span[0]: int(offset[i]) + r.user_span[1]].float().mean(dim=0)
                            for i, r in enumerate(recs)])
    rel = torch.tensor([r.position_index(position) for r in recs], device=h.device)
    idx = offset + rel
    assert bool((idx < T).all()) and bool((idx >= 0).all())
    return h[torch.arange(B, device=h.device), idx].float()


@torch.no_grad()
def extract(model, tokenizer, recs: Sequence[PromptRecord], layers: Sequence[int], positions: Sequence[str],
            batch_size: int = 8, padding_side: str = "right") -> dict[str, dict[int, torch.Tensor]]:
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    acc: dict[str, dict[int, list[torch.Tensor]]] = {p: {L: [] for L in layers} for p in positions}
    for b in range(0, len(recs), batch_size):
        chunk = recs[b:b + batch_size]
        ids, mask = _pad_batch(chunk, pad_id, padding_side, device)
        with residual_hooks(model, layers) as store:
            forward_no_logits(model, input_ids=ids, attention_mask=mask, position_ids=position_ids_from_mask(mask), use_cache=False)
            for L in layers:
                for p in positions:
                    acc[p][L].append(gather_positions(store[L], mask, chunk, p).cpu())
        del ids, mask
    assert hooks_are_clean(model, layers)
    return {p: {L: torch.cat(v) for L, v in d.items()} for p, d in acc.items()}


def row_metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(max_i ‖a_i−b_i‖/‖b_i‖, min_i cos(a_i,b_i)) — метрики ревью 2.2 (поэлементный max/‖h‖ пропускал баг индекса)."""
    rel = ((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-6)).max().item()
    cos = F.cosine_similarity(a, b, dim=1).min().item()
    return rel, cos


@torch.no_grad()
def check_padding_invariance(model, tokenizer, recs, layers, positions, rel_l2_max=1e-2, cos_min=0.999,
                             assert_sides: Sequence[str] = ("right",)) -> dict[str, Any]:
    """Проверка 1: батч == batch_size=1 поштучно. Right pad (единственный, что использует prefill) — ассерт.
    Left pad — только измеряется: на Qwen3.5-9B (DeltaNet + sdpa с маской) он даёт bf16-шум rel_l2≈0.016 при cos≈0.9999
    (dry-run 2026-08-26); ошибка индекса выглядит иначе (rel_l2≈1, cos≈0.45). Корректность left pad в generate — checkG."""
    single = extract(model, tokenizer, recs, layers, positions, batch_size=1)
    out: dict[str, Any] = {}
    for side in ("right", "left"):
        batched = extract(model, tokenizer, recs, layers, positions, batch_size=len(recs), padding_side=side)
        for p in positions:
            for L in layers:
                rel, cos = row_metrics(batched[p][L], single[p][L])
                out[f"{side}.{p}.L{L}.rel_l2_max"] = rel
                out[f"{side}.{p}.L{L}.cos_min"] = cos
                ok = rel <= rel_l2_max and cos >= cos_min
                msg = f"pad={side} pos={p} L={L}: rel_l2={rel:.3g} (max {rel_l2_max}) cos={cos:.5f} (min {cos_min})"
                if side in assert_sides:
                    assert ok, msg + " — индекс/паддинг/маска"
                elif not ok:
                    log.warning("%s — выше порога; %s pad для prefill не используется, число репортится", msg, side)
    return out


@torch.no_grad()
def check_naive_last_index_breaks_on_right_pad(model, tokenizer, recs, layer) -> dict[str, float]:
    single = extract(model, tokenizer, recs, [layer], ["last"], batch_size=1)["last"][layer]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    ids, mask = _pad_batch(recs, pad_id, "right", device)
    with residual_hooks(model, [layer]) as store:
        forward_no_logits(model, input_ids=ids, attention_mask=mask, position_ids=position_ids_from_mask(mask), use_cache=False)
        naive = store[layer][:, -1].float().cpu()
    rel, cos = row_metrics(naive, single)
    return {"rel_l2_max": rel, "cos_min": cos}


def check_last_token_uniform(recs: Sequence[PromptRecord]) -> dict[str, Any]:
    ids = {r.last_token_id for r in recs}
    strs = {r.last_token_str for r in recs}
    pre = {r.input_ids[-2] for r in recs}
    return {"n_records": len(recs), "distinct_last_token_ids": sorted(ids), "distinct_last_token_strs": sorted(strs),
            "distinct_prelast_token_ids": sorted(pre), "uniform": len(ids) == 1 and len(pre) == 1}


@torch.no_grad()
def check_hook_matches_hidden_states(model, tokenizer, rec: PromptRecord, layers, max_abs_diff=0.0) -> dict[str, Any]:
    model.eval()
    n = len(get_layers(model))
    for L in layers:
        assert L < n - 1, f"layer {L}: последний блок — hidden_states[-1] после финальной нормы"
    device = next(model.parameters()).device
    ids = torch.tensor([rec.input_ids], device=device)
    with residual_hooks(model, layers) as store:
        out = forward_no_logits(model, input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = getattr(out, "hidden_states", None)
        assert hs is not None, "модель не отдаёт output_hidden_states — сравнить хук с ручным прогоном layers[:L+1]"
        res: dict[str, Any] = {"n_hidden_states": len(hs)}
        for L in layers:
            d = (store[L].float().cpu() - hs[L + 1].float().cpu()).abs().max().item()   # device_map=auto: разные карты
            res[f"L{L}.max_abs_diff_vs_hidden_states"] = d
            assert d <= max_abs_diff, f"layer {L}: hook != hidden_states[{L + 1}] (Δ={d})"
    return res


def check_norm_vs_length(acts: torch.Tensor, recs: Sequence[PromptRecord]) -> dict[str, float]:
    norms = acts.norm(dim=1)
    lens = torch.tensor([float(r.n_prompt_tokens) for r in recs])
    base = {"norm_mean": norms.mean().item(), "norm_min": norms.min().item(), "norm_max": norms.max().item()}
    if len(recs) < 3 or norms.std() == 0 or lens.std() == 0:
        return {"pearson_r": float("nan"), **base}
    return {"pearson_r": torch.corrcoef(torch.stack([norms, lens]))[0, 1].item(), **base}


@torch.no_grad()
def check_repeatability_and_memory(model, tokenizer, recs, layers, positions, n_repeats=5) -> dict[str, Any]:
    a = extract(model, tokenizer, recs, layers, positions)
    b = extract(model, tokenizer, recs, layers, positions)
    same = all(torch.equal(a[p][L], b[p][L]) for p in positions for L in layers)
    rel = max(row_metrics(a[p][L], b[p][L])[0] for p in positions for L in layers)
    cos = min(row_metrics(a[p][L], b[p][L])[1] for p in positions for L in layers)
    mem = {}
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        m0 = torch.cuda.memory_allocated()
        for _ in range(n_repeats):
            extract(model, tokenizer, recs, layers, positions)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        m1 = torch.cuda.memory_allocated()
        mem = {"cuda_mem_before": m0, "cuda_mem_after": m1, "cuda_mem_growth_bytes": m1 - m0}
    return {"bitwise_repeatable": same, "rel_l2_max": rel, "cos_min": cos, "hooks_clean": hooks_are_clean(model, layers), **mem}
