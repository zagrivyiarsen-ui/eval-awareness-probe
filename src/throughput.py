"""Модуль d: генерация (left pad), throughput с прогревом, wall-clock по времени батча, правило обрыва, проверка G."""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Sequence

import torch

from src.activations import _pad_batch
from src.contracts import PromptRecord, finish_reason, generation_meta

log = logging.getLogger(__name__)


def eos_ids(tokenizer, model=None) -> list[int]:
    """tokenizer.eos ∪ <|im_end|> ∪ model.generation_config.eos_token_id (ревью 2.1: иначе <|endoftext|> не останавливает)."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        ids.add(int(im_end))
    gc = getattr(model, "generation_config", None) if model is not None else None
    e = getattr(gc, "eos_token_id", None)
    for t in ([e] if isinstance(e, int) else (e or [])):
        ids.add(int(t))
    assert im_end in ids
    return sorted(ids)


def _sync(model):
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


@torch.no_grad()
def generate_ids(model, tokenizer, recs: Sequence[PromptRecord], sampling: dict[str, Any], max_new_tokens: int) -> list[list[int]]:
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    ids, mask = _pad_batch(recs, pad_id, "left", device)
    eos = eos_ids(tokenizer, model)
    kw: dict[str, Any] = dict(max_new_tokens=max_new_tokens, do_sample=bool(sampling["do_sample"]), pad_token_id=pad_id, eos_token_id=eos)
    if sampling["do_sample"]:
        kw["temperature"] = float(sampling["temperature"])
        # Г-2: все sampling-ключи передаются ЯВНО (числа из assert_ready_for_generation), generate() не подставляет свои дефолты
        for k in ("top_p", "top_k", "repetition_penalty"):
            assert sampling.get(k) is not None, f"sampling[{k}] must be resolved to a number before generation"
        kw["top_p"] = float(sampling["top_p"]); kw["top_k"] = int(sampling["top_k"])
        kw["repetition_penalty"] = float(sampling["repetition_penalty"])
    else:
        kw["temperature"] = None; kw["top_p"] = None; kw["top_k"] = None
    out = model.generate(input_ids=ids, attention_mask=mask, **kw)
    T = ids.shape[1]
    gens = []
    for i in range(len(recs)):
        gen = out[i, T:].tolist()
        cut = len(gen)
        for j, t in enumerate(gen):
            if t in eos:
                cut = j + 1
                break
        gens.append(gen[:cut])
    return gens


@torch.no_grad()
def generate_batch(model, tokenizer, recs, sampling, max_new_tokens, model_name="?", model_revision=None,
                   sample_idx: int | Sequence[int] = 0) -> list[dict[str, Any]]:
    gens = generate_ids(model, tokenizer, recs, sampling, max_new_tokens)
    eos = eos_ids(tokenizer, model)
    strip = {tokenizer.eos_token, "<|im_end|>", "<|endoftext|>", tokenizer.pad_token} - {None}
    sidx = [sample_idx] * len(recs) if isinstance(sample_idx, int) else list(sample_idx)
    metas = []
    for r, gen, si in zip(recs, gens, sidx):
        text = tokenizer.decode(gen, skip_special_tokens=False)   # </think> может быть special — не терять
        for s in strip:
            text = text.replace(s, "")
        metas.append(generation_meta(r, si, sampling, text, gen, finish_reason(gen, max_new_tokens, eos), model_name, model_revision))
    return metas


@torch.no_grad()
def measure_throughput(model, tokenizer, recs: Sequence[PromptRecord], sampling, max_new_tokens, batch_size: int = 4,
                       n_warmup_batches: int = 2, total_generations_planned: int = 4900, **meta_kw) -> dict[str, Any]:
    """wall-clock = ceil(planned/batch_size) × mean(время батча) — батч ждёт самую длинную генерацию,
    средний tok/s это занижает/завышает (ревью 2.5). Оценка переносима только при том же batch_size."""
    batches = [recs[i:i + batch_size] for i in range(0, len(recs), batch_size)]
    assert len(batches) > n_warmup_batches
    for b in batches[:n_warmup_batches]:
        generate_batch(model, tokenizer, b, sampling, min(max_new_tokens, 64), **meta_kw)
    metas: list[dict[str, Any]] = []
    batch_times = []
    counter: dict[str, int] = {}
    for b in batches[n_warmup_batches:]:
        sidx = [counter.setdefault(r.prompt_key, 0) for r in b]
        for r in b:
            counter[r.prompt_key] += 1
        _sync(model); t0 = time.perf_counter()
        metas += generate_batch(model, tokenizer, b, sampling, max_new_tokens, sample_idx=sidx, **meta_kw)
        _sync(model); batch_times.append(time.perf_counter() - t0)
    dt = sum(batch_times)
    lens = [m["n_generated_tokens"] for m in metas]
    n_tok = sum(lens)
    cats = {c: sum(m["cot_category"] == c for m in metas) for c in ("ok", "truncated", "no_close_eos", "empty_answer", "unknown_finish")}
    by_cond: dict[str, dict[str, int]] = {}
    for m in metas:
        d = by_cond.setdefault(m["condition_id"], {"n": 0, "truncated": 0, "reopened": 0})
        d["n"] += 1; d["truncated"] += m["cot_category"] == "truncated"; d["reopened"] += m["reopened_think"] > 0
    tok_s = n_tok / dt if dt > 0 else float("nan")
    mean_batch = sum(batch_times) / len(batch_times)
    return {"n_measured": len(metas), "batch_size": batch_size, "n_warmup_batches": n_warmup_batches, "wall_s": dt,
            "tok_per_s": tok_s, "batch_seconds_mean": mean_batch, "batch_seconds_max": max(batch_times),
            "gen_len_mean": n_tok / len(lens), "gen_len_max": max(lens), "gen_len_min": min(lens),
            "truncated_frac": cats["truncated"] / len(metas), "unknown_finish_count": cats["unknown_finish"],
            "cot_categories": cats,
            "truncated_by_condition": {c: d["truncated"] / d["n"] for c, d in by_cond.items()},
            "reopened_by_condition": {c: d["reopened"] / d["n"] for c, d in by_cond.items()},
            "reopened_think_count": sum(m["reopened_think"] > 0 for m in metas),
            "wallclock_est_hours_planned": math.ceil(total_generations_planned / batch_size) * mean_batch / 3600,
            "wallclock_est_hours_tok_s_based": total_generations_planned * (n_tok / len(lens)) / tok_s / 3600 if tok_s == tok_s else float("nan"),
            "wallclock_valid_for_batch_size": batch_size, "metas": metas}


@torch.no_grad()
def check_batch_generation_matches_single(model, tokenizer, recs: Sequence[PromptRecord], n_tokens: int = 32) -> dict[str, Any]:
    """Проверка G: greedy в left-pad батче == greedy поштучно (первые n_tokens). Ловит неисправный left pad
    (position_ids / маска рекуррентных слоёв) в generate, которого prefill (right pad) не видит."""
    greedy = {"temperature": 0.0, "do_sample": False, "top_p": None, "top_k": None, "repetition_penalty": None}
    batched = generate_ids(model, tokenizer, recs, greedy, n_tokens)
    matches = []
    for r, g in zip(recs, batched):
        s = generate_ids(model, tokenizer, [r], greedy, n_tokens)[0]
        k = 0
        while k < min(len(s), len(g)) and s[k] == g[k]:
            k += 1
        matches.append(k)
    return {"n_tokens": n_tokens, "prefix_match_min": min(matches), "prefix_match_per_row": matches}
