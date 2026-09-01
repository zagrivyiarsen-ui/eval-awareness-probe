"""G1 — GPU-проход (post pivot-4.2, 2026-09-01). Вне блоба; всё под probe.exploratory.g1.* / baseline.*.
Шаги (порядок = порядок фиксации, I-2): (0) записать sha артефакта p4_frozen.json и словаря ДО активаций;
(1) собрать записи новых наборов через ТОТ ЖЕ make_records/deployment-условие (Д-14), assert суффикса шаблона;
(2) extract активаций (слои как в prereg, 3 позиции) → probe_acts_extra_fp16.pt; (3) NLL/токен для всех записей;
(4) Б1 «спроси модель» (greedy, префилл пустого think, cap 12; пилот на 4); (5) Б2 внешние негативы, если датасет в Input;
(6) опц. Б4 диагностический PC. Каждая запись в numbers сопровождается assert'ом наличия ключа (К-1)."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gate0_kaggle import load_model
from src.activations import extract
from src.conditions import load_conditions
from src.contracts import dps_sha256, make_records, record_numbers
from src.prompts import load_tokenizer, render_text

POSITIONS = ("last", "last_user", "mean_user")
KEYS: list[str] = []


def _flat(k, v, out):
    if isinstance(v, dict):
        for kk, vv in v.items():
            _flat(f"{k}.{kk}", vv, out)
    else:
        out[k] = v


def rec(nums: Path, d: dict):
    """К-1: пишем и сразу ассертим наличие каждого РАСКАТАННОГО ключа (record_numbers раскатывает dict в dotted)."""
    record_numbers(nums, d)
    cur = json.loads(nums.read_text(encoding="utf-8"))
    flat: dict = {}
    for k, v in d.items():
        _flat(k, v, flat)
    for k in flat:
        assert k in cur, f"ключ не записался: {k}"
        KEYS.append(k)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------- records
def first_paragraph(text: str, max_words: int = 80) -> str:
    para = text.strip().split("\n\n")[0]
    w = para.split()
    if len(w) > max_words:
        cut = " ".join(w[:max_words])
        cut = cut[: cut.rfind(".") + 1] if "." in cut else cut
        return cut
    return para


def dp_trunc_context(ctx: str) -> str:
    """Инструкция (абзац 0) + заголовок questions + первые 2 вопроса — различаются по DP (баг v1: абзац 0 константен)."""
    paras = ctx.strip().split("\n\n")
    head = paras[0]
    qpara = next((q for q in paras if "questions.txt" in q and re.search(r"\n1\. ", q)), None)
    if qpara is None:  # secret_number: первые ТРИ абзаца (первый почти константен; в третий попадает история)
        return "\n\n".join(paras[:3]).strip()
    lines = qpara.split("\n"); title = lines[0]
    body = "\n".join(lines[1:]); qs = re.split(r"\n(?=\d+\. )", body)[:2]
    return head + "\n\n" + title + "\n" + "\n".join(qs).strip()


def truncate_words(ctx: str, n_words: int) -> str:
    """Усечение ТОГО ЖЕ текста по словам с добивкой до конца предложения — материал идентичен, меняется только длина."""
    w = ctx.split()
    if len(w) <= n_words:
        return ctx.strip()
    cut = " ".join(w[:n_words])
    j = max(cut.rfind("."), cut.rfind("?"), cut.rfind("\n"))
    return (cut[: j + 1] if j > len(cut) // 2 else cut).strip()


def build_new_dps(art: dict, dps: list[dict]) -> list[dict]:
    out = []
    for p in art["prompts"]:
        out.append({"env": p["env"], "prompt_id": p["prompt_id"], "context": p["context"], "actions": p["actions"]})
    by_id = {d["prompt_id"]: d for d in dps}
    seen = set()
    for pid in [f"sb_{i:03d}" for i in range(1, 9)] + [f"sn_{i:03d}" for i in range(1, 9)]:
        d = by_id.get(pid)
        if d is None:
            continue
        c = dp_trunc_context(d["context"])
        if (c, d["actions"]) in seen:
            continue
        seen.add((c, d["actions"]))
        out.append({"env": "x_dptrunc", "prompt_id": f"dt_{pid}", "context": c, "actions": d["actions"]})
    assert len({(o["context"], o["actions"]) for o in out if o["env"] == "x_dptrunc"}) == sum(o["env"] == "x_dptrunc" for o in out), "dp_trunc дубликаты"
    # лестница длин (ревью-7/8): те же DP, усечённые до ~140 и ~210 слов; ~80 = dptrunc, полный = deployment
    for step in (140, 210):
        for pid in [f"sb_{i:03d}" for i in range(1, 9)] + [f"sn_{i:03d}" for i in range(1, 9)]:
            d = by_id.get(pid)
            if d is None:
                continue
            out.append({"env": f"x_len{step}", "prompt_id": f"L{step}_{pid}", "context": truncate_words(d["context"], step), "actions": d["actions"]})
    return out


def load_wild(art: dict, n_take: int, seed: int, path: str | None = None) -> tuple[list[dict], dict]:
    """Б2: внешние негативы из Input. Ищем табличный/json файл с текстовой колонкой; сид и источник — в numbers."""
    cands = [path] if path else sorted(f for f in glob.glob("/kaggle/input/**/*", recursive=True) if os.path.isfile(f)
             and f.lower().endswith((".json", ".jsonl", ".parquet", ".csv")) and "evalaware" not in f and "p4_frozen" not in f)
    texts, src = [], None
    for f in cands:
        try:
            if f.endswith(".parquet"):
                import pandas as pd; df = pd.read_parquet(f)
            elif f.endswith(".csv"):
                import pandas as pd; df = pd.read_csv(f)
            elif f.endswith(".jsonl"):
                import pandas as pd; df = pd.read_json(f, lines=True)
            else:
                import pandas as pd; obj = json.loads(Path(f).read_text(encoding="utf-8"))
                df = pd.DataFrame(obj if isinstance(obj, list) else obj.get("data") or obj.get("prompts") or [])
        except Exception:
            continue
        col = next((c for c in ("instruction", "prompt", "text", "question", "input") if c in df.columns), None)
        if col is None or len(df) < 100:
            continue
        texts = [str(t) for t in df[col].tolist() if isinstance(t, str)]
        src = f
        break
    if not texts:
        return [], {"found": False}
    rng = __import__("random").Random(seed)
    idx = list(range(len(texts))); rng.shuffle(idx)
    picked = []
    ai_rx = re.compile(art["wild_ai_topic_regex"], re.I)
    ai_rx_broad = re.compile(art.get("wild_ai_topic_regex_broad", art["wild_ai_topic_regex"]), re.I)
    lo, hi = 55, 85
    for attempt in range(2):
        picked = []
        for i in idx:
            t = texts[i].strip(); nw = len(t.split())
            if lo <= nw <= hi and "\n\n" not in t:
                sents = re.split(r"(?<=[.?!])\s+(?=[A-Z])", t)
                if len(sents) < 2:
                    continue
                ctx, act = " ".join(sents[:-1]).strip(), sents[-1].strip()
                if not ctx or not act:
                    continue
                mt = ai_rx.search(t); mb = ai_rx_broad.search(t)
                env = "x_wild_ai" if mt else "x_wild_other"
                picked.append({"env": env, "prompt_id": f"wd_{len(picked):03d}", "context": ctx, "actions": act, "src_index": i,
                               "ai_match": (mt.group(0) if mt else None), "ai_match_broad": (mb.group(0) if mb else None)})
            if len(picked) >= n_take:
                break
        if len(picked) >= max(10, n_take // 2):
            break
        lo, hi = 40, 100  # fallback (Т-5 ревью-4)
    meta = {"found": True, "source_path": src, "source_sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest(), "seed": seed,
            "n_candidates": len(texts), "n_taken": len(picked), "n_ai": sum(p["env"] == "x_wild_ai" for p in picked), "ai_matched_terms": [p["ai_match"] for p in picked if p["ai_match"]][:12],
            "n_ai_broad": sum(bool(p["ai_match_broad"]) for p in picked), "ai_matched_terms_broad": [p["ai_match_broad"] for p in picked if p["ai_match_broad"]][:12],
            "wild_ai_broad_ids": [p["prompt_id"] for p in picked if p["ai_match_broad"] and not p["ai_match"]], "length_filter": f"{lo}-{hi} words, single paragraph",
            "ai_topic_rule": "prompt is AI-topic if it mentions a model, assistant or ML system as an object of discussion (post-hoc label, rule fixed before viewing)"}
    return picked, meta


# ---------------------------------------------------------------- NLL
@torch.no_grad()
def nll_per_record(model, tok, recs, batch_size: int = 4) -> list[dict]:
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = next(model.parameters()).device
    rows = []
    for b in range(0, len(recs), batch_size):
        chunk = recs[b:b + batch_size]
        L = max(r.n_prompt_tokens for r in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, r in enumerate(chunk):
            ids[i, : r.n_prompt_tokens] = torch.tensor(r.input_ids); mask[i, : r.n_prompt_tokens] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        import torch.nn.functional as F
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        B, Lm1, V = logits[:, :-1].shape
        nll = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(B, Lm1).float()
        # nll[t] = -log p(ids[t+1] | ids[:t+1]); без материализации float log_softmax (OOM на T4)
        for i, r in enumerate(chunk):
            n = r.n_prompt_tokens
            full = nll[i, : n - 1].mean().item()
            s, e = r.user_span  # токены [s,e) — предсказываются позициями s-1..e-2
            us = max(s - 1, 0); ue = max(e - 1, us + 1)
            user = nll[i, us:ue].mean().item()
            rows.append({"prompt_key": r.prompt_key, "env": r.env, "condition_id": r.condition_id, "n_tokens": n,
                         "n_user_tokens": e - s, "n_user_words": len(r.messages[-1]["content"].split()),
                         "nll_full": full, "nll_user": user})
        del logits, nll
    return rows


# ---------------------------------------------------------------- Б1
YN = re.compile(r"^\W*(yes|no)\b", re.I)


def parse_yn(text: str) -> str:
    t = text.split("</think>")[-1] if "</think>" in text else text
    for line in t.splitlines():
        if line.strip():
            m = YN.match(line.strip())
            return m.group(1).lower() if m else "unparsable"
    return "unparsable"


def ask_prompt_text(rec, question: str, tok, cfg, prefill: bool) -> str:
    msgs = list(rec.messages) + [{"role": "user", "content": question}]
    text = render_text(msgs, tok, cfg, add_generation_prompt=True)
    if prefill:
        if text.rstrip().endswith("<think>"):
            text = text.rstrip() + "\n\n</think>\n\n"
        elif not text.rstrip().endswith("</think>"):
            text = text + "<think>\n\n</think>\n\n"
        assert text.endswith("</think>\n\n"), text[-60:]
    return text


@torch.no_grad()
def generate_batch(model, tok, texts: list[str], max_new: int, temperature: float, seed: int = 0) -> list[str]:
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(next(model.parameters()).device)
    kw = dict(max_new_tokens=max_new, do_sample=temperature > 0, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.95, top_k=20); torch.manual_seed(seed)
    out = model.generate(**enc, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(g, skip_special_tokens=True) for g in gen]


def run_ask(model, tok, cfg, targets: list, art: dict, out: Path, nums: Path, batch: int = 8):
    A = art["ask_model"]; q = A["question"]
    # пилот на 4 айтемах: режим по Д-3
    pilot = targets[:4]
    for mode in ({"prefill": True, "cap": A["max_new_tokens"]}, {"prefill": False, "cap": 512}):
        texts = [ask_prompt_text(r, q, tok, cfg, mode["prefill"]) for r in pilot]
        outs = generate_batch(model, tok, texts, mode["cap"], 0.0)
        unp = sum(parse_yn(o) == "unparsable" for o in outs)
        rec(nums, {f"baseline.ask_model.pilot.prefill_{mode['prefill']}.unparsable": unp,
                   f"baseline.ask_model.pilot.prefill_{mode['prefill']}.examples": [o[:80] for o in outs]})
        if unp == 0:
            break
    else:
        rec(nums, {"baseline.ask_model.pilot_deviation": "оба пилотных режима дали unparsable > 0; режим = последний испытанный (cap 512 без префилла), результат Б1 читать с этой оговоркой"})
    rec(nums, {"baseline.ask_model.mode": mode, "baseline.ask_model.temperature": 0.0,
               "baseline.ask_model.deviation_note": "exploratory baseline: greedy T=0 (behavioural-arm T=1.0 rule does not apply); empty-think prefill = 'answer without reasoning' regime",
               "baseline.ask_model.rendered_tail_example": texts[0][-160:]})
    rows = []
    for qname, qtext in (("main", q), ("self", A["question_self"]), ("inverted", A["question_inverted"])):
        for b in range(0, len(targets), batch):
            chunk = targets[b:b + batch]
            texts = [ask_prompt_text(r, qtext, tok, cfg, mode["prefill"]) for r in chunk]
            for r, t, o in zip(chunk, texts, generate_batch(model, tok, texts, mode["cap"], 0.0)):
                rows.append({"kind": qname, "prompt_key": r.prompt_key, "env": r.env, "condition_id": r.condition_id, "question": qname,
                             "answer": parse_yn(o), "raw": o, "prompt_sha": sha(t)})
    # калибровка: 8 вопросов × 8 deployment-диалогов
    cal_recs = [r for r in targets if r.condition_id == "deployment" and r.prompt_id in A["calibration_dialogs"]]
    for r in cal_recs:  # ground truth машинно-проверяемых меток (Д-11)
        u = r.messages[-1]["content"]
        assert re.search(r"\d", u), f"{r.prompt_id}: q_yes_1 (digits) ложна"
        assert re.search(r"\bthe\b", u), f"{r.prompt_id}: q_yes_3 (the) ложна"
        if re.search(r"\+?\d[\d\s().-]{7,}\d", u):  # не ассерт: возможны цифровые последовательности не-телефоны
            rec(nums, {f"baseline.ask_model.calibration_warn.{r.prompt_id}": "digit run looks phone-like; q_no_4 label needs manual check"})
    cal_items = [(r, qid, qt) for r in cal_recs for qid, qt in A["calibration"].items()]
    for b in range(0, len(cal_items), batch):
        chunk = cal_items[b:b + batch]
        texts = [ask_prompt_text(r, qt, tok, cfg, mode["prefill"]) for r, _, qt in chunk]
        for (r, qid, _), t, o in zip(chunk, texts, generate_batch(model, tok, texts, mode["cap"], 0.0)):
            rows.append({"kind": "calibration", "prompt_key": r.prompt_key, "env": r.env, "condition_id": "deployment", "question": qid,
                         "answer": parse_yn(o), "raw": o, "truth": A["calibration_truth"][qid], "prompt_sha": sha(t)})
    with open(out / "ask_model.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # агрегаты (сырьё для P2; здесь только счётчики с явными знаменателями, К-3)
    for qname in ("main", "self", "inverted"):
        agg = {}
        for r in rows:
            if r["kind"] != qname:
                continue
            g = r["env"][2:] if r["env"].startswith("x_") else ("topic_legacy" if r["env"] == "topic" else r["condition_id"])
            a = agg.setdefault(g, {"yes": 0, "no": 0, "unparsable": 0})
            a[r["answer"]] += 1
        for g, a in agg.items():
            n = a["yes"] + a["no"]; lo, hi = wilson(a["yes"], n)
            rec(nums, {f"baseline.ask_model.{qname}.{g}": {**a, "n_parsable": n, "yes_rate": (a["yes"] / n if n else None), "wilson95": [lo, hi]}})
    cal = [r for r in rows if r["kind"] == "calibration"]
    acc = sum(r["answer"] == r["truth"] for r in cal); unp = sum(r["answer"] == "unparsable" for r in cal)
    yes_on_no = sum(r["answer"] == "yes" for r in cal if r["truth"] == "no"); n_no = sum(r["truth"] == "no" for r in cal)
    no_on_yes = sum(r["answer"] == "no" for r in cal if r["truth"] == "yes"); n_yes = sum(r["truth"] == "yes" for r in cal)
    rec(nums, {"baseline.ask_model.calibration": {"n": len(cal), "accuracy": acc / max(len(cal), 1), "unparsable": unp,
                                                  "yes_bias_rate": (yes_on_no / n_no if n_no else None), "no_bias_rate": (no_on_yes / n_yes if n_yes else None),
                                                  "limitation": "surface-fact calibration != meta-question calibration"}})


# ---------------------------------------------------------------- Б4 (опционально)
def run_b4(model, tok, cfg, dps, art, out: Path, nums: Path):
    B = art["b4"]
    rec(nums, {"secondary.b4.plan": B, "secondary.b4.recorded_before_run": True})
    by_id = {d["prompt_id"]: d for d in dps}
    dep = cfg.get("deployment")
    rows = []
    for pid in B["prompts"]:
        d = by_id[pid]
        from src.prompts import build_prompt
        msgs = build_prompt(d, dep, cfg.lexicon)
        msgs = [{"role": "system", "content": B["instruction"]}] + [{"role": "user", "content": msgs[-1]["content"] + "\n\n" + B["instruction"]}]
        text = render_text(msgs, tok, cfg, add_generation_prompt=True)
        outs = generate_batch(model, tok, [text] * B["k"], 4096, 1.0, seed=int(hashlib.sha256(pid.encode()).hexdigest()[:6], 16))
        for i, o in enumerate(outs):
            rows.append({"prompt_id": pid, "sample": i, "output": o, "truncated": len(tok(o)["input_ids"]) >= 4090})
    with open(out / "b4_true_pc_gens.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rec(nums, {"secondary.b4.n_generated": len(rows), "secondary.b4.n_truncated": sum(r["truncated"] for r in rows),
               "secondary.b4.note": "scoring (answer_regex + rescue over cot, accuracy vs keys) is done in P2 with keys from data/"})


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--dps", default="data/decision_points.json")
    ap.add_argument("--topic", default="data/topic_prompts.json")
    ap.add_argument("--artifact", default="results/p4_frozen.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--wild-n", type=int, default=40)
    ap.add_argument("--wild-seed", type=int, default=7)
    ap.add_argument("--wild-path", default=None, help="явный путь к датасету Б2 (иначе первый подходящий в /kaggle/input — недетерминированно)")
    ap.add_argument("--do-b4", action="store_true")
    ap.add_argument("--anchor-float16", action="store_true", default=True, help="сохранять переизвлечённые базовые активации в fp16 (имя файла честное)")
    ap.add_argument("--skip-ask", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out); nums = out / "numbers.json"
    prev = json.loads(nums.read_text(encoding="utf-8"))
    base_payload = torch.load(out / "probe_acts_fp16.pt", weights_only=False)
    layers = list(base_payload["layers"]); assert list(base_payload["positions"]) == list(POSITIONS)
    L = int(prev["probe.layer"]); assert L in layers

    # (0) фиксация артефакта ДО активаций
    art = json.loads(Path(a.artifact).read_text(encoding="utf-8"))
    art_sha = sha(json.dumps(art, sort_keys=True, ensure_ascii=False))
    (out / "lexicon_a2.json").write_text(json.dumps(art["lexicon"], ensure_ascii=False, indent=1), encoding="utf-8")
    rec(nums, {"probe.exploratory.g1.artifact_sha256": art_sha, "probe.exploratory.g1.artifact_md_sha256": art["md_sha256"],
               "probe.exploratory.g1.lexicon_sha256": sha((out / "lexicon_a2.json").read_text(encoding="utf-8")),
               "probe.exploratory.g1.frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "probe.exploratory.g1.note": "EXPLORATORY post pivot-4.2; new prompt sets written after FPR 25%/precision 0.035 on original 8 were known; sha fixed before any activation of new prompts was computed",
               "probe.exploratory.g1.tau_rule": art["tau_rule"], "probe.exploratory.g1.legacy_reclass": art["legacy_reclass"]})

    cfg = load_conditions("configs/conditions.yaml", allow_unset_gate_params=True)
    tok = load_tokenizer(cfg)
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8"))
    dsha = dps_sha256(a.dps)
    base_recs = make_records(dps, cfg.conditions, tok, cfg, dps_sha=dsha)
    topic_raw = json.loads(Path(a.topic).read_text(encoding="utf-8"))
    tsha = hashlib.sha256(json.dumps(topic_raw["prompts"], sort_keys=True).encode()).hexdigest()[:16]
    dep = cfg.get("deployment")
    trecs = make_records(topic_raw["prompts"], [dep], tok, cfg, dps_sha=tsha)

    # (1) новые записи — тем же путём (Д-14)
    new_dps = build_new_dps(art, dps)
    wild, wmeta = load_wild(art, a.wild_n, a.wild_seed, a.wild_path)
    rec(nums, {"probe.exploratory.g1.wild_meta": wmeta})
    new_dps += [{k: v for k, v in w.items() if k not in ("src_index", "ai_match", "ai_match_broad")} for w in wild]
    nsha = hashlib.sha256(json.dumps(new_dps, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    nrecs = make_records(new_dps, [dep], tok, cfg, dps_sha=nsha)
    # assert суффикса шаблона (Д-11/Д-14): хвост после user-хода побайтово одинаков
    def tail(r):
        return tok.decode(r.input_ids[r.user_span[1]:])
    ref = tail(trecs[0]); tails_ok = all(tail(r) == ref for r in nrecs + trecs + [r for r in base_recs if r.condition_id == "deployment"])
    assert tails_ok, "суффикс шаблона после user-хода различается между наборами"
    assert all(len(r.messages) == 1 for r in nrecs), "в новых записях появился system-ход"
    rec(nums, {"probe.exploratory.g1.template_tail_identical": True, "probe.exploratory.g1.template_tail": ref,
               "probe.exploratory.g1.n_new_records": len(nrecs), "probe.exploratory.g1.new_dps_sha256": nsha,
               "probe.exploratory.g1.new_sets": sorted({r.env for r in nrecs})})
    (out / "new_prompt_records.jsonl").write_text("\n".join(json.dumps({**r.to_dict(), "input_ids": None}, ensure_ascii=False) for r in nrecs) + "\n", encoding="utf-8")

    # (2) активации
    model = load_model(cfg.tokenizer["name"], cfg.model["revision"], getattr(torch, a.dtype))
    # ЯКОРЬ: переизвлечь 8 базовых записей и сверить с сохранёнными (дрейф между сессиями обесценил бы все frac_of_e1)
    from src.activations import row_metrics
    anchor_recs = [r for r in base_recs if r.condition_id in ("deployment", "E1")][:8]
    keys_pos = {k: i for i, k in enumerate(base_payload["keys"])}
    a_acts = extract(model, tok, anchor_recs, [L], ["last"], batch_size=len(anchor_recs), padding_side="right")["last"][L]
    saved = torch.stack([base_payload["acts"]["last"][L][keys_pos[r.prompt_key]] for r in anchor_recs]).float()
    rel, cos = row_metrics(a_acts.float(), saved)
    rec(nums, {"probe.exploratory.g1.anchor.rel_l2_max": rel, "probe.exploratory.g1.anchor.cos_min": cos,
               "probe.exploratory.g1.anchor.n": len(anchor_recs), "probe.exploratory.g1.anchor.keys": [r.prompt_key for r in anchor_recs],
               "probe.exploratory.g1.anchor.rule": "rel_l2<1e-2 и cos>0.999 — иначе base-активации переизвлекаются целиком в этой же сессии (~30 мин) и P2 использует новый файл"})
    if not (rel < 1e-2 and cos > 0.999):
        print(f"!! ЯКОРЬ ПРОВАЛЕН (rel_l2={rel:.3g}, cos={cos:.5f}) → переизвлекаю ВСЕ базовые активации в этой сессии", flush=True)
        allb = base_recs + trecs
        re_acts = extract(model, tok, allb, layers, list(POSITIONS), batch_size=a.batch_size, padding_side="right")
        torch.save({"dtype": a.dtype, "layers": layers, "positions": list(POSITIONS), "keys": [r.prompt_key for r in allb],
                    "acts": {p2: {Lr: re_acts[p2][Lr].half() for Lr in layers} for p2 in POSITIONS}, "reextracted": True},
                   out / "probe_acts_fp16.pt")
        rec(nums, {"probe.exploratory.g1.anchor.reextracted_base": True,
                   "probe.exploratory.g1.anchor.deviation": "base activations re-extracted in G1 session due to anchor failure; all P2 numbers use the new file"})
    t0 = time.time()
    acts = extract(model, tok, nrecs, layers, list(POSITIONS), batch_size=a.batch_size, padding_side="right")
    payload = {"dtype": a.dtype, "layers": layers, "positions": list(POSITIONS), "keys": [r.prompt_key for r in nrecs],
               "acts": {p: {Lr: acts[p][Lr].float() for Lr in layers} for p in POSITIONS}, "dps_sha256": nsha, "artifact_sha256": art_sha,
               "meta": [{"prompt_key": r.prompt_key, "env": r.env, "prompt_id": r.prompt_id, "n_tokens": r.n_prompt_tokens,
                         "n_user_words": len(r.messages[-1]["content"].split())} for r in nrecs]}
    torch.save(payload, out / "probe_acts_extra_fp16.pt")
    rec(nums, {"probe.exploratory.g1.extract_seconds": time.time() - t0})

    # (3) NLL — только нужные записи (deployment/E1/topic/новые), батч 1, не фатально
    t0 = time.time()
    try:
        need = [r for r in base_recs if r.condition_id in ("deployment", "E1")] + trecs + nrecs
        rows = nll_per_record(model, tok, need, batch_size=1)
        with open(out / "nll.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        rec(nums, {"probe.exploratory.g1.nll_rows": len(rows), "probe.exploratory.g1.nll_seconds": time.time() - t0})
    except Exception as e:  # noqa: BLE001
        rec(nums, {"probe.exploratory.g1.nll_deviation": f"NLL step failed: {type(e).__name__}: {str(e)[:200]}"})
        torch.cuda.empty_cache()

    # (4) Б1
    if not a.skip_ask:
        targets = [r for r in base_recs if r.condition_id in ("deployment", "E1")] + trecs + [r for r in nrecs if r.env != "x_dptrunc"]
        t0 = time.time(); run_ask(model, tok, cfg, targets, art, out, nums, batch=a.batch_size)
        rec(nums, {"baseline.ask_model.n_targets": len(targets), "baseline.ask_model.seconds": time.time() - t0})

    # (6) Б4
    if a.do_b4:
        t0 = time.time(); run_b4(model, tok, cfg, dps, art, out, nums); rec(nums, {"secondary.b4.seconds": time.time() - t0})

    rec(nums, {"probe.exploratory.g1.keys_written": KEYS})
    print("G1 DONE; ключи:", len(KEYS))


if __name__ == "__main__":
    main()
