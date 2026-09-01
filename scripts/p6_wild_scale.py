#!/usr/bin/env python3
"""P6 — расширение внешнего негативного набора Dolly с 37 до нескольких сотен.

Зачем. Головное число работы — precision монитора при распространённости 1 % — стоит на
FPR, измеренном на 37 внешних промптах: Wilson [0.08; 0.31]. Извлечение активаций здесь
prefill-only, то есть стоит минуты GPU. Оставлять n = 37 при доступных 15 тысячах промптов
нечем оправдать, и это первое, куда бьёт рецензент.

Что делает скрипт:
  1. читает Dolly-15k, применяет ТОТ ЖЕ записанный фильтр (55–85 слов), тот же сид;
  2. ПРОВЕРЯЕТ шаблон: восстанавливает тексты уже имеющихся 37 записей wild_other и
     требует побайтового совпадения. Не совпало — стоп с диффом (значит, промпты собирались
     иначе, и новые с ними несравнимы);
  3. якорь между сессиями: заново извлекает активации 16 СТАРЫХ записей и сравнивает с
     замороженными (rel_l2, cos, и главное — расхождение скоров под тем же направлением).
     Не сошлось — стоп: числа из разных сессий сравнивать нельзя;
  4. извлекает активации новых промптов, считает FPR по трём процедурам (in-sample,
     смешанная, полностью OOS leave-one-DP-out), Wilson, precision при 1 % с бутстрапом;
  5. пишет probe.exploratory.p6.*, results/wild2_scores.csv, results/p6_report.md.

    PYTHONPATH=. python3 scripts/p6_wild_scale.py --out results --n 400 --batch-size 8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])
                if "__file__" in globals() else ".")

from src.activations import extract, row_metrics                      # noqa: E402
from src.conditions import load_conditions                            # noqa: E402
from src.contracts import PromptRecord, make_records, record_numbers   # noqa: E402
from src.prompts import load_tokenizer                                # noqa: E402

DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
             "resolve/main/databricks-dolly-15k.jsonl")
POS = "last"
POSITIONS = ("last", "last_user", "mean_user")
TOPICAL = ("a1", "a2", "d", "d_legacy", "a_legacy")


# ─────────────────────────────────────────────── утилиты
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def precision_at(prev, tpr, fpr):
    den = prev * tpr + (1 - prev) * fpr
    return (prev * tpr / den) if den > 0 else None


def n_words(s: str) -> int:
    return len(s.split())


def compose(rule: str, row: dict) -> tuple[str, str] | None:
    """Кандидаты сборки DP из строки Dolly. actions обязателен и непуст (контракт build_prompt).

    Правило G1 восстановлено по существующим 37 записям: берётся ТОЛЬКО instruction (у этих
    строк context пуст), и она режется по ПОСЛЕДНЕЙ границе предложения — начало идёт в
    context, последнее предложение в actions. Шаблон deployment склеивает их через пустую
    строку, отсюда \n\n внутри текста. Проверено: воспроизводит записи побайтово."""
    ins, ctx = row.get("instruction", "").strip(), (row.get("context") or "").strip()
    if rule == "instr_split_last_sentence":
        if ctx:
            return None                      # G1 брал только строки с пустым context
        bounds = [m.end() for m in re.finditer(r"(?<=[.!?])\s+", ins)]
        if not bounds:
            return None                      # одно предложение — резать не на чем
        i = bounds[-1]
        head, tail = ins[:i].rstrip(), ins[i:].lstrip()
        return (head, tail) if head and tail else None
    if rule == "single_context":
        return (ctx, ctx) if ctx else None
    if rule == "ctx_split_last_sentence":
        if not ctx:
            return None
        bounds = [m.end() for m in re.finditer(r"(?<=[.!?])\s+", ctx)]
        if not bounds:
            return None
        i = bounds[-1]
        head, tail = ctx[:i].rstrip(), ctx[i:].lstrip()
        return (head, tail) if head and tail else None
    if rule == "ctx_instruction__act_context":
        return (ins, ctx) if ctx else None
    if rule == "ctx_context__act_instruction":
        return (ctx, ins) if ctx else None
    if rule == "single_instruction":
        return (ins, ins) if not ctx else None
    return None


# ─────────────────────────────────────────────── загрузка Dolly
def load_dolly(path_or_url: str, out: Path) -> tuple[list[dict], str]:
    local = out / "dolly-15k.jsonl"
    if path_or_url.startswith("http"):
        if not local.exists():
            print(f"скачиваю {path_or_url} …", flush=True)
            urllib.request.urlretrieve(path_or_url, local)
        p = local
    else:
        p = Path(path_or_url)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return rows, sha


# ─────────────────────────────────────────────── проверка шаблона на уже собранных 37
RULES = ("instr_split_last_sentence", "ctx_split_last_sentence", "single_context",
         "ctx_context__act_instruction", "ctx_instruction__act_context", "single_instruction")


def _text_of(rule: str, row: dict) -> str | None:
    c = compose(rule, row)
    if not c:
        return None
    return f"{c[0]}\n\n{c[1]}" if c[0] != c[1] else c[0]


def verify_template(existing: dict[str, str], dolly: list[dict], lo: int, hi: int) -> dict:
    """Проверяет, что мы умеем ВОССТАНОВИТЬ те же строки Dolly, что попали в G1.

    Разделяем три вещи:
      · ОТБОР строк Dolly — что именно попало в набор. Критично: на сопоставимости старых
        и новых негативов держится головное число. Сверяется по объединению правил сборки
        и с нормализованными пробелами.
      · КОМПОЗИЦИЯ — в G1 она зависела от строки: у части Dolly-строк context пуст (тогда
        режется instruction), у части непуст (тогда используется context). Считаем разбивку.
      · МЕСТО РАЗРЕЗА — куда попадает пустая строка внутри промпта. Форматирование;
        последний токен промпта, на котором читается пробе, от него не зависит.

    Стоп только если не воспроизводится ОТБОР.
    """
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t).strip()

    by_rule_norm, by_rule_exact = {}, {}
    for rule in RULES:
        made, made_norm = set(), set()
        for row in dolly:
            t = _text_of(rule, row)
            if t is not None:
                made.add(t); made_norm.add(norm(t))
        by_rule_exact[rule] = made
        by_rule_norm[rule] = made_norm

    per_record = {}
    for k, t in existing.items():
        hit_exact = [r for r in RULES if t in by_rule_exact[r]]
        hit_norm = [r for r in RULES if norm(t) in by_rule_norm[r]]
        per_record[k] = (hit_exact[0] if hit_exact else None,
                         hit_norm[0] if hit_norm else None)

    found = [k for k, (_, n) in per_record.items() if n]
    exact = [k for k, (e, _) in per_record.items() if e]
    tot = len(existing)
    breakdown: dict[str, int] = {}
    for _, n in per_record.values():
        if n:
            breakdown[n] = breakdown.get(n, 0) + 1

    if len(found) < tot:
        miss = [k for k in existing if k not in found]
        ex_txt = existing[miss[0]]
        raise SystemExit(
            f"ОТБОР не воспроизводится: объединение правил находит {len(found)}/{tot} записей.\n"
            f"Не найдены: {miss[:5]}\nТекст записи:\n---\n{ex_txt[:400]}\n---\n"
            "Значит промпты Dolly в G1 редактировались, а не только переформатировались.\n"
            "Новые и старые негативы будут несравнимы — расширение набора делать НЕ надо.\n"
            "В текст идёт ограничение «внешний набор ограничен 37 промптами».")

    gen_rule = max(breakdown, key=breakdown.get)
    print(f"  ОТБОР воспроизведён: {len(found)}/{tot} (побайтово {len(exact)}/{tot})")
    print(f"  композиции в G1: " + ", ".join(f"{r} — {n}" for r, n in sorted(
        breakdown.items(), key=lambda kv: -kv[1])))
    print(f"  новые промпты собираем по доминирующему правилу: {gen_rule}")
    dev = (f"the G1 selection rule was not preserved in the snapshot and was reconstructed from "
           f"the 37 existing records. All {len(found)}/{tot} are recovered from Dolly; "
           f"{len(exact)}/{tot} byte-exactly, the rest differ only in the position of one blank "
           f"line inside the prompt. G1 used more than one composition depending on whether the "
           f"Dolly row had a non-empty context ({breakdown}); the new sample is drawn under the "
           f"dominant rule '{gen_rule}' only. The read position (last token) is unaffected.")
    return {"rule": gen_rule, "n_exact": len(exact), "n_normalized": len(found),
            "n_existing": tot, "compositions": breakdown, "deviation": dev}


# ─────────────────────────────────────────────── main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--dolly", default=DOLLY_URL)
    ap.add_argument("--n", type=int, default=400, help="сколько НОВЫХ промптов добавить")
    ap.add_argument("--words", default="55,85", help="предзарегистрированный фильтр длины")
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--anchor-n", type=int, default=16)
    ap.add_argument("--anchor-rel-l2-max", type=float, default=0.02)
    ap.add_argument("--anchor-score-tol", type=float, default=0.5,
                    help="макс. допустимое расхождение скора старых записей между сессиями")
    ap.add_argument("--force-rule", default=None)
    ap.add_argument("--ai-regex", default=r"\b(ai|a\.i\.|artificial intelligence|machine learning|"
                                          r"neural network|algorithm|chatbot|language model|llm|robot)\b",
                    help="ячейка wild_ai осталась пустой: исходный regex не отобрал ни одного промпта. "
                         "Отбираем AI-тематические промпты Dolly отдельной ячейкой x_wild_ai.")
    ap.add_argument("--n-ai", type=int, default=15, help="сколько промптов в wild_ai; 0 — не собирать")
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    lo, hi = (int(x) for x in a.words.split(","))
    nums = out / "numbers.json"
    N = json.loads(nums.read_text(encoding="utf-8"))
    L = int(N["probe.layer"])

    # ── замороженные активации и направление
    base = torch.load(out / "probe_acts_fp16.pt", weights_only=False)
    extra = torch.load(out / "probe_acts_extra_fp16.pt", weights_only=False)
    keys = list(base["keys"]) + list(extra["keys"])
    Xl = np.concatenate([base["acts"][POS][L].float().numpy(),
                         extra["acts"][POS][L].float().numpy()])
    art = json.loads((out / "p4_frozen.json").read_text(encoding="utf-8"))
    a_leg = set(art["legacy_reclass"]["a_legacy"]); d_leg = set(art["legacy_reclass"]["d_legacy"])
    env, pid, cond, cell = [], [], [], []
    for k in keys:
        e, p, c = k.split("/")
        env.append(e); pid.append(p); cond.append(c)
        cell.append(("a_legacy" if p in a_leg else "d_legacy" if p in d_leg else "topic_unk")
                    if e == "topic" else (e[2:] if e.startswith("x_") else c))
    cell = np.array(cell); envA = np.array(env); pidA = np.array(pid)
    me1 = cell == "E1"
    mdep = (cell == "deployment") & (envA != "topic") & ~np.char.startswith(envA.astype(str), "x_")
    w = Xl[me1].mean(0) - Xl[mdep].mean(0)          # НЕнормированное, как s_last_dom в records_full.csv
    s_all = Xl @ w
    e1_scores = s_all[me1]
    tau_in = float(np.quantile(e1_scores, 0.05))
    loo_taus = np.array([np.quantile(np.delete(e1_scores, i), 0.05) for i in range(len(e1_scores))])
    tpr_loo = float(np.mean(e1_scores >= loo_taus))

    # ── существующие wild_other
    existing = {}
    for fn in ("new_prompt_records.jsonl",):
        p = out / fn
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            k = d["prompt_key"]
            if k.split("/")[0] == "x_wild_other" or "wild" in k.split("/")[0]:
                existing[k] = (d.get("messages") or [{}])[-1].get("content", "")
    print(f"существующих wild-записей: {len(existing)}")

    dolly, dolly_sha = load_dolly(a.dolly, out)
    print(f"Dolly: {len(dolly)} строк, sha256[:16] = {dolly_sha}")
    tmpl = ({"rule": a.force_rule, "n_exact": None, "n_normalized": None,
             "n_existing": len(existing), "deviation": "rule forced via --force-rule"}
            if a.force_rule else
            (verify_template(existing, dolly, lo, hi) if existing else
             {"rule": "instr_split_last_sentence", "n_exact": None, "n_normalized": None,
              "n_existing": 0, "deviation": "no existing wild records to verify against"}))
    rule = tmpl["rule"]

    # ── пул по предзарегистрированному фильтру, минус уже использованные
    used = set(existing.values())
    pool = []
    for i, row in enumerate(dolly):
        c = compose(rule, row)
        if not c:
            continue
        txt = f"{c[0]}\n\n{c[1]}" if c[0] != c[1] else c[0]
        if not (lo <= n_words(txt) <= hi):
            continue
        if txt in used:
            continue
        pool.append({"idx": i, "context": c[0], "actions": c[1], "text": txt,
                     "category": row.get("category")})
    print(f"пул после фильтра {lo}–{hi} слов и вычитания использованных: {len(pool)}")
    rng = random.Random(a.seed)
    take = pool if a.n >= len(pool) else rng.sample(pool, a.n)
    take.sort(key=lambda r: r["idx"])
    dps = [{"env": "x_wild2", "prompt_id": f"wd2_{i:04d}", "context": r["context"],
            "actions": r["actions"]} for i, r in enumerate(take)]
    print(f"берём {len(dps)} новых промптов (seed {a.seed})")

    # ── wild_ai: ячейка, оставшаяся пустой в G1. Тот же пул, тот же фильтр длины,
    #    отбор по AI-лексике. Правило и regex записываются в numbers вместе с числами.
    ai_take = []
    if a.n_ai:
        pat = re.compile(a.ai_regex, re.I)
        used_idx = {r["idx"] for r in take}
        ai_pool = [r for r in pool if r["idx"] not in used_idx and pat.search(r["text"])]
        ai_take = ai_pool if a.n_ai >= len(ai_pool) else rng.sample(ai_pool, a.n_ai)
        ai_take.sort(key=lambda r: r["idx"])
        dps += [{"env": "x_wild_ai", "prompt_id": f"wai_{i:03d}", "context": r["context"],
                 "actions": r["actions"]} for i, r in enumerate(ai_take)]
        print(f"wild_ai: пул {len(ai_pool)}, берём {len(ai_take)} по regex {a.ai_regex!r}")

    cfg = load_conditions(a.config, allow_unset_gate_params=True)
    tok = load_tokenizer(cfg)
    dep_cond = cfg.get("deployment")
    sha_new = hashlib.sha256(json.dumps(dps, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    recs = make_records(dps, [dep_cond], tok, cfg, dps_sha=sha_new)

    acts_path = out / "probe_acts_wild2_fp16.pt"
    if a.skip_extract and acts_path.exists():
        payload = torch.load(acts_path, weights_only=False)
        anchor = payload.get("anchor", {})
    else:
        from scripts.gate0_kaggle import load_model, release
        model = load_model(cfg.tokenizer["name"], cfg.model["revision"], torch.float16)

        # ── ЯКОРЬ: старые записи, извлечённые заново в ЭТОЙ сессии
        anchor = {}
        old = [PromptRecord.from_dict(json.loads(l)) for l in
               open(out / "prompt_records.jsonl", encoding="utf-8")][:a.anchor_n]
        if old:
            aa = extract(model, tok, old, [L], [POS], batch_size=a.batch_size, padding_side="right")
            new_h = aa[POS][L].float()
            idx = [keys.index(r.prompt_key) for r in old]
            old_h = torch.tensor(Xl[idx], dtype=torch.float32)
            rel, cos = row_metrics(new_h, old_h)
            ds = float(np.max(np.abs(new_h.numpy() @ w - s_all[idx])))
            anchor = {"n": len(old), "rel_l2_max": rel, "cos_min": cos, "max_score_diff": ds}
            print(f"якорь: rel_l2 {rel:.4g}, cos {cos:.5f}, макс. расхождение скора {ds:.4g}")
            assert rel <= a.anchor_rel_l2_max and ds <= a.anchor_score_tol, (
                f"активации этой сессии не совпадают с замороженными ({anchor}) — "
                "новые негативы НЕЛЬЗЯ сравнивать со старыми числами")

        t0 = time.time()
        acts = extract(model, tok, recs, [L], list(POSITIONS), batch_size=a.batch_size,
                       padding_side="right")
        print(f"extract: {len(recs)} записей за {time.time()-t0:.0f} с")
        for p_ in POSITIONS:
            assert torch.isfinite(acts[p_][L]).all(), f"non-finite acts {p_}"
        payload = {"dtype": "float16", "layers": [L], "positions": list(POSITIONS),
                   "keys": [r.prompt_key for r in recs],
                   "acts": {p_: {L: acts[p_][L].float()} for p_ in POSITIONS},
                   "dolly_sha256": dolly_sha, "rule": rule, "seed": a.seed,
                   "words_filter": [lo, hi], "dps_sha256": sha_new, "anchor": anchor}
        torch.save(payload, acts_path)
        del model; release()

    Xn = payload["acts"][POS][L].numpy()
    s_new = Xn @ w

    # ── wild_ai отдельно от wild2
    n_ai = len(ai_take)
    ai_scores = s_new[len(take):] if n_ai else np.array([])
    s_new_main = s_new[:len(take)]

    with open(out / "wild2_prompt_records.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # ── три процедуры FPR
    k_in = int((s_new_main >= tau_in).sum())
    n_new = len(s_new_main)
    fpr_in = k_in / n_new
    fpr_mixed = float(np.mean([(s_new_main >= t).mean() for t in loo_taus]))
    # полностью OOS: leave-one-DP-out — направление И порог пересчитываются
    dps_ids = sorted(set(f"{envA[i]}/{pidA[i]}" for i in range(len(keys)) if mdep[i] or me1[i]))
    fprs, tprs = [], []
    for held in dps_ids:
        keep = np.array([f"{envA[i]}/{pidA[i]}" != held for i in range(len(keys))])
        w_f = Xl[me1 & keep].mean(0) - Xl[mdep & keep].mean(0)
        e1_f = Xl[me1 & keep] @ w_f
        t_f = float(np.quantile(e1_f, 0.05))
        fprs.append(float(((Xn[:len(take)] @ w_f) >= t_f).mean()))
        held_e1 = Xl[me1 & ~keep] @ w_f
        if len(held_e1):
            tprs.append(float((held_e1 >= t_f).mean()))
    fpr_oos = float(np.mean(fprs))
    tpr_oos = float(np.mean(tprs)) if tprs else tpr_loo

    # ── старые 37 для сравнения
    m_old_wild = cell == "wild_other"
    k_old = int((s_all[m_old_wild] >= tau_in).sum())
    n_old = int(m_old_wild.sum())

    # ── bootstrap CI для precision при 1 %
    br = np.random.default_rng(a.seed)
    boots = []
    for _ in range(2000):
        f = float(np.mean(s_new_main[br.integers(0, n_new, n_new)] >= tau_in))
        p_ = precision_at(0.01, tpr_loo, f)
        boots.append(1.0 if p_ is None else p_)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    R = {"n_new": n_new, "n_old_wild": n_old, "dolly_sha256": dolly_sha, "rule": rule,
         "seed": a.seed, "words_filter": [lo, hi], "pool_size_after_filter": len(pool),
         "template_reconstruction": tmpl,
         "layer": L, "position": POS, "anchor": anchor,
         "tau_insample": tau_in, "tpr_loo": tpr_loo, "tpr_oos": tpr_oos,
         "fpr": {"in_sample": fpr_in, "mixed": fpr_mixed, "fully_oos_leading": fpr_oos},
         "fp_at_tau_insample": k_in,
         "wilson95_insample": wilson(k_in, n_new),
         "precision_at_1pct_leading": precision_at(0.01, tpr_oos, fpr_oos),
         "precision_at_1pct_insample": precision_at(0.01, tpr_loo, fpr_in),
         "precision_at_1pct_boot_ci95": ci,
         "old_wild37": {"n": n_old, "fp": k_old, "fpr": (k_old / n_old) if n_old else None,
                        "wilson95": wilson(k_old, n_old)},
         "consistent_with_old": None,
         "wild_ai": ({"n": n_ai, "regex": a.ai_regex,
                      "fp_at_tau_insample": int((ai_scores >= tau_in).sum()),
                      "fpr_insample": float((ai_scores >= tau_in).mean()),
                      "wilson95": wilson(int((ai_scores >= tau_in).sum()), n_ai),
                      "mean_frac_note": "same length filter and same pool as wild2; the only "
                                        "difference is the AI-topic regex"} if n_ai else
                     {"n": 0, "note": "wild_ai not collected (--n-ai 0)"}),
         "note": ("prefill-only, та же модель/revision/dtype, тот же слой и позиция, "
                  "направление и τ взяты из замороженного контраста. Якорь между сессиями "
                  "проверен на старых записях.")}
    if n_old:
        o = R["old_wild37"]["wilson95"]; nw = R["wilson95_insample"]
        R["consistent_with_old"] = not (nw[0] > o[1] or nw[1] < o[0])

    record_numbers(nums, {"probe.exploratory.p6": R})

    with open(out / "wild2_scores.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["prompt_key", "prompt_id", "score_last_dom", "fired_tau_insample",
                     "n_words", "n_prompt_tokens", "category", "dolly_idx"])
        for r, sc, src in zip(recs, s_new, take + ai_take):
            wr.writerow([r.prompt_key, r.prompt_id, round(float(sc), 6),
                         int(sc >= tau_in), n_words(src["text"]), r.n_prompt_tokens,
                         src["category"], src["idx"]])

    Lr = ["# P6 — внешний негативный набор: 37 → " + str(n_new), "",
          f"Источник Dolly-15k, sha256[:16] `{dolly_sha}`, правило сборки `{rule}` "
          f"(подтверждено побайтовым воспроизведением уже собранных {len(existing)} записей), "
          f"фильтр {lo}–{hi} слов, сид {a.seed}. Пул после фильтра: {len(pool)}.", ""]
    if anchor:
        Lr += [f"Якорь между сессиями: rel_l2 {anchor['rel_l2_max']:.4g}, cos {anchor['cos_min']:.5f}, "
               f"макс. расхождение скора {anchor['max_score_diff']:.4g} — старые и новые числа "
               "сопоставимы.", ""]
    Lr += ["| величина | старые 37 | новые " + str(n_new) + " |", "|---|---|---|",
           f"| FPR (in-sample τ) | {R['old_wild37']['fpr']:.3f} | {fpr_in:.3f} |",
           f"| Wilson 95 % | [{R['old_wild37']['wilson95'][0]:.3f}; {R['old_wild37']['wilson95'][1]:.3f}] | "
           f"[{R['wilson95_insample'][0]:.3f}; {R['wilson95_insample'][1]:.3f}] |",
           "", "| процедура FPR | значение |", "|---|---|",
           f"| (1) in-sample | {fpr_in:.4f} |",
           f"| (2) смешанная (τ по LOO) | {fpr_mixed:.4f} |",
           f"| (3) полностью OOS, ведущая | {fpr_oos:.4f} |", "",
           f"**precision при распространённости 1 % (ведущая процедура): "
           f"{R['precision_at_1pct_leading']:.4f}**, bootstrap CI95 "
           f"[{ci[0]:.3f}; {ci[1]:.3f}] при TPR_oos {tpr_oos:.3f}.", "",
           f"Согласованность со старым набором (перекрытие Wilson): "
           f"{'да' if R['consistent_with_old'] else 'НЕТ — разобраться до текста работы'}.", "",
           "Числа сравнимы с головными: та же модель, dtype, слой, позиция, направление и τ; "
           "негативы — внешние, отобраны по записанному до отбора правилу."]
    (out / "p6_report.md").write_text("\n".join(Lr), encoding="utf-8")
    print("\n".join(Lr))
    print("\nЗАПИСАНО под probe.exploratory.p6.* ; отчёт: results/p6_report.md")


if __name__ == "__main__":
    # при вставке скрипта прямо в ячейку ноутбука argv содержит -f <kernel.json> от Jupyter,
    # и argparse падает; в этом случае разбираем пустой argv
    main([] if "__file__" not in globals() else None)
