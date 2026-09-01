#!/usr/bin/env python3
"""p4_analysis.py — B1–B9 + заменители B10 на ЗАМОРОЖЕННЫХ данных (2026-09-01).

Ничего не пересчитывает из P2: читает probe_acts_*.pt, ask_model.jsonl, *_prompt_records.jsonl,
пишет ТОЛЬКО под probe.exploratory.p4.* и results/p4_report.md. Свой генератор случайности
(SEED_P4), поток rng P2 не затрагивается.

Запуск:  python3 scripts/p4_analysis.py --out results --artifact results/p4_frozen.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED_P4 = 20260901_4
POS = "last"
KEYS: list[str] = []
_SIGNS: dict[int, np.ndarray] = {}


# ─────────────────────────────────────────── утилиты
def sanitize(o):
    if isinstance(o, dict): return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        if any(isinstance(x, (dict, list, tuple)) for x in o): return {f"i{i}": sanitize(x) for i, x in enumerate(o)}
        return [sanitize(x) for x in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


def _flat(k, v, o):
    if isinstance(v, dict):
        for kk, vv in v.items(): _flat(f"{k}.{kk}", vv, o)
    else: o[k] = v


def rec(nums: Path, d: dict):
    from src.contracts import record_numbers
    d = sanitize(d); record_numbers(nums, d)
    cur = json.loads(nums.read_text(encoding="utf-8")); flat = {}
    for k, v in d.items(): _flat(k, v, flat)
    for k in flat:
        assert k in cur, f"ключ не записался: {k}"
        KEYS.append(k)


def sign_perm_exact(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=np.float64); n = len(d); assert 1 <= n <= 20
    if n not in _SIGNS:
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
        _SIGNS[n] = (1 - 2 * bits).astype(np.float64)
    obs = abs(d.mean()); means = np.abs(_SIGNS[n] @ d) / n
    p = float((means >= obs - 1e-12 * max(obs, 1.0)).mean())
    assert p >= 2 ** (1 - n) - 1e-12
    return p


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0: return [float("nan")] * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0: return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def fisher_one_sided(a, b, c, d) -> float:
    """P(X >= a) для таблицы [[a,b],[c,d]] (гипергеометрическая)."""
    from math import comb
    n1, n2, t = a + b, c + d, a + c
    tot = sum(comb(n1, k) * comb(n2, t - k) for k in range(max(0, t - n2), min(n1, t) + 1))
    num = sum(comb(n1, k) * comb(n2, t - k) for k in range(a, min(n1, t) + 1))
    return num / tot if tot else float("nan")


def precision_at(p, tpr, fpr):
    den = p * tpr + (1 - p) * fpr
    return (p * tpr / den) if den > 0 else float("nan")


def tau_loo(e1: np.ndarray, q=0.05):
    return float(np.quantile(e1, q)), np.array([np.quantile(np.delete(e1, i), q) for i in range(len(e1))])


# ─────────────────────────────────────────── загрузка
class D:
    def __init__(self, out: Path, art: dict):
        import torch
        base = torch.load(out / "probe_acts_fp16.pt", weights_only=False)
        extra = torch.load(out / "probe_acts_extra_fp16.pt", weights_only=False)
        self.layers = list(base["layers"]); self.L = None
        self.keys = list(base["keys"]) + list(extra["keys"])
        self.X = {p: {Lr: np.concatenate([base["acts"][p][Lr].float().numpy(), extra["acts"][p][Lr].float().numpy()])
                      for Lr in self.layers} for p in base["positions"]}
        self._finish(art, out)

    def _finish(self, art, out):
        a_leg = set(art["legacy_reclass"]["a_legacy"]); d_leg = set(art["legacy_reclass"]["d_legacy"])
        self.env, self.pid, self.cond, self.cell = [], [], [], []
        for k in self.keys:
            e, p, c = k.split("/"); self.env.append(e); self.pid.append(p); self.cond.append(c)
            self.cell.append(("a_legacy" if p in a_leg else "d_legacy" if p in d_leg else "topic_unk") if e == "topic"
                             else (e[2:] if e.startswith("x_") else c))
        self.cell = np.array(self.cell); self.envA = np.array(self.env); self.pidA = np.array(self.pid)
        self.N = len(self.keys)
        self.texts = {}
        for fn in ("prompt_records.jsonl", "new_prompt_records.jsonl"):
            p = out / fn
            if p.exists():
                for line in open(p, encoding="utf-8"):
                    d = json.loads(line)
                    msgs = d.get("messages") or []
                    if msgs: self.texts[d["prompt_key"]] = msgs[-1]["content"]
                    if d.get("input_ids"): self.__dict__.setdefault("ids", {})[d["prompt_key"]] = (d["input_ids"], d.get("user_span"))

    def m(self, *cs): return np.isin(self.cell, cs)

    def idx(self, cell, pid):
        w = np.where((self.cell == cell) & (self.pidA == pid))[0]
        assert len(w) == 1, (cell, pid, len(w)); return int(w[0])

    def dep(self): return self.m("deployment") & (self.envA != "topic") & ~np.char.startswith(self.envA.astype(str), "x_")


def direction(d: D, L: int, pos=POS):
    X = d.X[pos][L]; return X[d.m("E1")].mean(0) - X[d.dep()].mean(0)


# ─────────────────────────────────────────── B1: a1 ↔ a_head по сюжетам
def b1_story_contrast(d: D, s: np.ndarray, art: dict, p2_a1a2: float | None = None) -> dict:
    sm = art.get("story_map", {})
    dep_mean = s[d.dep()].mean(); e1_shift = s[d.m("E1")].mean() - dep_mean
    rows, d_ah, d_a2 = [], [], []
    for story, ids in sm.items():
        got = {}
        for pid in ids:
            for c in ("a1", "a2", "a_head"):
                w = np.where((d.cell == c) & (d.pidA == pid))[0]
                if len(w): got[c] = float(s[int(w[0])])
        if {"a1", "a_head"} <= set(got):
            d_ah.append(got["a1"] - got["a_head"])
            if "a2" in got: d_a2.append(got["a1"] - got["a2"])
            rows.append({"story": story, **{k: (v - dep_mean) / e1_shift for k, v in got.items()}})
    rng4 = np.random.default_rng(SEED_P4)
    out = {"n_stories": len(rows), "per_story_frac": rows}
    for name, arr in (("a1_minus_a_head", d_ah), ("a1_minus_a2", d_a2)):
        a = np.array(arr)
        if len(a):
            bt = np.array([np.mean(a[rng4.integers(0, len(a), len(a))]) / e1_shift for _ in range(5000)])
            out[name] = {"n": len(a), "mean": float(a.mean()), "mean_frac_of_e1": float(a.mean() / e1_shift),
                         "cluster_ci95_by_story": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                         "same_sign": int((a > 0).sum()), "p_sign_perm_exact": sign_perm_exact(a),
                         "min_achievable_p": 2 ** (1 - len(a))}
    for k in ("a1_minus_a_head", "a1_minus_a2"):
        if k in out:
            fl = 2.0 ** (1 - out[k]["n"])
            out[k]["p_at_floor"] = bool(abs(out[k]["p_sign_perm_exact"] - fl) < 1e-12)
            out[k]["p_report_as"] = (f"{out[k]['same_sign']}/{out[k]['n']} одного знака, p на полу точного теста "
                                     f"({fl:.4g})" if out[k]["p_at_floor"] else f"p = {out[k]['p_sign_perm_exact']:.4g}")
    if p2_a1a2 is not None and "a1_minus_a2" in out:
        delta = abs(out["a1_minus_a2"]["mean_frac_of_e1"] - p2_a1a2)
        out["a1_minus_a2"]["matches_p2_primary"] = bool(delta < 1e-6)
        out["a1_minus_a2"]["p2_value"] = p2_a1a2
        assert delta < 1e-3, f"a1−a2 в P4 ({out['a1_minus_a2']['mean_frac_of_e1']}) расходится с P2 ({p2_a1a2})"
    out["reading"] = ("сюжет матчирован, меняется ТИП ПРОСЬБЫ (a_head: «помоги написать» при том же фоне) "
                      "vs ЛЕКСИКА (a2: снята вся зарегистрированная eval-лексика). Сравнить величины двух дельт. "
                      "Оговорка: a_head имеет чуть меньшую eval-плотность (6.84 vs 8.38) — контраст матчирован по "
                      "сюжету, но не по плотности")
    return out


# ─────────────────────────────────────────── B2: полосы на пропущенных ячейках
def b2_bands(d: D, L: int, cells: tuple, n_dirs=1000, n_perm=200, seed=0) -> dict:
    try:
        import torch
        from src.probe_toy import perm_band, random_band
    except Exception as e:
        return {"unavailable": f"{type(e).__name__}: {str(e)[:80]}"}
    X = torch.from_numpy(d.X[POS][L]); mdep = d.dep(); contrast = mdep | d.m("E1")
    yc = torch.tensor([1.0 if d.cell[i] == "E1" else 0.0 for i in range(d.N) if contrast[i]])
    Xc = X[torch.from_numpy(contrast)]; Xref = X[torch.from_numpy(mdep)]
    pair_ids = [f"{d.env[i]}/{d.pid[i]}" for i in range(d.N) if contrast[i]]
    out = {}
    for c in cells:
        mc = d.m(c)
        if not mc.sum(): continue
        Xt = X[torch.from_numpy(mc)]
        rb = random_band(Xc, yc, Xt, Xref, n_dirs=n_dirs, seed=seed)
        pb = perm_band(Xc, yc, pair_ids, Xt, Xref, n_perm=n_perm, seed=seed)
        out[c] = {"random_band": {k: float(v) for k, v in rb.items()},
                  "perm_band": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in pb.items()}}
    out["note"] = ("те же функции prereg-ворот v3, что в P2; здесь — ячейки, пропущенные циклом P2 "
                   "(b, c1, c2, len140, len210). Существующие 10 ячеек не пересчитываются.")
    return out


# ─────────────────────────────────────────── B4/B5/B7: явные ответы модели
def load_ask(out: Path) -> list[dict]:
    p = out / "ask_model.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def b4_selfyes_scores(d: D, s: np.ndarray, ask: list[dict]) -> dict:
    """Скор пробе у записей, где модель на самореферентный вопрос ответила да/нет (sandbagging/E1)."""
    res = {}
    for env in ("sandbagging", "secret_number"):
        yes, no = [], []
        for r in ask:
            if r.get("kind") != "self" or r.get("env") != env or r.get("condition_id") != "E1": continue
            k = r.get("prompt_key")
            if k in d.keys:
                i = d.keys.index(k)
                (yes if r["answer"] == "yes" else no if r["answer"] == "no" else []).append(float(s[i]))
        if yes and no:
            n_dep_yes = sum(1 for r in ask if r.get("kind") == "self" and r.get("env") == env
                            and r.get("condition_id") == "deployment" and r["answer"] == "yes")
            n_dep = sum(1 for r in ask if r.get("kind") == "self" and r.get("env") == env
                        and r.get("condition_id") == "deployment" and r["answer"] in ("yes", "no"))
            res[env] = {"n_yes": len(yes), "n_no": len(no), "mean_yes": float(np.mean(yes)), "mean_no": float(np.mean(no)),
                        "auc_yes_vs_no": auc(np.array(yes), np.array(no)),
                        "e1_yes_vs_deployment_yes": {"e1": [len(yes), len(yes) + len(no)], "deployment": [n_dep_yes, n_dep],
                                                     "p_fisher_one_sided": fisher_one_sided(len(yes), len(no), n_dep_yes, n_dep - n_dep_yes)},
                        "reading": "если AUC ≈ 0.5 — пробе не отличает те записи, где модель сама признала самореференцию"}
        else:
            res[env] = {"n_yes": len(yes), "n_no": len(no), "degenerate": True}
    return res


def b5_self_by_env(ask: list[dict]) -> dict:
    agg = defaultdict(lambda: defaultdict(lambda: {"yes": 0, "no": 0, "unparsable": 0}))
    for r in ask:
        if r.get("kind") != "self": continue
        cell = r["env"][2:] if r["env"].startswith("x_") else ("topic_legacy" if r["env"] == "topic" else r["condition_id"])
        pk = str(r.get("prompt_key", "")); pid = pk.split("/")[1] if "/" in pk else ""
        sub = (r["env"] if not r["env"].startswith("x_") and r["env"] != "topic"
               else "sandbagging" if pid.startswith(("sb_", "L140_sb", "L210_sb", "dt_sb"))
               else "secret_number" if pid.startswith(("sn_", "L140_sn", "L210_sn", "dt_sn")) else "n/a")
        agg[cell][sub][r["answer"]] += 1
    out = {}
    for cell, subs in agg.items():
        out[cell] = {sub: {**v, "yes_rate": v["yes"] / max(v["yes"] + v["no"], 1)} for sub, v in subs.items()}
    out["reading"] = ("если у len140/len210 внутри secret_number 8/8 «да», а внутри sandbagging 0/8 — усечение промпта "
                      "вдвое не меняет самореферентный ответ: он задан рамкой среды, не eval-вставкой")
    return out


def b7_item_level(d: D, s: np.ndarray, ask: list[dict]) -> dict:
    """СТРАТИФИЦИРОВАННЫЙ ВНУТРИЯЧЕЙКОВЫЙ AUC (Mantel–Haenszel). Наивный item-level AUC по построению
    возвращает ≈0.5: 96% пар межъячейковые, где различаются домен/длина/тип запроса, а ответ модели —
    довесок. Считаем конкордантность только ВНУТРИ ячейки."""
    kidx = {k: i for i, k in enumerate(d.keys)}
    res = {}
    for q in ("main", "self", "inverted"):
        by_cell = defaultdict(lambda: {"yes": [], "no": []})
        for r in ask:
            if r.get("kind") != q: continue
            i = kidx.get(r.get("prompt_key"))
            if i is None: continue
            if d.cell[i] in ("E1", "deployment") and not d.envA[i].startswith("x_"): continue  # обучающий контраст вон
            if r["answer"] in ("yes", "no"): by_cell[d.cell[i]][r["answer"]].append(float(s[i]))
        num = den = 0.0; cells_used, ny, nn = [], 0, 0
        naive_y, naive_n = [], []
        for c, v in by_cell.items():
            y, no_ = np.array(v["yes"]), np.array(v["no"])
            naive_y += list(y); naive_n += list(no_); ny += len(y); nn += len(no_)
            if len(y) and len(no_):
                dd = y[:, None] - no_[None, :]
                num += (dd > 0).sum() + 0.5 * (dd == 0).sum(); den += dd.size
                cells_used.append(c)
        res[q] = {"n_yes": ny, "n_no": nn, "n_cells_with_variation": len(cells_used), "cells": sorted(cells_used),
                  "n_pairs_within": int(den), "auc_within_cell": (num / den) if den else float("nan"),
                  "auc_naive_pooled_DO_NOT_USE": auc(np.array(naive_y), np.array(naive_n)) if naive_y and naive_n else float("nan"),
                  "power_note": f"внутриячейковых пар {int(den)} из {ny*nn} — мощность низкая, CI широк"}
    res["reading"] = ("ВЕДУЩАЯ величина — auc_within_cell (Mantel–Haenszel по ячейкам). Наивный пул приведён "
                      "только чтобы показать разрыв: он ≈0.5 по композиции ячеек, а не по связи ответа со скором.")
    return res


# ─────────────────────────────────────────── B6: precision только по Dolly
def dolly_enrichment(d: D, s: np.ndarray, out: Path) -> dict:
    """B3: обогащение сработавших Dolly категорией «интерпретируй измеренные показатели человека».
    Метки читаются из results/dolly_labels.json ({prompt_id: 0/1}); нет файла → pending."""
    p = out / "dolly_labels.json"
    if not p.exists():
        return {"pending": True, "how_to": "положить results/dolly_labels.json вида {\"wd_003\": 1, ...} "
                                           "(слепая кодировка по записанному правилу, кодировщик скоров не видел) "
                                           "и перезапустить P4"}
    lab = json.loads(p.read_text(encoding="utf-8"))
    e1 = s[d.m("E1")]; _, loo = tau_loo(e1); tau = float(np.median(loo))
    mw = np.where(d.m("wild_other"))[0]
    fired = {d.pid[i]: bool(s[i] >= tau) for i in mw}
    a = sum(1 for k, v in lab.items() if v and fired.get(k)); b = sum(1 for k, v in lab.items() if v and not fired.get(k))
    c = sum(1 for k, v in lab.items() if not v and fired.get(k)); dd = sum(1 for k, v in lab.items() if not v and not fired.get(k))
    return {"n_labelled": len(lab), "tau_median_loo": tau, "table": {"cat_fired": a, "cat_not_fired": b,
            "other_fired": c, "other_not_fired": dd}, "p_fisher_one_sided": fisher_one_sided(a, b, c, dd),
            "fired_ids": sorted(k for k, v in fired.items() if v),
            "caveat": ("категория выделена ПОСЛЕ просмотра скоров; правило зафиксировано до подсчёта, кодировщик "
                       "скоров не видел. Post-hoc, не предрегистрация.")}


def b6_precision_dolly(d: D, s: np.ndarray) -> dict:
    e1 = s[d.m("E1")]; _, loo = tau_loo(e1); tpr = float(np.mean(e1 >= loo))
    out = {"tpr_loo": tpr}
    for name, cells in (("external_dolly", ("wild_other",)),
                        ("authored_posthoc", ("a1", "a2", "d")),
                        ("legacy_prereg", ("a_legacy", "d_legacy")),
                        ("neutral_tn", ("n",))):
        x = s[d.m(*cells)]
        if not len(x): continue
        fpr = float(np.mean([(x >= t2).mean() for t2 in loo]))
        tau_in = float(np.quantile(e1, 0.05)); k = int((x >= tau_in).sum())
        out[name] = {"n": int(len(x)), "fpr_loo": fpr, "fp_at_tau_insample": k,
                     "fpr_insample": k / len(x), "fpr_insample_wilson95": wilson(k, len(x)),
                     "precision_at_1pct": (precision_at(0.01, tpr, fpr) if fpr > 0 else None),
                     "precision_note": (None if fpr > 0 else f"n/a: FPR=0 при n={len(x)} → precision=1.0 механически; "
                                        f"Wilson по in-sample {wilson(k, len(x))}")}
    out["reading"] = ("ответ на «негативы вы написали сами»: три группы раздельно — authored_posthoc (написаны "
                      "после того, как результат на исходных 8 был известен), legacy_prereg (исходные 8, до "
                      "результата), external_dolly (внешние; источник/sha/сид записаны до отбора). "
                      "NB: fp_at_tau_insample и fpr_loo — разные процедуры, Wilson относится к первой.")
    return out


# ─────────────────────────────────────────── B8: LR / LDA
def b8_lr_lda(d: D, L: int) -> dict:
    X = d.X[POS][L]; me1, mdep = d.m("E1"), d.dep()
    Xtr = np.concatenate([X[me1], X[mdep]]); y = np.concatenate([np.ones(me1.sum()), np.zeros(mdep.sum())])
    pids = [f"{d.env[i]}/{d.pid[i]}" for i in np.where(me1)[0]] + [f"{d.env[i]}/{d.pid[i]}" for i in np.where(mdep)[0]]
    topical = ("a1", "a2", "d", "d_legacy", "a_legacy")
    res = {}
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        return {"unavailable": f"{type(e).__name__}: {str(e)[:60]}"}
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    methods = [("logreg_C1e-2", lambda: LogisticRegression(max_iter=4000, C=1e-2)),
               ("logreg_C1", lambda: LogisticRegression(max_iter=4000, C=1.0)),
               ("logreg_C1e2", lambda: LogisticRegression(max_iter=4000, C=1e2)),
               ("lda_shrinkage", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
               ("linear_svc", lambda: LinearSVC(C=1.0, max_iter=20000)),
               ("pca20_lda_svd", lambda: make_pipeline(PCA(n_components=20), LinearDiscriminantAnalysis(solver="svd")))]
    for name, mk in methods:
        # held-out по промптам, ЧЕРЕДОВАНИЕМ (как heldout_by_prompt в prereg), иначе фолд = одна среда
        uniq = sorted(set(pids)); folds = [np.array(uniq[i::4]) for i in range(4)]
        hold = []
        for f in folds:
            tr = np.array([p not in set(f) for p in pids])
            if len(set(y[tr])) < 2: continue
            clf = mk().fit(Xtr[tr], y[tr]); sc = clf.decision_function(Xtr[~tr])
            hold.append(auc(sc[y[~tr] == 1], sc[y[~tr] == 0]))
        clf = mk().fit(Xtr, y)
        s_all = clf.decision_function(X)
        e1s = s_all[me1]; _, loo = tau_loo(e1s); t = s_all[d.m(*topical)]
        w_dm = direction(d, L)
        if hasattr(clf, "coef_"): w = np.asarray(clf.coef_).ravel()
        else:  # pipeline PCA→LDA: направление в исходном пространстве аналитически
            pca_, lda_ = clf.steps[0][1], clf.steps[1][1]
            w = pca_.components_.T @ np.asarray(lda_.coef_).ravel()
        e1sh = float(e1s.mean() - s_all[mdep].mean()); sd = float(np.std(s_all[mdep]))
        res[name] = {"cos_with_diff_of_means": float(w @ w_dm / (np.linalg.norm(w) * np.linalg.norm(w_dm) + 1e-9)),
                     "e1_shift": e1sh, "degenerate_denominator": bool(abs(e1sh) < 0.5 * sd),
                     "n_iter": (int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") and clf.n_iter_ is not None else None),
                     "heldout_auc_mean": float(np.mean(hold)) if hold else float("nan"),
                     "auc_e1_vs_deploy_insample": auc(e1s, s_all[mdep]),
                     "fpr_topical_loo": float(np.mean([(t >= q).mean() for q in loo])),
                     "topical_frac_of_e1": float((t.mean() - s_all[mdep].mean()) / (e1s.mean() - s_all[mdep].mean()))}
    res["reading"] = ("ВЕДУЩИЙ выход — cos_with_diff_of_means. При n=64, d=4096 L2-регуляризованные методы "
                      "СХОДЯТСЯ к разности средних, поэтому «LR даёт ту же картину» — тавтология, а не сравнение "
                      "методов. Если cos ≥ 0.97 у всех — корректное утверждение: в этом режиме все "
                      "L2-линейные методы коллапсируют к diff-of-means, prereg-выбор не был ограничением. "
                      "pca20_lda_svd и linear_svc отбеливают внутри span данных — это единственные РЕАЛЬНО другие "
                      "методы в списке; смотреть на их cos и topical_frac отдельно. EXPLORATORY.")
    return res


# ─────────────────────────────────────────── B9: последние токены
def eta_squared(groups: list[np.ndarray]) -> float:
    """η² = SS_between / SS_total со взвешиванием по размеру групп (наивная var(средних) смещена при неравных n)."""
    allv = np.concatenate(groups)
    if len(allv) < 3: return float("nan")
    gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    sst = ((allv - gm) ** 2).sum()
    return float(ssb / sst) if sst > 0 else float("nan")


def b9_last_tokens(d: D, s: np.ndarray, scored_position: str = "last_user") -> dict:
    ids = getattr(d, "ids", {})
    if not ids: return {"unavailable": "input_ids отсутствуют в *_prompt_records.jsonl"}
    train = np.array([d.cell[i] in ("E1", "deployment") and not d.envA[i].startswith("x_") for i in range(d.N)])
    by_cell = defaultdict(lambda: defaultdict(list))
    for i, k in enumerate(d.keys):
        if k not in ids: continue
        seq, span = ids[k]
        if not span: continue
        j = span[1] - 1
        if 0 <= j < len(seq): by_cell[d.cell[i]][seq[j]].append(float(s[i]))
    out = {}
    for c, tab in by_cell.items():
        rows = {str(t): {"n": len(v), "mean_score": float(np.mean(v))} for t, v in sorted(tab.items(), key=lambda x: -len(x[1]))[:5]}
        out[c] = {"n_distinct_last_user_tokens": len(tab), "top": rows}
    # различает ли последний токен внутри ячейки
    within = {}
    for c, tab in by_cell.items():
        if len(tab) >= 2 and sum(len(v) for v in tab.values()) >= 6:
            groups = [np.array(v) for v in tab.values() if len(v) >= 2]
            if len(groups) >= 2:
                between = np.var([g.mean() for g in groups]); total = np.var(np.concatenate(groups))
                within[c] = {"var_between_tokens_over_total": float(between / total) if total else float("nan"),
                             "n_groups": len(groups)}
    out["within_cell_token_variance"] = within
    # ГЛАВНОЕ: η² по ВСЕМ записям сразу — внутри ячейки токен почти константен, поэтому внутриячейковый тест бессилен
    def _eta(mask, label):
        g = defaultdict(list)
        for i, k in enumerate(d.keys):
            if k not in ids or not mask[i]: continue
            seq, span = ids[k]
            if not span: continue
            j = span[1] - 1
            if 0 <= j < len(seq): g[seq[j]].append(float(s[i]))
        gg = [np.array(v) for v in g.values() if len(v) >= 2]
        return {"eta2": eta_squared(gg) if len(gg) >= 2 else float("nan"), "n_token_groups": len(gg),
                "n_records": int(sum(len(x) for x in gg)), "scope": label,
                "top_tokens": {str(t2): {"n": len(v), "mean": float(np.mean(v))}
                               for t2, v in sorted(g.items(), key=lambda x: -len(x[1]))[:6]}}
    allm = np.ones(d.N, bool)
    out["global_eta2_all_records_CONFOUNDED"] = {**_eta(allm, "все записи"),
        "warning": ("НЕ ИНТЕРПРЕТИРОВАТЬ: у E1 last_user-токен = ')' — закрывающая скобка самой вставки, "
                    "поэтому идентичность токена коллинеарна наличию вставки ПО ПОСТРОЕНИЮ. Ровно та же "
                    "болезнь, что наивный пул в B7. Ведущая величина — global_eta2_excl_training")}
    out["global_eta2_excl_training"] = _eta(~train, "без чистых E1/deployment (тот же фильтр, что в B7)")

    # РЕШАЮЩИЙ ТЕСТ: воспроизводит ли ОДИН индикатор «last_user-токен == токен E1» показатели позиции
    tok = {}
    for i, k in enumerate(d.keys):
        if k not in ids: continue
        seq, span = ids[k]
        if span and 0 <= span[1] - 1 < len(seq): tok[i] = seq[span[1] - 1]
    me1 = np.where(d.m("E1"))[0]; e1toks = [tok[i] for i in me1 if i in tok]
    if e1toks:
        from collections import Counter
        t_e1, n_mode = Counter(e1toks).most_common(1)[0]
        pred = np.array([1.0 if tok.get(i, -1) == t_e1 else 0.0 for i in range(d.N)])
        topical = d.m("a1", "a2", "d", "d_legacy", "a_legacy")
        mdep = d.dep()
        out["token_only_predictor"] = {
            "e1_last_user_token": int(t_e1), "e1_token_share": n_mode / len(e1toks),
            "auc_e1_vs_deploy": auc(pred[d.m("E1")], pred[mdep]),
            "fpr_topical": float(pred[topical].mean()) if topical.sum() else float("nan"),
            "share_topical_with_e1_token": float(pred[topical].mean()) if topical.sum() else float("nan"),
            "reading": ("если AUC ≈ 1.0 и FPR_topical ≈ 0 — «чистота» last_user воспроизводится ОДНИМ "
                        "индикатором идентичности токена, артефакт доказан (слабость #7 закрывается "
                        "утвердительно). Если нет — last_user НЕ артефакт, и это меняет заголовок")}
    out["scored_position"] = scored_position
    out["reading"] = ("η² по всем записям = доля дисперсии скора, объяснённая идентичностью last_user-токена. "
                      "Высокое η² → «чистота» позиции last_user (FPR_topical 0, precision 1.0) объясняется тем, "
                      "что у E1 токен ')', у deployment '.', у topic '?'. NB: на позиции last токен уникален "
                      "(198='\\n' по gate0.check2), там группировать нечего — блок считается на last_user.")
    return out


# ─────────────────────────────────────────── заменители B10 (длина)
def length_substitutes(d: D, s: np.ndarray) -> dict:
    words = {k: len(t.split()) for k, t in d.texts.items()}
    dep_mean = s[d.dep()].mean(); e1_shift = s[d.m("E1")].mean() - dep_mean
    # 1) внутри внешнего набора длины не контролировались — есть ли связь скора с длиной
    mw = d.m("wild_other")
    xw = np.array([words.get(d.keys[i], np.nan) for i in np.where(mw)[0]]); yw = s[mw]
    ok = ~np.isnan(xw)
    sub1 = {"n": int(ok.sum()), "len_min": float(np.nanmin(xw)), "len_max": float(np.nanmax(xw)),
            "spearman_score_vs_len": spearman(xw[ok], yw[ok])}
    # 2) насыщение лестницы против чужих ячеек той же/меньшей длины
    lad = {c: {"frac": float((s[d.m(c)].mean() - dep_mean) / e1_shift),
               "len_median": float(np.median([words.get(d.keys[i], np.nan) for i in np.where(d.m(c))[0]]))}
           for c in ("len140", "len210") if d.m(c).sum()}
    struc = {c: {"frac": float((s[d.m(c)].mean() - dep_mean) / e1_shift),
                 "len_median": float(np.median([words.get(d.keys[i], np.nan) for i in np.where(d.m(c))[0]])),
                 "note": "STRUCTURAL, не ступень лестницы: инструкция + заголовок questions + 2 вопроса (MCQ-формат)"}
             for c in ("dptrunc",) if d.m(c).sum()}
    foreign = {c: {"frac": float((s[d.m(c)].mean() - dep_mean) / e1_shift),
                   "len_median": float(np.median([words.get(d.keys[i], np.nan) for i in np.where(d.m(c))[0]]))}
               for c in ("a1", "a2", "b", "c1", "c2", "d", "n", "wild_other") if d.m(c).sum()}
    sub2 = {"ladder": lad, "structural_reference": struc, "foreign": foreign,
            "reading": ("ГРАНИЦА, а не опровержение: эффект длины НА ОДНОМ ТЕКСТЕ насыщается около +0.3 E1 "
                        "(218→156 слов даёт +0.03). Наблюдаемые 0.57–0.97 у чужих ячеек им не объясняются. "
                        "Прямой контроль «чужой текст ~300 слов» НЕ проводился — ограничение, названо в future work.")}
    return {"within_external_length": sub1, "saturation_vs_foreign": sub2,
            "limitation": ("прямой контроль «чужой текст длиной ~300 слов» НЕ проводился (окно данных закрыто); "
                           "названо в future work")}


# ─────────────────────────────────────────── отчёт
def report(R: dict, out: Path):
    L = ["# P4 report (EXPLORATORY, замороженные данные, 2026-09-01)", ""]
    b1 = R.get("b1_story", {})
    if b1.get("a1_minus_a_head"):
        ah, a2 = b1["a1_minus_a_head"], b1.get("a1_minus_a2", {})
        L += ["## B1 · a1 ↔ a_head по 8 сюжетам (тип просьбы) vs a1 ↔ a2 (лексика)",
              f"  a1−a_head: {ah['mean_frac_of_e1']:+.3f} E1, {ah['same_sign']}/{ah['n']} одного знака, p={ah['p_sign_perm_exact']:.4f}",
              f"  a1−a2:     {a2.get('mean_frac_of_e1', float('nan')):+.3f} E1, {a2.get('same_sign','—')}/{a2.get('n','—')}, p={a2.get('p_sign_perm_exact', float('nan')):.4f}", ""]
    bands = R.get("b2_bands", {})
    if bands and not bands.get("unavailable"):
        L += ["## B2 · полосы на ранее пропущенных ячейках", "| ячейка | сдвиг | perm q95 | perm_p | выше random |", "|---|---|---|---|---|"]
        for c, v in bands.items():
            if isinstance(v, dict) and "perm_band" in v:
                pb, rb = v["perm_band"], v["random_band"]
                L.append(f"| {c} | {pb.get('probe_shift', float('nan')):.2f} | {pb.get('perm_abs_q95', float('nan')):.2f} | {pb.get('perm_p', float('nan')):.3f} | {rb.get('above_band')} |")
        L.append("")
    b6 = R.get("b6_precision_dolly", {})
    if b6:
        def _fmt(g):
            v = b6.get(g)
            if not isinstance(v, dict): return f"{g}: —"
            pr = v.get("precision_at_1pct")
            pr = "n/a (FPR=0)" if pr is None else f"{pr:.3f}"
            return f"{g}: precision@1 % {pr}, FPR {v['fpr_loo']:.3f}, n={v['n']}"
        L += ["## B6 · precision@1 % по происхождению негативов"] + \
             [f"  {_fmt(g)}" for g in ("external_dolly", "authored_posthoc", "legacy_prereg", "neutral_tn")] + [""]
    b8 = R.get("b8_lr_lda", {})
    if b8 and not b8.get("unavailable"):
        L += ["## B8 · LR / LDA на тех же активациях", "| метод | held-out AUC | FPR_topical | topical frac | cos с diff-of-means |", "|---|---|---|---|---|"]
        for k, v in b8.items():
            if isinstance(v, dict) and "heldout_auc_mean" in v:
                L.append(f"| {k} | {v['heldout_auc_mean']:.3f} | {v['fpr_topical_loo']:.3f} | {v['topical_frac_of_e1']:.2f} | {v['cos_with_diff_of_means']:.3f} |")
        L.append("")
    b7 = R.get("b7_item_level", {})
    if b7:
        L += ["## B7 · AUC скора по явному ответу, ВНУТРИЯЧЕЙКОВЫЙ (Mantel–Haenszel)"]
        for q, v in b7.items():
            if isinstance(v, dict) and "auc_within_cell" in v:
                aw = v['auc_within_cell']; an = v['auc_naive_pooled_DO_NOT_USE']
                if not v['n_pairs_within']:
                    L.append(f"  {q}: — (нет ячеек, где ответ модели варьирует; внутриячейковых пар 0)")
                else:
                    L.append(f"  {q}: within-cell AUC {aw:.3f} по {v['n_pairs_within']} парам "
                             f"в {v['n_cells_with_variation']} ячейках "
                             f"(наивный пул {an:.3f} — не использовать)")
        L.append("")
    ls = R.get("length_substitutes", {})
    if ls:
        s1 = ls["within_external_length"]
        L += [f"## Заменители B10 · ρ(скор, длина) внутри Dolly = {s1['spearman_score_vs_len']:+.3f} "
              f"(n={s1['n']}, длины {s1['len_min']:.0f}–{s1['len_max']:.0f} слов)", f"  {ls['saturation_vs_foreign']['reading']}", ""]
    (out / "p4_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# ─────────────────────────────────────────── main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results"); ap.add_argument("--artifact", default="results/p4_frozen.json")
    ap.add_argument("--n-perm", type=int, default=200); ap.add_argument("--n-dirs", type=int, default=1000)
    a = ap.parse_args(argv)
    out = Path(a.out); nums = out / "numbers.json"
    n = json.loads(nums.read_text(encoding="utf-8")); L = int(n["probe.layer"])
    art = json.loads(Path(a.artifact).read_text(encoding="utf-8"))
    d = D(out, art); d.L = L
    w = direction(d, L); s = d.X[POS][L] @ w
    ask = load_ask(out)

    R = {"seed": SEED_P4, "layer": L, "position": POS, "n_records": d.N,
         "note": ("P4 (2026-09-01): B1–B9 + заменители B10 на ЗАМОРОЖЕННЫХ активациях. p2_analysis.py не "
                  "перезапускался, поток rng P2 не затронут; все числа P2 остаются битово теми же.")}
    R["b1_story"] = b1_story_contrast(d, s, art, n.get("probe.exploratory.p2.primary_pairs.mean_diff_frac_of_e1"))
    R["b2_bands"] = b2_bands(d, L, ("b", "c1", "c2", "len140", "len210"), a.n_dirs, a.n_perm)
    R["b4_selfyes_scores"] = b4_selfyes_scores(d, s, ask)
    R["b5_self_by_env"] = b5_self_by_env(ask)
    R["b6_precision_dolly"] = b6_precision_dolly(d, s)
    R["b3_dolly_enrichment"] = dolly_enrichment(d, s, out)
    R["b7_item_level"] = b7_item_level(d, s, ask)
    try:
        import sklearn, scipy
        R["env_versions"] = {"sklearn": sklearn.__version__, "scipy": scipy.__version__}
    except Exception: pass
    R["b8_lr_lda"] = b8_lr_lda(d, L)
    Xlu = d.X["last_user"][L]
    w_lu = Xlu[d.m("E1")].mean(0) - Xlu[d.dep()].mean(0)
    s_lu = Xlu @ w_lu
    R["b9_last_tokens"] = b9_last_tokens(d, s_lu, "last_user")
    R["b9_last_tokens"]["control_on_last"] = b9_last_tokens(d, s, "last")["global_eta2_excl_training"]
    R["length_substitutes"] = length_substitutes(d, s)
    # проверка иммутабельности P2 (заявление в note становится проверенным)
    probe_keys = [k for k in n if k.startswith("probe.exploratory.p2.")][:400]
    before = {k: n[k] for k in probe_keys}
    rec(nums, {"probe.exploratory.p4": R})
    after = json.loads(nums.read_text(encoding="utf-8"))
    changed = [k for k in probe_keys if after.get(k) != before[k]]
    rec(nums, {"probe.exploratory.p4.p2_unchanged": {"checked_keys": len(probe_keys), "changed": changed,
                                                     "ok": not changed}})
    assert not changed, f"P2-ключи изменились: {changed[:5]}"
    report(R, out)
    print("\nЗАПИСАНО под probe.exploratory.p4.* ; отчёт: results/p4_report.md")


if __name__ == "__main__":
    main()
