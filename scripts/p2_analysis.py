"""P2 — CPU-анализы (post pivot-4.2, 2026-09-01). Вне блоба; числа под probe.exploratory.p2.* / baseline.*.
Ядро — numpy, чтобы смоук-тест бежал без GPU (К-7). Порядок: Alt3-гейт → ячейки → первичный/вторичные контрасты →
поверхность precision → A1/A2/A4/A5 → В1–В4 → Alt1/2/4/6/7 → Б3 → агрегаты Б1 → отчёт. Все дроби — с явными n (К-3),
вырожденность — флагом (К-6), сиды — в numbers (К-5)."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POSITIONS = ("last", "last_user", "mean_user")
SEED = 20260901
# ВЕДУЩИЙ пул (сопоставим с prereg-числами 0.25/0.035): промпты, НЕ сконструированные под срабатывание
TOPICAL = ("a1", "a2", "d", "d_legacy", "a_legacy")
TOPICAL_DIAG = ("a_head", "a_tail")      # сконструированы под развилку «запрос vs фон» — только диагностика
NEUTRAL = ("n", "wild_other")            # dptrunc вынесен: после фикса это MCQ-структура (проверка формата, не нейтраль)
STRUCTURAL = ("dptrunc",)
LENGTH_LADDER = ("dptrunc", "len140", "len210")   # + deployment (полная длина) — кривая «скор vs длина» на одном тексте
ALL_CELLS = ("a_legacy", "d_legacy", "a1", "a2", "a_head", "a_tail", "d", "b", "c1", "c2", "n", "dptrunc", "len140", "len210", "wild_ai", "wild_other")
KEYS: list[str] = []


# ============================================================ утилиты
def _flat(k, v, out):
    if isinstance(v, dict):
        for kk, vv in v.items():
            _flat(f"{k}.{kk}", vv, out)
    else:
        out[k] = v


def sanitize(o):
    """numpy-скаляры → python; списки dict'ов → dict с индексами (record_numbers запрещает dict в списках)."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        if any(isinstance(x, (dict, list, tuple)) for x in o):
            return {f"i{i}": sanitize(x) for i, x in enumerate(o)}
        return [sanitize(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def rec(nums: Path, d: dict):
    from src.contracts import record_numbers
    d = sanitize(d)
    record_numbers(nums, d)
    cur = json.loads(nums.read_text(encoding="utf-8"))
    flat: dict = {}
    for k, v in d.items():
        _flat(k, v, flat)
    for k in flat:
        assert k in cur, f"ключ не записался: {k}"
        KEYS.append(k)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).mean() + 0.5 * (d == 0).mean()))


def hodges_lehmann(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    i, j = np.triu_indices(len(x))
    return float(np.median((x[i] + x[j]) / 2))


def boot_ci(x: np.ndarray, stat, n_boot: int, rng: np.random.Generator) -> list[float]:
    if len(x) < 2:
        return [float("nan"), float("nan")]
    vals = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [float("nan")] * 2


def perm_unpaired(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    obs = abs(a.mean() - b.mean()); pool = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool); cnt += abs(pool[:na].mean() - pool[na:].mean()) >= obs
    return (cnt + 1) / (n_perm + 1)


_SIGNS: dict[int, np.ndarray] = {}


def sign_perm_exact(d: np.ndarray) -> float:
    """Точный знаковый перестановочный тест (2^n расстановок), двусторонний; векторизован (n ≤ 20).
    float64 + ОТНОСИТЕЛЬНЫЙ допуск: с float32 и абсолютным 1e-9 экстремальные строки (все +1 / все −1)
    проваливали сравнение из-за расхождения np.mean и BLAS-dot (~1e-7) → возвращался p ниже пола, вплоть до 0.0."""
    d = np.asarray(d, dtype=np.float64); n = len(d); assert 1 <= n <= 20
    if n not in _SIGNS:
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
        _SIGNS[n] = (1 - 2 * bits).astype(np.float64)
    obs = abs(d.mean()); means = np.abs(_SIGNS[n] @ d) / n
    p = float((means >= obs - 1e-12 * max(obs, 1.0)).mean())
    assert p >= 2 ** (1 - n) - 1e-12, f"p={p} ниже пола 2/2^{n}"
    return p


def jonckheere(groups: list[np.ndarray], n_perm: int, rng: np.random.Generator) -> dict:
    """JT-статистика для убывающего тренда g0 > g1 > ... (считаем U(i<j) = #пар(x_i > x_j) + 0.5 ties)."""
    def stat(gs):
        s = 0.0
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                d = gs[i][:, None] - gs[j][None, :]
                s += (d > 0).sum() + 0.5 * (d == 0).sum()
        return s
    obs = stat(groups); sizes = [len(g) for g in groups]; pool = np.concatenate(groups); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool); gs = np.split(pool, np.cumsum(sizes)[:-1]); cnt += stat(gs) >= obs
    return {"JT": float(obs), "p_one_sided": (cnt + 1) / (n_perm + 1), "sizes": sizes}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def diff_of_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def tau_loo(e1: np.ndarray, q: float = 0.05) -> tuple[float, np.ndarray]:
    """Возвращает in-sample τ и вектор LOO-порогов (по одному на E1-точку)."""
    loo = np.array([np.quantile(np.delete(e1, i), q) for i in range(len(e1))])
    return float(np.quantile(e1, q)), loo


def precision_at(p: float, tpr: float, fpr: float) -> float:
    den = p * tpr + (1 - p) * fpr
    return (p * tpr / den) if den > 0 else float("nan")


# ============================================================ данные
class Data:
    """Единый фрейм: X[pos][layer] (N×d), cell, env, prompt_id, key, n_words, nll."""

    def __init__(self, base, extra, art, nll_rows, words: dict | None = None):
        keys = list(base["keys"]) + (list(extra["keys"]) if extra else [])
        self.keys = keys; self.N = len(keys)
        self.X = {p: {L: np.concatenate([np.asarray(base["acts"][p][L], dtype=np.float32)] +
                                        ([np.asarray(extra["acts"][p][L], dtype=np.float32)] if extra else []))
                      for L in base["layers"]} for p in POSITIONS}
        self.layers = list(base["layers"])
        self.env = [k.split("/")[0] for k in keys]; self.pid = [k.split("/")[1] for k in keys]; self.cond = [k.split("/")[2] for k in keys]
        pm = {p["prompt_id"]: p for p in art["prompts"]}
        a_leg = set(art["legacy_reclass"]["a_legacy"]); d_leg = set(art["legacy_reclass"]["d_legacy"])
        self.cell = []
        for e, pid, c in zip(self.env, self.pid, self.cond):
            if e == "topic":
                self.cell.append("a_legacy" if pid in a_leg else "d_legacy" if pid in d_leg else "topic_unclassified")
            elif e.startswith("x_"):
                self.cell.append(e[2:] if e != "x_dptrunc" else "dptrunc")
            else:
                self.cell.append(c)  # deployment/E1/S1/...
        self.cell = np.array(self.cell); self.envA = np.array(self.env); self.pidA = np.array(self.pid)
        nll = {r["prompt_key"]: r for r in nll_rows} if nll_rows else {}
        self.trend_order = tuple(art.get("cells_order_trend", ["a", "b", "c"]))
        words = words or {}
        self.n_words = np.array([float(words[k]) if k in words else (float(nll[k]["n_user_words"]) if k in nll else np.nan) for k in keys])
        self.nll = np.array([float(nll[k]["nll_user"]) if k in nll else np.nan for k in keys])

    def m(self, *cells) -> np.ndarray:
        return np.isin(self.cell, cells)

    def idx(self, cell, pid) -> int:
        w = np.where((self.cell == cell) & (self.pidA == pid))[0]; assert len(w) == 1, (cell, pid, len(w)); return int(w[0])


def direction(D: Data, pos: str, L: int, pos_cells=("E1",), neg_cells=("deployment",), extra_neg_mask=None) -> np.ndarray:
    X = D.X[pos][L]; mp = D.m(*pos_cells); mn = D.m(*neg_cells) & (D.envA != "topic") & ~np.char.startswith(D.envA.astype(str), "x_")
    if extra_neg_mask is not None:
        mn = mn | extra_neg_mask
    return X[mp].mean(0) - X[mn].mean(0)


def scores(D: Data, pos: str, L: int, w: np.ndarray, cosine: bool = False) -> np.ndarray:
    X = D.X[pos][L]
    if cosine:
        return (X @ w) / (np.linalg.norm(X, axis=1) * np.linalg.norm(w) + 1e-8)
    return X @ w


# ============================================================ анализы
def story_cluster_ci(D: Data, s: np.ndarray, story_map: dict, rng, n_boot: int) -> dict:
    """Кластерный бутстрап ПО СЮЖЕТАМ: a-семья = 8 сюжетов × до 3 вариантов, поэтому обычный бутстрап
    по 24 промптам занижает CI. Ресемплим сюжеты целиком."""
    if not story_map:
        return {"unavailable": True}
    pid2i = {D.pid[i]: i for i in range(D.N)}
    groups = [[pid2i[p] for p in v if p in pid2i] for v in story_map.values()]
    groups = [g for g in groups if g]
    dep = s[D.m("deployment")].mean(); e1_shift = s[D.m("E1")].mean() - dep
    obs = float(np.mean([s[i] for g in groups for i in g]) - dep) / e1_shift if e1_shift else float("nan")
    vals = []
    for _ in range(n_boot):
        pick = [groups[k] for k in rng.integers(0, len(groups), len(groups))]
        vals.append(float(np.mean([s[i] for g in pick for i in g]) - dep) / e1_shift if e1_shift else np.nan)
    vals = np.array([v for v in vals if not np.isnan(v)])
    return {"n_stories": len(groups), "n_prompts": sum(len(g) for g in groups), "frac_of_e1": obs,
            "cluster_ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [float("nan")] * 2,
            "note": "сюжетное n = число сюжетов, не промптов; CI по промптам был бы уже"}


def cell_table(D: Data, s: np.ndarray, cells: list[str], rng, n_boot: int) -> dict:
    dep = s[D.m("deployment")]; e1 = s[D.m("E1")]; e1_shift = e1.mean() - dep.mean()
    tau, loo = tau_loo(e1)
    tpr_loo = float(np.mean(e1 >= loo)); tpr_in = float(np.mean(e1 >= tau))
    out = {"e1_shift": float(e1_shift), "tau_insample": tau, "tpr_loo": tpr_loo, "tpr_insample": tpr_in}
    for c in cells:
        x = s[D.m(c)]
        if len(x) == 0:
            out[c] = {"n": 0, "degenerate": True}; continue
        fp = int((x >= tau).sum())
        fpr_loo = float(np.mean([(x >= t).mean() for t in loo]))
        sh = x - dep.mean()
        out[c] = {"n": int(len(x)), "mean_shift": float(sh.mean()), "hl_shift": hodges_lehmann(sh),
                  "shift_ci95_boot": boot_ci(sh, np.mean, n_boot, rng),
                  "frac_of_e1": float(sh.mean() / e1_shift) if e1_shift != 0 else float("nan"),
                  "fp_at_tau": fp, "fpr_at_tau": fp / len(x), "fpr_wilson95": wilson(fp, len(x)), "fpr_loo": fpr_loo,
                  "precision_at_1pct_loo": precision_at(0.01, tpr_loo, fpr_loo),
                  "auc_vs_deploy": auc(x, dep), "fired_ids": [D.pid[i] for i in np.where(D.m(c) & (s >= tau))[0]]}
    return out


def alt3_gate(D: Data, s: np.ndarray) -> dict:
    """Какие из исходных 8 topic сработали; формулировка заголовка — по заранее записанным исходам."""
    e1 = s[D.m("E1")]; tau, _ = tau_loo(e1)
    fired = [D.pid[i] for i in np.where((D.envA == "topic") & (s >= tau))[0]]
    a_leg = {D.pid[i] for i in np.where(D.cell == "a_legacy")[0]}
    a_f = [p for p in fired if p in a_leg]
    d_f = [p for p in fired if p not in a_f]
    na = int((D.cell == "a_legacy").sum()); nd = int((D.cell == "d_legacy").sum())
    if fired and not d_f:
        outcome = "A: сработали только ИИ-оценивание (a_legacy) → конъюнкция ИИ∧оценивание; заголовок про ИИ-оценивание чужих систем"
    elif d_f and not a_f:
        outcome = "B: сработали не-ИИ промпты → тема оценивания как таковая"
    elif not fired:
        outcome = "none: при in-sample τ ни один не сработал (несовместимо с ранее записанным 2/8 — проверить τ/позицию)"
    else:
        outcome = "C: серый — по одному из групп; заголовок через дозу-ответ новых ячеек"
    return {"tau": tau, "fired": fired, "fired_a_legacy": a_f, "fired_d_legacy": d_f, "n_a_legacy": na, "n_d_legacy": nd,
            "fpr_a_legacy": [len(a_f), na], "fpr_d_legacy": [len(d_f), nd], "outcome": outcome, "slot1_locked": True}


def primary_pairs(D: Data, s: np.ndarray, pairs: list, rng, n_boot: int, margin: float, texts: dict | None = None) -> dict:
    e1_shift = s[D.m("E1")].mean() - s[D.m("deployment")].mean()
    jacc, axes = [], []
    if texts:
        E1W = set("evaluation ai systems part note".split())
        AIRX = re.compile(r"\b(ai|assistant|model|chatbot|llm|language model)\b", re.I)
        STR = ["evaluat","assess","test","audit","review","benchmark","scor","grade","grading","examin","monitor","measur","rated","rating","judge","judging","red-?team","logged","logging"]
        def dens(x):
            low = x.lower(); n = max(len(low.split()), 1)
            return 100 * sum(len(re.findall(r"\b" + st + r"\w*", low)) for st in STR) / n
        for a, b in pairs:
            ta = texts.get(D.keys[D.idx("a1", a)], ""); tb = texts.get(D.keys[D.idx("a2", b)], "")
            A, B = set(ta.lower().split()), set(tb.lower().split())
            jacc.append(round(len(A & B) / max(len(A | B), 1), 3))
            axes.append({"pair": [a, b], "d_strict": round(dens(ta) - dens(tb), 2),
                         "d_overlap_e1": round(len(A & E1W) / max(len(E1W), 1) - len(B & E1W) / max(len(E1W), 1), 3),
                         "d_ai_mentions": len(AIRX.findall(ta)) - len(AIRX.findall(tb)),
                         "jaccard": jacc[-1]})
    d = np.array([s[D.idx("a1", a)] - s[D.idx("a2", b)] for a, b in pairs])
    dl = np.array([D.n_words[D.idx("a1", a)] - D.n_words[D.idx("a2", b)] for a, b in pairs])
    dn = np.array([D.nll[D.idx("a1", a)] - D.nll[D.idx("a2", b)] for a, b in pairs])
    frac = d / e1_shift if e1_shift != 0 else d * np.nan
    ci = boot_ci(frac, np.mean, n_boot, rng)
    tost = bool(ci[0] > -margin and ci[1] < margin) if not any(map(math.isnan, ci)) else False
    return {"n_pairs": len(pairs), "mean_diff": float(d.mean()), "hl_diff": hodges_lehmann(d), "mean_diff_frac_of_e1": float(np.nanmean(frac)),
            "frac_ci95_boot": ci, "p_sign_perm_exact": sign_perm_exact(d), "min_achievable_p": 2 / 2 ** len(pairs),
            "tost_margin_e1_units": margin, "equivalent_within_margin": tost,
            "spearman_diff_vs_len_diff": spearman(d, dl), "spearman_diff_vs_nll_diff": spearman(d, dn) if not np.isnan(dn).all() else float("nan"),
            "per_pair_frac": [float(x) for x in frac], "per_pair_jaccard": jacc, "per_pair_axes": axes,
            "axes_note": "манипуляция a1→a2 меняет ТРИ оси сразу: strict-плотность, перекрытие со словами вставки E1 (0.15→0.05) "
                         "и число упоминаний ИИ; при исходе a1>a2 разделить их нельзя — квалифицировать честно",
            "manipulation_note": "медианный Jaccard пар ≈ {:.2f} → переписано около двух третей слов: манипуляция НЕ минимальная (длины выровнены: Δ −3…+3)".format(float(np.median(jacc)) if jacc else float("nan")),
            "note": "антиконсервативность перцентильного бутстрапа при n=8; главный инференс — точный знаковый тест; отсутствие значимости ≠ равенство"}


def secondary(D: Data, s: np.ndarray, rng, n_perm: int, n_boot: int) -> dict:
    dep_mean = s[D.m("deployment")].mean()
    cellsets = {"a": ("a1", "a2"), "b": ("b",), "c": ("c1", "c2"), "c1": ("c1",), "c2": ("c2",)}
    order = D.trend_order
    g = [s[D.m(*cellsets[c])] for c in order]
    jt = jonckheere(g, n_perm, rng) if all(len(x) for x in g) else {"degenerate": True}
    jt["order"] = list(order)
    dd = s[D.m("d", "d_legacy")] - dep_mean; cc = s[D.m("c1", "c2")] - dep_mean
    d_vs_c = {"n_d": int(len(dd)), "n_c": int(len(cc)), "hl_d_minus_c": hodges_lehmann(dd) - hodges_lehmann(cc) if len(dd) and len(cc) else float("nan"),
              "mean_d_minus_c": float(dd.mean() - cc.mean()) if len(dd) and len(cc) else float("nan"),
              "p_perm": perm_unpaired(dd, cc, n_perm, rng) if len(dd) and len(cc) else float("nan")}
    bb = s[D.m("b")] - dep_mean; cc2 = s[D.m("c1", "c2")] - dep_mean
    b_vs_c = {"n_b": int(len(bb)), "n_c": int(len(cc2)),
              "mean_b_minus_c": float(bb.mean() - cc2.mean()) if len(bb) and len(cc2) else float("nan"),
              "hl_b_minus_c": (hodges_lehmann(bb) - hodges_lehmann(cc2)) if len(bb) and len(cc2) else float("nan"),
              "p_perm": perm_unpaired(bb, cc2, n_perm, rng) if len(bb) and len(cc2) else float("nan"),
              "note": "b и c одинаково нулевые по eval-лексике; AI-плотность b=1.17 vs c=0.00 — если b>c, это ось «ИИ», не «оценивание». Докладывать отдельно: JT может пройти от одного доминирующего a"}
    return {"jt_trend": jt, "d_vs_c": d_vs_c, "b_vs_c": b_vs_c}


def precision_surface(D: Data, s: np.ndarray, out: Path) -> dict:
    dep = s[D.m("deployment")]; e1 = s[D.m("E1")]; _, loo = tau_loo(e1); tpr = float(np.mean(e1 >= loo))
    topical = s[D.m(*TOPICAL)]; neutral = s[D.m(*NEUTRAL)]
    if len(neutral) == 0:
        neutral = s[D.m("c2")]; neutral_src = "c2 (fallback: tn/wild_other отсутствуют)"
    else:
        neutral_src = "tn+wild_other (wild_ai исключён по записанному regex; dptrunc вынесен в structural)"
    fpr_t = float(np.mean([(topical >= t).mean() for t in loo])); fpr_n = float(np.mean([(neutral >= t).mean() for t in loo]))
    tau_in = float(np.quantile(e1, 0.05)); fp_n = int((neutral >= tau_in).sum())
    ps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]; qs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    with open(out / "precision_surface.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["prevalence", "q_topic", "fpr", "precision"])
        for p in ps:
            for q in qs:
                fpr = q * fpr_t + (1 - q) * fpr_n; w.writerow([p, q, fpr, precision_at(p, tpr, fpr)])
    q_break = next((q for q in np.linspace(0, 1, 101) if precision_at(0.01, tpr, q * fpr_t + (1 - q) * fpr_n) < 0.5), None)
    diag = s[D.m(*TOPICAL_DIAG)]; struc = s[D.m(*STRUCTURAL)]
    extra = {"fpr_topical_diag_headtail": float(np.mean([(diag >= t2).mean() for t2 in loo])) if len(diag) else float("nan"),
             "n_topical_diag": int(len(diag)),
             "fpr_structural_dptrunc": float(np.mean([(struc >= t2).mean() for t2 in loo])) if len(struc) else float("nan"),
             "n_structural": int(len(struc)),
             "scope_note": "ведущий topical = a1,a2,d,d_legacy,a_legacy (не сконструированы под срабатывание); a_head/a_tail и dptrunc — отдельными строками"}
    return {**extra, "tpr_loo": tpr, "fpr_topical": fpr_t, "n_topical": int(len(topical)), "fpr_neutral": fpr_n, "n_neutral": int(len(neutral)),
            "fpr_neutral_wilson95": wilson(fp_n, len(neutral)), "fp_neutral_at_tau": fp_n,
            "neutral_degenerate": bool(len(neutral) < 10),
            "neutral_degenerate_note": "n_neutral < 10 → «FPR=0 на чистых негативах» НЕ утверждать: писать долю с Wilson-CI и n",
            "neutral_source": neutral_src, "q_topic_where_precision_at_1pct_below_0.5": q_break,
            "precision_at_1pct_q0": precision_at(0.01, tpr, fpr_n), "precision_at_1pct_q1": precision_at(0.01, tpr, fpr_t), "csv": "precision_surface.csv"}


def oos_scores(D: Data, pos: str, L: int) -> dict:
    """Leave-one-DP-out: убираем E1_i и deployment_i, перефитим w, скорим обе held-out записи → честные OOS-скоры.
    Закрывает тавтологию «FPR=0 на deployment при in-sample направлении и τ»."""
    X = D.X[pos][L]; e1_idx = np.where(D.m("E1"))[0]; dep_idx = np.where(D.m("deployment"))[0]
    pid_of = lambda i: (D.env[i], D.pid[i])
    dep_by = {pid_of(i): i for i in dep_idx}
    e1_oos, dep_oos = [], []
    for i in e1_idx:
        j = dep_by[pid_of(i)]
        me = D.m("E1").copy(); md = D.m("deployment").copy(); me[i] = False; md[j] = False
        w = X[me].mean(0) - X[md].mean(0); w = w / (np.linalg.norm(w) + 1e-8)   # общая шкала: 32 разных LOO-направления
        e1_oos.append(float(X[i] @ w)); dep_oos.append(float(X[j] @ w))
    e1_oos = np.array(e1_oos); dep_oos = np.array(dep_oos)
    tau_o = float(np.quantile(e1_oos, 0.05))
    out = {"n": int(len(e1_oos)), "tau_oos": tau_o, "tpr_oos": float((e1_oos >= tau_o).mean()), "fpr_deploy_oos": float((dep_oos >= tau_o).mean()),
           "fp_deploy_oos": int((dep_oos >= tau_o).sum()), "fpr_deploy_oos_wilson95": wilson(int((dep_oos >= tau_o).sum()), len(dep_oos)),
           "auc_oos": auc(e1_oos, dep_oos), "note": "все направления нормированы (единая шкала для квантиля по 32 LOO-векторам); СОДЕРЖАТЕЛЬНОЕ число здесь — только fpr_deploy_oos: tpr_oos ≈ 0.95 по построению (τ = 5%-квантиль тех же OOS E1-скоров)"}
    # ПОЛНОСТЬЮ OOS: каждый LOO-вектор со СВОИМ порогом применяется к topical → усреднение (третья, сопоставимая процедура)
    full_oos = {}
    for c in TOPICAL + TOPICAL_DIAG + STRUCTURAL + NEUTRAL:
        mc = D.m(c)
        if not mc.sum():
            continue
        fr = []
        for i in e1_idx:
            j = dep_by[pid_of(i)]
            me = D.m("E1").copy(); md = D.m("deployment").copy(); me[i] = False; md[j] = False
            w = X[me].mean(0) - X[md].mean(0); w = w / (np.linalg.norm(w) + 1e-8)
            s = X @ w; tau_i = float(np.quantile(s[me], 0.05))
            fr.append(float((s[mc] >= tau_i).mean()))
        full_oos[c] = float(np.mean(fr))
    out["fully_oos_fpr"] = full_oos
    out["three_procedures_note"] = ("три несопоставимые процедуры, цитировать раздельно: (1) cells_last.*.fpr_loo — направление на всех "
                                    "E1/deployment, τ LOO (консервативно); (2) fpr_*_at_tau_oos — τ из LOO-векторов на скорах полного w; "
                                    "(3) fully_oos_fpr — каждый LOO-вектор со своим τ (ведущая)")
    w_full = X[D.m("E1")].mean(0) - X[D.m("deployment")].mean(0); w_full = w_full / (np.linalg.norm(w_full) + 1e-8); s_full = X @ w_full
    for c in TOPICAL + TOPICAL_DIAG + STRUCTURAL + NEUTRAL + ("b", "c1", "c2", "wild_ai"):
        x = s_full[D.m(c)]
        if len(x):
            out[f"fpr_{c}_at_tau_oos"] = float((x >= tau_o).mean())
    # кросс-среда: w на одной среде → deployment другой
    ce = {}
    for tr in ("sandbagging", "secret_number"):
        te = "secret_number" if tr == "sandbagging" else "sandbagging"
        mtr_e = D.m("E1") & (D.envA == tr); mtr_d = D.m("deployment") & (D.envA == tr)
        w = X[mtr_e].mean(0) - X[mtr_d].mean(0); w = w / (np.linalg.norm(w) + 1e-8); s = X @ w
        tau_tr = float(np.quantile(s[mtr_e], 0.05)); d_other = s[D.m("deployment") & (D.envA == te)]; e_other = s[D.m("E1") & (D.envA == te)]
        ce[f"train_{tr}"] = {"fpr_deploy_other": float((d_other >= tau_tr).mean()), "tpr_e1_other": float((e_other >= tau_tr).mean()), "auc_other": auc(e_other, d_other)}
    out["cross_env"] = ce
    return out


def head_tail(D: Data, s: np.ndarray, rng, n_perm: int, lex_strict: dict | None = None, lex_ai: dict | None = None, lex_ai_broad: dict | None = None) -> dict:
    """Развилка «читает хвост» vs «читает тему»: a_head (eval-контекст + нейтральный хвост) vs a_tail (нейтральный контекст + eval-хвост)."""
    dep = s[D.m("deployment")]; e1 = s[D.m("E1")]; sh = e1.mean() - dep.mean(); _, loo = tau_loo(e1)
    out = {}
    for c in ("a_head", "a_tail"):
        x = s[D.m(c)]
        out[c] = {"n": int(len(x)), "frac_of_e1": float((x.mean() - dep.mean()) / sh) if len(x) and sh else float("nan"),
                  "fpr_loo": float(np.mean([(x >= q).mean() for q in loo])) if len(x) else float("nan")}
    a, b = s[D.m("a_tail")], s[D.m("a_head")]
    out["tail_minus_head_frac"] = float((a.mean() - b.mean()) / sh) if len(a) and len(b) and sh else float("nan")
    out["p_perm"] = perm_unpaired(a, b, n_perm, rng) if len(a) and len(b) else float("nan")
    # лексические ковариаты внутри 16 промптов head/tail (ячейки не сбалансированы: strict 6.73 vs 4.37, AI 1.08 vs 2.20)
    if lex_strict is not None:
        m16 = D.m("a_head", "a_tail")
        xs = np.array([lex_strict.get(k, np.nan) for k in D.keys])[m16]
        xa = np.array([lex_ai.get(k, np.nan) for k in D.keys])[m16] if lex_ai else np.full(m16.sum(), np.nan)
        y16 = s[m16]
        out["within16_spearman_score_vs_strict"] = spearman(xs[~np.isnan(xs)], y16[~np.isnan(xs)]) if (~np.isnan(xs)).sum() >= 3 else float("nan")
        out["within16_spearman_score_vs_ai_narrow"] = spearman(xa[~np.isnan(xa)], y16[~np.isnan(xa)]) if (~np.isnan(xa)).sum() >= 3 else float("nan")
        xb = np.array([lex_ai_broad.get(k, np.nan) for k in D.keys])[m16] if lex_ai_broad else np.full(m16.sum(), np.nan)
        out["within16_spearman_score_vs_ai_broad"] = spearman(xb[~np.isnan(xb)], y16[~np.isnan(xb)]) if (~np.isnan(xb)).sum() >= 3 else float("nan")
        out["domain_confound_note"] = ("a_head = ИИ-индустриальные сюжеты + нейтральная просьба, a_tail = бытовые сюжеты + eval-просьба: "
                                       "к лексическому дисбалансу добавляется конфаунд ДОМЕНА. Внутрисюжетное сравнение — только "
                                       "сценарные пары tn↔at (n=4); развилка 2 подаётся как диагностика")
        out["density_note"] = "ячейки не сбалансированы: strict head 6.73 / tail 4.37, AI head 1.08 / tail 2.20 — обе плотности в обе ветки формулировки"
    out["within_cell_variance_note"] = "внутри a_head все просьбы «помоги написать/спланировать», внутри a_tail все «как бы ты это оценил» — эффективное n меньше 8"
    out["reading"] = ("контраст = eval-содержание В ЗАПРОСЕ (a_tail) vs В ФОНЕ (a_head), а не «позиция токенов»: "
                      "у a_head просьба нейтральная (структурируй письмо), у a_tail просьба про оценивание. "
                      "tail ≫ head → монитор реагирует на предмет запроса; head ≈ tail → на тему в любом месте; обе ячейки n=8")
    return out


SCENARIO_PAIRS = [("tn_001", "at_001"), ("tn_002", "at_002"), ("tn_003", "at_003"), ("tn_004", "at_004")]


def bands_on_cells(D: Data, pos: str, L: int, n_dirs: int = 1000, n_perm: int = 200, seed: int = 0) -> dict:
    """Полоса случайных единичных направлений и перестановочная полоса (те же функции, что в prereg-воротах v3,
    probe_toy.random_band/perm_band; здесь применены к НОВЫМ ячейкам). Отвечает на «а любое ли направление
    разводит короткие тематические промпты»."""
    try:
        import torch
        from src.probe_toy import perm_band, random_band
    except Exception as e:  # noqa: BLE001 — смоук вне Kaggle бежит без torch
        return {"unavailable": f"{type(e).__name__}: {str(e)[:80]}"}
    X = torch.from_numpy(D.X[pos][L]); mdep = D.m("deployment") & (D.envA != "topic") & ~np.char.startswith(D.envA.astype(str), "x_")
    contrast = mdep | D.m("E1")
    yc = torch.tensor([1.0 if D.cell[i] == "E1" else 0.0 for i in range(D.N) if contrast[i]])
    Xc = X[torch.from_numpy(contrast)]
    pair_ids = [f"{D.env[i]}/{D.pid[i]}" for i in range(D.N) if contrast[i]]
    Xref = X[torch.from_numpy(mdep)]
    out = {}
    for c in TOPICAL + TOPICAL_DIAG + STRUCTURAL + NEUTRAL:
        mc = D.m(c)
        if not mc.sum():
            continue
        Xt = X[torch.from_numpy(mc)]
        rb = random_band(Xc, yc, Xt, Xref, n_dirs=n_dirs, seed=seed)
        pb = perm_band(Xc, yc, pair_ids, Xt, Xref, n_perm=n_perm, seed=seed)
        out[c] = {"random_band": {k: float(v) for k, v in rb.items()}, "perm_band": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in pb.items()}}
    out["note"] = "функции из prereg-блока (ворота v3), применены к новым ячейкам; полоса определена до данных, не подобрана сейчас"
    return out


def density_regression(D: Data, s: np.ndarray, ev: dict, ai_n: dict, ai_b: dict) -> dict:
    """DESCRIPTIVE: score ~ eval_density + ai_density + n_words по всем новым ячейкам сразу.
    Стандартизованные коэффициенты + частные ранговые корреляции. НЕ инференс: промпты сгруппированы по
    конструкции, точки не независимы. Дополняет (не заменяет) парный контраст a1−a2, где дизайн матчирован."""
    cells = TOPICAL + TOPICAL_DIAG + STRUCTURAL + NEUTRAL + ("b", "c1", "c2")
    m = D.m(*cells) & ~np.isnan(D.n_words)
    keys = [D.keys[i] for i in np.where(m)[0]]
    Xd = np.column_stack([[ev.get(k, np.nan) for k in keys], [ai_n.get(k, np.nan) for k in keys],
                          [ai_b.get(k, np.nan) for k in keys], D.n_words[m]])
    y = s[m]; ok = ~np.isnan(Xd).any(1)
    Xd, y, keys = Xd[ok], y[ok], [k for k, o in zip(keys, ok) if o]
    if len(y) < 10:
        return {"degenerate": True, "n": int(len(y))}
    names = ["eval_density", "ai_narrow", "ai_broad", "n_words"]
    Z = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-9); yz = (y - y.mean()) / (y.std() + 1e-9)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z]), yz, rcond=None)
    r2 = 1 - ((yz - np.column_stack([np.ones(len(Z)), Z]) @ beta) ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
    partial = {}
    for i, nm in enumerate(names):
        others = np.column_stack([np.ones(len(Z))] + [Z[:, j] for j in range(len(names)) if j != i])
        rx = Z[:, i] - others @ np.linalg.lstsq(others, Z[:, i], rcond=None)[0]
        ry = yz - others @ np.linalg.lstsq(others, yz, rcond=None)[0]
        partial[nm] = spearman(rx, ry)
    vif = {}
    for i, nm in enumerate(names):
        others = np.column_stack([np.ones(len(Z))] + [Z[:, j] for j in range(len(names)) if j != i])
        pred = others @ np.linalg.lstsq(others, Z[:, i], rcond=None)[0]
        r2i = 1 - ((Z[:, i] - pred) ** 2).sum() / max(((Z[:, i] - Z[:, i].mean()) ** 2).sum(), 1e-9)
        vif[nm] = float(1 / max(1 - r2i, 1e-6))
    return {"n": int(len(y)), "r2": float(r2), "std_beta": {nm: float(b) for nm, b in zip(names, beta[1:])},
            "partial_spearman": partial, "vif": vif,
            "note": "DESCRIPTIVE: точки не независимы (сгруппированы по конструкции), плотности коллинеарны (см. vif); "
                    "дополняет, а не заменяет матчированный контраст a1−a2"}


def scenario_pairs(D: Data, s: np.ndarray, pairs=None, diffs: dict | None = None) -> dict:
    """DESCRIPTIVE: четыре одних и тех же сюжета (пекарня/аренда/бухгалтерия/спортзал) с нейтральной и eval-просьбой.
    Контекст тот же ±несколько слов (перефразирован, word-diff прилагается по каждой паре), меняется в основном просьба.
    n=4 → минимальный достижимый двусторонний p = 0.125; это иллюстрация, не тест."""
    pairs = [tuple(x) for x in (pairs or SCENARIO_PAIRS)]
    dep = s[D.m("deployment")]; sh = s[D.m("E1")].mean() - dep.mean()
    rows, d = [], []
    for tn, at in pairs:
        try:
            i, j = D.idx("n", tn), D.idx("a_tail", at)
        except AssertionError:
            continue
        dd = float(s[j] - s[i]); d.append(dd)
        row = {"scenario": tn, "neutral_frac": float((s[i] - dep.mean()) / sh) if sh else float("nan"),
               "evaltail_frac": float((s[j] - dep.mean()) / sh) if sh else float("nan"), "diff_frac": dd / sh if sh else float("nan")}
        if diffs and (tn, at) in diffs:
            row.update(diffs[(tn, at)])
        rows.append(row)
    d = np.array(d)
    return {"pairs": rows, "n": int(len(d)), "mean_diff_frac_of_e1": float(d.mean() / sh) if len(d) and sh else float("nan"),
            "same_direction": int((d > 0).sum()), "p_sign_perm_exact": sign_perm_exact(d) if len(d) else float("nan"),
            "note": "descriptive; контексты перефразированы (см. words_only_in_* в каждой паре), n=4 → min p = 0.125"}


def a1_loio(D: Data, pos: str, L: int) -> dict:
    ins = ["E1", "e2_a", "e2_b", "e2_c"]; out = {}
    topic_mask = D.m(*TOPICAL)
    def evalw(w, label):
        s = scores(D, pos, L, w); dep = s[D.m("deployment")]; e1 = s[D.m("E1")]; sh = e1.mean() - dep.mean()
        _, loo = tau_loo(e1); t = s[topic_mask]
        return {"auc_e1_vs_deploy": auc(e1, dep), "topic_frac_of_e1": float((t.mean() - dep.mean()) / sh) if sh else float("nan"),
                "fpr_topic_loo": float(np.mean([(t >= q).mean() for q in loo]))}
    sysneg = D.m("sys_neutral")  # e2_b — system-канал: без sys_neutral в негативах направление впитает «есть system»
    out["pool_all_inserts"] = evalw(direction(D, pos, L, pos_cells=tuple(ins), extra_neg_mask=sysneg), "pool")
    for held in ins:
        train = tuple(i for i in ins if i != held); w = direction(D, pos, L, pos_cells=train, extra_neg_mask=(sysneg if "e2_b" in train else None))
        s = scores(D, pos, L, w); h = s[D.m(held)]; dep = s[D.m("deployment")]
        out[f"loio_holdout_{held}"] = {**evalw(w, held), "auc_holdout_vs_deploy": auc(h, dep)}
    out["note"] = "topic-промпты только тестовые; e2_b — system-канал"
    return out


def a5_hard_negative(D: Data, pos: str, L: int, pairs: list) -> dict:
    X = D.X[pos][L]; dep = D.m("deployment") & (D.envA != "topic") & ~np.char.startswith(D.envA.astype(str), "x_")
    e1 = D.m("E1"); topical_cells = ["a1", "a2", "a_head", "a_tail", "b", "c1", "c2", "d", "d_legacy", "a_legacy"]
    def fit(train_mask):
        # равные веса групп: 32 длинных deployment и ~50 коротких topical иначе смешиваются в один центроид → направление длины
        neg = 0.5 * X[dep].mean(0) + 0.5 * X[train_mask].mean(0) if train_mask.sum() else X[dep].mean(0)
        return X[e1].mean(0) - neg
    def fpr_on(w, mask):
        s = X @ w; e = s[e1]; _, loo = tau_loo(e); t = s[mask]
        return float(np.mean([(t >= q).mean() for q in loo])) if len(t) else float("nan"), auc(e, s[dep])
    res = {}
    # baseline (без hard negatives)
    w0 = X[e1].mean(0) - X[dep].mean(0)
    res["no_hard_neg"] = {"fpr_topic_loo": fpr_on(w0, D.m(*topical_cells))[0], "auc": fpr_on(w0, D.m(*topical_cells))[1]}
    # LOCO
    loco = {}
    groups = {"a1": ("a1",), "a2": ("a2",), "a_head": ("a_head",), "a_tail": ("a_tail",), "b": ("b",), "c": ("c1", "c2"),
              "d": ("d", "d_legacy"), "a_legacy": ("a_legacy",), "a_family": ("a1", "a2", "a_head", "a_tail", "a_legacy")}
    for held, hc in groups.items():
        train = D.m(*[c for c in topical_cells if c not in hc])
        w = fit(train); f, a = fpr_on(w, D.m(*hc)); loco[held] = {"fpr_holdout_loo": f, "auc_e1_vs_deploy": a, "n_holdout": int(D.m(*hc).sum())}
    res["loco"] = loco
    # LOPO по парам a1/a2
    lopo = []
    for a, b in pairs:
        ia, ib = D.idx("a1", a), D.idx("a2", b)
        train = D.m(*topical_cells).copy(); train[ia] = False; train[ib] = False
        w = fit(train); s = X @ w; e = s[e1]; _, loo = tau_loo(e)
        lopo.append({"pair": [a, b], "fired_a1": float(np.mean(s[ia] >= loo)), "fired_a2": float(np.mean(s[ib] >= loo))})
    res["lopo_pairs"] = lopo
    res["lopo_fpr_a1"] = float(np.mean([x["fired_a1"] for x in lopo])); res["lopo_fpr_a2"] = float(np.mean([x["fired_a2"] for x in lopo]))
    res["rule"] = ("ровно один вариант (diff-of-means с hard negatives, равные веса групп). ВЕДУЩИЙ — LOPO по парам a1/a2 "
                   "(LOCO по одной ячейке течёт: держим a1, её парафраз a2 остаётся в негативах) и LOCO 'a_family' "
                   "(вся a-семья делит домен). Критерий развилки 3 применять к ним, не к однокелейному LOCO.")
    return res


def lexicon_scores(recs_text: dict, lex: dict, which: str, exclude_checklist: bool) -> dict:
    stems = lex[which]; excl = ["checkpoint"] + (["checklist"] if exclude_checklist else [])
    out = {}
    for key, text in recs_text.items():
        low = text.lower()
        for e in excl:
            low = re.sub(r"\b" + e + r"\w*", " ", low)
        n = max(len(low.split()), 1); hits = 0
        for st in stems:
            hits += len(re.findall(r"\b" + st + r"\w*", low)) if " " not in st else len(re.findall(r"\b" + re.escape(st) + r"\b", low))
        out[key] = 100.0 * hits / n
    return out


def a2_grep(D: Data, texts_user: dict, texts_full: dict, lex: dict) -> dict:
    res = {}
    for chan, texts in (("user_only", texts_user), ("full_prompt", texts_full)):
        for which in ("eval_strict", "eval_broad", "eval_adjacent"):
            for excl in (False, True):
                sc = lexicon_scores(texts, lex, which, excl); s = np.array([sc.get(k, np.nan) for k in D.keys])
                dep = s[D.m("deployment")]; e1 = s[D.m("E1")]
                tau, loo = tau_loo(e1)
                row = {"auc_e1_vs_deploy": auc(e1, dep), "tau": tau}
                for c in ALL_CELLS:
                    x = s[D.m(c)]; x = x[~np.isnan(x)]
                    if len(x):
                        row[f"fpr_{c}"] = float(np.mean([(x >= q).mean() for q in loo])); row[f"density_median_{c}"] = float(np.median(x))
                res[f"{chan}.{which}.{'excl_checklist' if excl else 'keep_checklist'}"] = row
    res["headline_scope"] = ("под eval_strict a1/a2 = 7.61/0.00 (различие 100%), под eval_adjacent = 5.46/5.07 (~7%): "
                             "корректный вывод из a1≈a2 — «загрязнение не сводится к 19 ЗАРЕГИСТРИРОВАННЫМ стемам», "
                             "не «семантическое». a2 по смежному словарю — самая насыщенная ячейка набора.")
    res["a2_zero_density_claim"] = ("у a2 плотность eval_strict = 0 ПО ПОСТРОЕНИЮ, поэтому любое срабатывание пробе на a2 "
                                    "нелексично по определению — этот вывод НЕ зависит от порога τ")
    res["tau_caveat"] = ("τ-версия сравнения вторична: sandbagging-deployment сам содержит 'score higher than 50%' (≈0.33/100 слов), "
                         "что выше 5%-квантиля E1 (≈0.29) → grep срабатывает на всех 16 sb-deployment и его AUC заметно ниже 1.0. "
                         "Это свойство обучающего контраста, а не подгонка порога.")
    res["note"] = "манипуляция a1/a2 определена относительно eval_strict; вывод «grep проваливается на a2» — только strict"
    return res


def positions_table(D: Data, L: int, rng, n_boot: int) -> dict:
    out = {}
    for p in POSITIONS:
        w = direction(D, p, L); nrm = float(np.linalg.norm(w)); s = scores(D, p, L, w)
        t = cell_table(D, s, list(ALL_CELLS), rng, n_boot)
        topical = s[D.m(*TOPICAL)]; dep = s[D.m("deployment")]; e1 = s[D.m("E1")]; _, loo = tau_loo(e1)
        fpr = float(np.mean([(topical >= q).mean() for q in loo])) if len(topical) else float("nan")
        out[p] = {"w_norm": nrm, "auc_e1_vs_deploy": auc(e1, dep), "e1_shift": t["e1_shift"], "tpr_loo": t["tpr_loo"],
                  "fpr_topical_loo": fpr, "precision_at_1pct": precision_at(0.01, t["tpr_loo"], fpr),
                  "topical_frac_of_e1": float((topical.mean() - dep.mean()) / t["e1_shift"]) if t["e1_shift"] else float("nan"),
                  "degenerate": bool(nrm < 1e-3 * np.linalg.norm(D.X[p][L][D.m("deployment")], axis=1).mean())}
    return out


def objections(D: Data, s: np.ndarray, e1_words: set, texts_user: dict, pairs: list, rng, n_perm: int) -> dict:
    """NB: знаменатель overlap = число содержательных слов вставки (≈5), поэтому overlap принимает мало значений
    и Спирмен на 8 точках почти весь в связках — доложить как ограничение."""
    out = {}
    # В1 перекрытие с E1-вставкой (внутри ячеек и на парах)
    def overlap(key):
        ws = set(re.findall(r"[a-z]+", texts_user.get(key, "").lower())) - STOP
        return len(ws & e1_words) / max(len(e1_words), 1)
    ov = np.array([overlap(k) for k in D.keys])
    v1 = {}
    for c in ("a1", "a2", "a_head", "a_tail", "b", "c1", "c2", "d", "d_legacy", "n"):
        m = D.m(c)
        if not m.sum():
            continue
        v1[f"spearman_{c}"] = spearman(ov[m], s[m]) if m.sum() >= 3 else float("nan")
        v1[f"overlap_mean_{c}"] = float(np.mean(ov[m])); v1[f"overlap_max_{c}"] = float(np.max(ov[m]))
        v1[f"overlap_argmax_{c}"] = D.pid[int(np.where(m)[0][int(np.argmax(ov[m]))])]
    d_ov = np.array([ov[D.idx("a1", a)] - ov[D.idx("a2", b)] for a, b in pairs]); d_s = np.array([s[D.idx("a1", a)] - s[D.idx("a2", b)] for a, b in pairs])
    v1["pairs_spearman_overlapdiff_vs_scorediff"] = spearman(d_ov, d_s); v1["pairs_mean_overlap_a1"] = float(np.mean([ov[D.idx("a1", a)] for a, _ in pairs]))
    v1["pairs_mean_overlap_a2"] = float(np.mean([ov[D.idx("a2", b)] for _, b in pairs]))
    out["V1_lexical_overlap"] = v1
    # В2 пороги
    dep = s[D.m("deployment")]; e1 = s[D.m("E1")]; topical = s[D.m(*TOPICAL)]
    v2 = {"auc_e1_vs_topical": auc(e1, topical)}
    for tpr_t in (0.80, 0.90, 0.95, 0.99):
        tau = float(np.quantile(e1, 1 - tpr_t)); fpr = float((topical >= tau).mean())
        v2[f"tpr_{tpr_t}"] = {"fpr_topical": fpr, "precision_at_1pct": precision_at(0.01, tpr_t, fpr)}
    out["V2_thresholds"] = v2
    # В3 длина внутри ячеек
    v3 = {}
    for c in ("deployment", "E1", "a1", "a2", "a_head", "a_tail", "b", "c1", "c2", "d", "d_legacy"):
        m = D.m(c) & ~np.isnan(D.n_words); v3[f"spearman_len_{c}"] = spearman(D.n_words[m], s[m]) if m.sum() >= 3 else float("nan")
        v3[f"len_median_{c}"] = float(np.nanmedian(D.n_words[D.m(c)])) if D.m(c).sum() else float("nan")
    mn = ~np.isnan(D.nll)
    for c in ("a1", "a2", "a_head", "a_tail", "d", "d_legacy", "b", "c1", "c2"):
        m = D.m(c) & mn; v3[f"spearman_nll_{c}"] = spearman(D.nll[m], s[m]) if m.sum() >= 3 else float("nan"); v3[f"nll_median_{c}"] = float(np.nanmedian(D.nll[D.m(c)])) if D.m(c).sum() else float("nan")
    out["V3_length_nll"] = v3
    return out


STOP = set("the a an and or of to in on for with we our i my is are be it this that as at by from how what which can could you help me".split())


def confirmatory_checks(D: Data, s: np.ndarray, rng, n_perm: int) -> dict:
    """Alt6: знаки/значимость каждой дельты; В4: LOPO для S−Dstr внутри среды."""
    out = {}
    for env in ("sandbagging", "secret_number"):
        pids = sorted({p for p, e, c in zip(D.pid, D.env, D.cond) if e == env and c == "deployment"})
        def d_of(c):
            return np.array([s[D.idx(c, p)] - s[D.idx("deployment", p)] for p in pids])
        e1 = d_of("E1"); r = {"n": len(pids)}
        for c in ("S1", "S2", "Dstr", "E1_S1"):
            dc = d_of(c); r[c] = {"mean": float(dc.mean()), "frac_of_e1": float(dc.mean() / e1.mean()) if e1.mean() else float("nan"),
                                 "p_sign_perm": sign_perm_exact(dc) if len(dc) <= 16 else float("nan"), "sign": int(np.sign(dc.mean()))}
        for sc in ("S1", "S2"):
            dd = d_of(sc) - d_of("Dstr"); p_all = sign_perm_exact(dd)
            lopo = [sign_perm_exact(np.delete(dd, i)) for i in range(len(dd))]
            r[f"{sc}_minus_Dstr"] = {"mean": float(dd.mean()), "p_sign_perm": p_all, "lopo_p_max": float(max(lopo)), "lopo_all_below_0.05": bool(max(lopo) < 0.05)}
        out[env] = r
    out["reading"] = "формулировка: S ОТЛИЧИМЫ от структурного контроля; не «S двигают пробе вверх» (Dstr сам отрицателен)"
    return out


def alt1_alt2(D: Data, pos: str, L: int, pairs: list, rng, n_boot: int) -> dict:
    out = {}
    w = direction(D, pos, L)
    # Alt1 косинусы
    sc = scores(D, pos, L, w, cosine=True)
    t = cell_table(D, sc, list(ALL_CELLS), rng, n_boot)
    out["Alt1_cosine"] = {"tpr_loo": t["tpr_loo"], **{c: {"frac_of_e1": t[c].get("frac_of_e1"), "fpr_loo": t[c].get("fpr_loo")} for c in ("a1", "a2", "a_head", "a_tail", "d", "d_legacy", "a_legacy", "b", "c2", "n")},
                          "pairs_mean_diff_frac": primary_pairs(D, sc, pairs, rng, 200, 0.25)["mean_diff_frac_of_e1"],
                          "norm_topical_over_deploy": float(np.linalg.norm(D.X[pos][L][D.m(*TOPICAL)], axis=1).mean() / np.linalg.norm(D.X[pos][L][D.m("deployment")], axis=1).mean())}
    # Alt2 анизотропия: PC1..2 на deployment
    Xd = D.X[pos][L][D.m("deployment") & (D.envA != "topic") & ~np.char.startswith(D.envA.astype(str), "x_")]
    mu = Xd.mean(0); U, S, Vt = np.linalg.svd(Xd - mu, full_matrices=False); pcs = Vt[:2]
    cos_pc1 = float(abs(w @ pcs[0]) / (np.linalg.norm(w) * np.linalg.norm(pcs[0]) + 1e-8))
    X = D.X[pos][L] - mu; Xp = X - (X @ pcs.T) @ pcs
    yc = D.m("E1"); dep = D.m("deployment") & (D.envA != "topic") & ~np.char.startswith(D.envA.astype(str), "x_")
    wp = Xp[yc].mean(0) - Xp[dep].mean(0); sp = Xp @ wp
    e1 = sp[yc]; dp = sp[dep]; sh = e1.mean() - dp.mean(); _, loo = tau_loo(e1); topical = sp[D.m(*TOPICAL)]
    out["Alt2_anisotropy"] = {"cos_w_pc1": cos_pc1, "pc1_var_frac": float(S[0] ** 2 / (S ** 2).sum()),
                              "after_proj_out_pc1_2": {"auc_e1_vs_deploy": auc(e1, dp), "topical_frac_of_e1": float((topical.mean() - dp.mean()) / sh) if sh else float("nan"),
                                                       "fpr_topical_loo": float(np.mean([(topical >= q).mean() for q in loo]))}}
    return out


def alt4_lexicon_S(D: Data, texts_user: dict, lex: dict) -> dict:
    sc = lexicon_scores(texts_user, lex, "eval_strict", False); s = np.array([sc.get(k, np.nan) for k in D.keys])
    return {c: {"density_median": float(np.nanmedian(s[D.m(c)])), "n": int(D.m(c).sum())} for c in ("deployment", "E1", "S1", "S2", "Dstr", "E1_S1")}


def layers_check(D: Data, rng, n_boot: int) -> dict:
    out = {}
    for L in D.layers:
        w = direction(D, "last", L); s = scores(D, "last", L, w); dep = s[D.m("deployment")]; e1 = s[D.m("E1")]; sh = e1.mean() - dep.mean()
        topical = s[D.m(*TOPICAL)]; _, loo = tau_loo(e1)
        out[f"L{L}"] = {"auc_e1_vs_deploy": auc(e1, dep), "topical_frac_of_e1": float((topical.mean() - dep.mean()) / sh) if sh else float("nan"),
                        "fpr_topical_loo": float(np.mean([(topical >= q).mean() for q in loo])) if len(topical) else float("nan")}
    return out


def ask_model_agg(rows: list[dict]) -> dict:
    if not rows:
        return {"present": False}
    return {q: _ask_group(rows, q) for q in ("main", "self", "inverted")} | {"present": True,
            "acquiescence_note": "yes и на main/self, и на inverted = поддакивание, а не знание; смотреть разности, не уровни",
            "format_note": "вопрос приклеен вторым подряд user-ходом (prefill без ассистентского ответа) — off-distribution формат поверх уже записанного конфаунда «сам вопрос бенчмарк-подобен»; рендер одинаков во всех условиях, поэтому разности интерпретируемы"}


def _ask_group(rows: list[dict], qname: str) -> dict:
    main = [r for r in rows if r["kind"] == qname]; groups = {}
    for r in main:
        g = r["env"][2:] if r["env"].startswith("x_") else ("topic_legacy" if r["env"] == "topic" else r["condition_id"])
        a = groups.setdefault(g, {"yes": 0, "no": 0, "unparsable": 0}); a[r["answer"]] += 1
    res = {"groups": {}}
    for g, a in groups.items():
        n = a["yes"] + a["no"]; res["groups"][g] = {**a, "n_parsable": n, "yes_rate": a["yes"] / n if n else None, "wilson95": wilson(a["yes"], n)}
    yr = {g: v["yes_rate"] for g, v in res["groups"].items() if v["yes_rate"] is not None}
    if "E1" in yr and "deployment" in yr:
        res["diff_E1_minus_deployment"] = yr["E1"] - yr["deployment"]
        for g in ("a1", "a2", "a_head", "a_tail", "d", "b", "c1", "c2", "n", "wild_ai", "wild_other", "topic_legacy"):
            if g in yr:
                res[f"diff_{g}_minus_deployment"] = yr[g] - yr["deployment"]
    res["reading"] = "только разности между условиями; потолок yes во всех условиях = результат про формат вопроса"
    return res


# ============================================================ отчёт
def md_report(R: dict, out: Path):
    L = ["# P2 report (EXPLORATORY, post pivot-4.2)", ""]
    g = R["alt3_gate"]; L += [f"## Alt3-гейт: {g['outcome']}", f"fired={g['fired']} (a_legacy {g['fpr_a_legacy']}, d_legacy {g['fpr_d_legacy']})", ""]
    pp = R["primary_pairs"]; L += ["## Первичный контраст a1−a2", f"mean diff = {pp['mean_diff_frac_of_e1']:.3f} E1-единиц (CI {pp['frac_ci95_boot']}), p_exact={pp['p_sign_perm_exact']:.4f}, TOST±{pp['tost_margin_e1_units']}: {pp['equivalent_within_margin']}", ""]
    ct = R["cells_last"]; L += ["## Ячейки (last, L*)", "| cell | n | frac_of_E1 | FPR@τ (LOO) | precision@1% (n/a при FP=0 → k/n + Wilson) | AUC vs dep |", "|---|---|---|---|---|---|"]
    for c in ALL_CELLS:
        r = ct.get(c, {})
        if r.get("n"):
            prec = (f"{r['precision_at_1pct_loo']:.3f}" if r["fp_at_tau"] > 0
                    else f"n/a ({r['fp_at_tau']}/{r['n']}, Wilson {r['fpr_wilson95'][0]:.2f}–{r['fpr_wilson95'][1]:.2f})")
            L.append(f"| {c} | {r['n']} | {r['frac_of_e1']:.2f} | {r['fpr_loo']:.2f} | {prec} | {r['auc_vs_deploy']:.2f} |")
    ps = R["precision_surface"]; L += ["", f"## Поверхность precision: FPR_topical={ps['fpr_topical']:.2f} (n={ps['n_topical']}), FPR_neutral={ps['fpr_neutral']:.2f} (n={ps['n_neutral']}, {ps['neutral_source']}); precision@1% ломается ниже 0.5 при q_topic ≥ {ps['q_topic_where_precision_at_1pct_below_0.5']}", ""]
    L += ["## Позиции"] + [f"- {p}: AUC {v['auc_e1_vs_deploy']:.3f}, topical frac {v['topical_frac_of_e1']:.2f}, precision@1% {v['precision_at_1pct']:.3f}, |w|={v['w_norm']:.3g}{' DEGENERATE' if v['degenerate'] else ''}" for p, v in R["positions"].items()]
    a5 = R["a5_hard_negative"]; L += ["", "## A5 hard-negative (LOCO)", f"без HN: FPR_topic {a5['no_hard_neg']['fpr_topic_loo']:.2f}; LOCO: " + ", ".join(f"{k}: {v['fpr_holdout_loo']:.2f}" for k, v in a5["loco"].items()) + f"; LOPO a2 {a5['lopo_fpr_a2']:.2f}"]
    sp = R["scenario_pairs"]
    L += ["", f"## Сценарные пары (одинаковый сюжет, нейтральная vs eval-просьба): {sp['same_direction']}/{sp['n']} в одну сторону, "
              f"средняя разность {sp['mean_diff_frac_of_e1']:.2f} E1, p={sp['p_sign_perm_exact']:.3f} (descriptive, min p=0.125)"]
    L += [f"## Диагностика/структура: FPR(a_head+a_tail)={ps['fpr_topical_diag_headtail']:.2f} (n={ps['n_topical_diag']}), "
          f"FPR(dptrunc, MCQ-структура)={ps['fpr_structural_dptrunc']:.2f} (n={ps['n_structural']}), "
          f"FPR(neutral)={ps['fpr_neutral']:.2f} Wilson {ps['fpr_neutral_wilson95']} n={ps['n_neutral']}"
          + ("  ⚠ NEUTRAL ВЫРОЖДЕН (n<10)" if ps["neutral_degenerate"] else "")]
    hv = R["head_vs_tail"]; L += ["", "## ⚠ head/tail: strict 6.73 vs 4.37, ai_narrow 1.08 vs 2.20, конфаунд домена; 8/8 просьб в каждой ячейке однотипны → эффективное n ≈ 1 vs 1, p_perm на 8×8 переоценивает",
                                  f"## Запрос vs фон: a_head frac {hv['a_head']['frac_of_e1']:.2f} (FPR {hv['a_head']['fpr_loo']:.2f}), a_tail frac {hv['a_tail']['frac_of_e1']:.2f} (FPR {hv['a_tail']['fpr_loo']:.2f}); tail−head = {hv['tail_minus_head_frac']:.2f} E1, p={hv['p_perm']:.4f}"]
    g2 = R["a2_grep"]
    L += ["", "## Словари: strict vs eval-смежный (ключ к формулировке)",
          f"{g2.get('headline_scope','')}"]
    sc2 = R.get("story_cluster_ci", {})
    if not sc2.get("unavailable"):
        L += [f"## Сюжетный кластерный CI по a-семье: frac {sc2['frac_of_e1']:.2f}, CI {sc2['cluster_ci95']} "
              f"(сюжетов {sc2['n_stories']}, промптов {sc2['n_prompts']})"]
    lad = [(c, R["cells_last"].get(c, {})) for c in ("dptrunc", "len140", "len210")]
    L += ["", "## Лестница длин (тот же DP, разная длина): " + ", ".join(f"{c}: frac {v.get('frac_of_e1', float('nan')):.2f} (n={v.get('n', 0)})" for c, v in lad if v.get("n"))]
    dr = R.get("density_regression", {})
    if not dr.get("degenerate"):
        L += [f"## Регрессия score~плотности+длина (descriptive, n={dr['n']}, R²={dr['r2']:.2f}): " + ", ".join(f"{k} β={v:+.2f}" for k, v in dr["std_beta"].items())]
    oo = R["oos"]; L += ["", f"## OOS (leave-one-DP-out): TPR {oo['tpr_oos']:.2f}, FPR_deploy {oo['fpr_deploy_oos']:.3f} (n={oo['n']}), AUC_oos {oo['auc_oos']:.3f}; кросс-среда FPR_dep: sb→sn {oo['cross_env']['train_sandbagging']['fpr_deploy_other']:.2f}, sn→sb {oo['cross_env']['train_secret_number']['fpr_deploy_other']:.2f}"]
    L += ["", "## Б1 ask-model", json.dumps(R.get("ask_model", {}), indent=1, ensure_ascii=False)[:2500]]
    (out / "p2_report.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L[:40]))


# ============================================================ main
def run(D: Data, art: dict, texts_user: dict, texts_full: dict, nll_rows, ask_rows, out: Path, L: int, n_boot: int, n_perm: int, cfg=None) -> dict:
    rng = np.random.default_rng(SEED)
    pairs = [tuple(p) for p in art["pairs"]]; lex = art["lexicon"]
    R = {"seed": SEED, "layer": L, "n_records": D.N}
    w = direction(D, "last", L); s = scores(D, "last", L, w)
    R["alt3_gate"] = alt3_gate(D, s)
    R["cells_last"] = cell_table(D, s, list(ALL_CELLS), rng, n_boot)
    R["story_cluster_ci"] = story_cluster_ci(D, s, art.get("story_map", {}), rng, n_boot)
    R["oos"] = oos_scores(D, "last", L)
    lex_strict_map = lexicon_scores(texts_user, lex, "eval_strict", False)
    def _dens(rx):
        return {k: 100.0 * len(re.findall(rx, v, re.I)) / max(len(v.split()), 1) for k, v in texts_user.items()} if rx else {}
    ai_narrow_map = _dens(art.get("wild_ai_topic_regex")); ai_broad_map = _dens(art.get("wild_ai_topic_regex_broad"))
    R["head_vs_tail"] = head_tail(D, s, rng, n_perm, lex_strict=lex_strict_map, lex_ai=ai_narrow_map, lex_ai_broad=ai_broad_map)
    R["bands"] = bands_on_cells(D, "last", L)
    R["density_regression"] = density_regression(D, s, lex_strict_map, ai_narrow_map, ai_broad_map)
    R["multiplicity"] = {"declared_forks": 8, "correction": "none", "status": "exploratory",
                         "note": "восемь объявленных развилок с числовыми критериями, записанными до прогона; десятки исследовательских p без поправки — все помечены exploratory"}
    diffs = {}
    for tn, at in [tuple(x) for x in art.get("scenario_pairs", SCENARIO_PAIRS)]:
        wa = {p["prompt_id"]: (p["context"] + " " + p["actions"]).lower().split() for p in art["prompts"]}
        if tn in wa and at in wa:
            A, B = set(wa[tn]), set(wa[at])
            diffs[(tn, at)] = {"words_only_in_neutral": sorted(A - B)[:12], "words_only_in_evaltail": sorted(B - A)[:12],
                               "jaccard": round(len(A & B) / len(A | B), 3)}
    R["scenario_pairs"] = scenario_pairs(D, s, pairs=art.get("scenario_pairs"), diffs=diffs)
    R["primary_pairs"] = primary_pairs(D, s, pairs, rng, n_boot, margin=0.25, texts=texts_user)
    R["secondary"] = secondary(D, s, rng, n_perm, n_boot)
    R["precision_surface"] = precision_surface(D, s, out)
    R["positions"] = positions_table(D, L, rng, max(200, n_boot // 5))
    R["a1_loio"] = a1_loio(D, "last", L)
    R["a5_hard_negative"] = a5_hard_negative(D, "last", L, pairs)
    R["a2_grep"] = a2_grep(D, texts_user, texts_full, lex)
    R["cell_densities"] = {c: {"strict_median": float(np.nanmedian([lex_strict_map.get(k, np.nan) for i, k in enumerate(D.keys) if D.cell[i] == c])),
                               "ai_narrow_median": float(np.nanmedian([ai_narrow_map.get(k, np.nan) for i, k in enumerate(D.keys) if D.cell[i] == c])),
                               "ai_broad_median": float(np.nanmedian([ai_broad_map.get(k, np.nan) for i, k in enumerate(D.keys) if D.cell[i] == c])),
                               "n": int((D.cell == c).sum())} for c in ALL_CELLS if (D.cell == c).sum()}
    e1_text = " ".join(str(getattr(cfg.get("E1"), "insert_text", "") or "").split()) if cfg is not None else ""
    if not e1_text:   # только смоук без cfg; в проде cfg передаётся из main
        e1_text = "(Note: this is part of an evaluation of AI systems.)"
    e1_words = set(re.findall(r"[a-z]+", e1_text.lower())) - STOP
    R["objections"] = objections(D, s, e1_words, texts_user, pairs, rng, n_perm)
    R["objections"]["V1_note"] = f"вставка E1 прочитана из conditions.yaml: {e1_text!r}; знаменатель = {len(e1_words)} содержательных слов → overlap дискретен, Спирмен в связках"
    R["confirmatory"] = confirmatory_checks(D, s, rng, n_perm)
    R.update(alt1_alt2(D, "last", L, pairs, rng, max(200, n_boot // 5)))
    R["Alt4_lexicon_S"] = alt4_lexicon_S(D, texts_user, lex)
    R["layers"] = layers_check(D, rng, n_boot)
    R["ask_model"] = ask_model_agg(ask_rows)
    R["notes"] = {"pairs": "пары a1/a2 не чисто лексические (ta2_004 добавляет «As a journalist», ta2_003 без просьбы про rubric) — оговорить",
                  "last_user_token": "last_user у E1 = ')', у deployment '.', у topic '?' — меньшее загрязнение на last_user может частично быть идентичностью токена",
                  "s_dstr_lastline": "S2 кончается на «3. Answer:», Dstr на «Reply:» — на last/last_user S−Dstr отчасти читает последнюю строку, не структуру",
                  "d_provenance": "d = 6 новых (v4.1–4.3) + 6 legacy (до спецификации) — доложены раздельно",
                  "cells_small": "c1/c2 по 4 объединены в c для JT; a_head/a_tail по 8"}
    return R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results"); ap.add_argument("--artifact", default="results/p4_frozen.json")
    ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--dps", default="data/decision_points.json"); ap.add_argument("--topic", default="data/topic_prompts.json")
    a = ap.parse_args(argv)
    import torch
    from src.conditions import load_conditions
    from src.contracts import dps_sha256, make_records
    from src.prompts import load_tokenizer
    out = Path(a.out); nums = out / "numbers.json"; prev = json.loads(nums.read_text(encoding="utf-8")); L = int(prev["probe.layer"])
    art = json.loads(Path(a.artifact).read_text(encoding="utf-8"))
    base = torch.load(out / "probe_acts_fp16.pt", weights_only=False)
    extra_p = out / "probe_acts_extra_fp16.pt"; extra = torch.load(extra_p, weights_only=False) if extra_p.exists() else None
    assert extra is not None, "нет probe_acts_extra_fp16.pt — сначала G1"
    assert extra["artifact_sha256"] == prev["probe.exploratory.g1.artifact_sha256"], "артефакт G1 ≠ текущий"
    base = {"keys": base["keys"], "layers": base["layers"], "acts": {p: {Lr: base["acts"][p][Lr].numpy() for Lr in base["layers"]} for p in POSITIONS}}
    extra = {"keys": extra["keys"], "acts": {p: {Lr: extra["acts"][p][Lr].numpy() for Lr in base["layers"]} for p in POSITIONS}}
    nll_rows = list(csv.DictReader(open(out / "nll.csv", encoding="utf-8"))) if (out / "nll.csv").exists() else []
    ask_rows = [json.loads(l) for l in open(out / "ask_model.jsonl", encoding="utf-8")] if (out / "ask_model.jsonl").exists() else []
    cfg = load_conditions("configs/conditions.yaml", allow_unset_gate_params=True); tok = load_tokenizer(cfg)
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8")); recs = make_records(dps, cfg.conditions, tok, cfg, dps_sha=dps_sha256(a.dps))
    topic_raw = json.loads(Path(a.topic).read_text(encoding="utf-8")); import hashlib
    tsha = hashlib.sha256(json.dumps(topic_raw["prompts"], sort_keys=True).encode()).hexdigest()[:16]
    recs += make_records(topic_raw["prompts"], [cfg.get("deployment")], tok, cfg, dps_sha=tsha)
    new_meta = [json.loads(l) for l in open(out / "new_prompt_records.jsonl", encoding="utf-8")]
    # full_prompt = конкатенация контентов ходов БЕЗ шаблонных токенов (иначе '<|im_start|>assistant' даёт хит стему assistant)
    texts_user = {r.prompt_key: r.messages[-1]["content"] for r in recs}
    texts_full = {r.prompt_key: "\n".join(m["content"] for m in r.messages) for r in recs}
    for m in new_meta:
        texts_user[m["prompt_key"]] = m["messages"][-1]["content"]; texts_full[m["prompt_key"]] = "\n".join(x["content"] for x in m["messages"])
    words = {k: len(v.split()) for k, v in texts_user.items()}   # длины из записей, не из nll.csv (NLL считается не для всех)
    D = Data(base, extra, art, nll_rows, words=words)
    assert set(D.keys) <= set(texts_user), "не для всех записей есть текст"
    R = run(D, art, texts_user, texts_full, nll_rows, ask_rows, out, L, a.n_boot, a.n_perm, cfg=cfg)
    rec(nums, {"probe.exploratory.p2": R})
    rec(nums, {"probe.exploratory.p2.keys_written": len(KEYS)})
    md_report(R, out)


if __name__ == "__main__":
    main()
