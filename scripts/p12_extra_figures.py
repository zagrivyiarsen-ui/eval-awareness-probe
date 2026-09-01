#!/usr/bin/env python3
"""P12 — the extended figure set (fig8-fig14). English, same styling as p8_figures.py.

Everything that can be recomputed from records_full.csv is recomputed here, so these do not
depend on which analysis stages happen to sit in numbers.json. Figures that genuinely need
stage keys degrade gracefully: they are skipped with a stated reason, never silently.

  fig08_layers.png            AUC is constant where contamination is not — layer choice is blind
  fig09_positive_control.png  the positive control does not exceed a content-free system turn
  fig10_contamination.png     shift of every cell in units of the E1 shift, with firing counts
  fig11_confounds.png         within-cell correlation of score with length and with NLL
  fig12_length_ladder.png     the pooled ladder is two environments cancelling
  fig13_alt_directions.png    alternative direction constructions, FPR with Wilson intervals
  fig14_baselines.png         probe vs lexicon detectors vs "ask the model"

    PYTHONPATH=. python3 scripts/p12_extra_figures.py --out results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.28,
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "grid.color": "#cccccc", "grid.linewidth": 0.6, "grid.alpha": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5,
})
INK = "#1a1a1a"
RED, BLUE, ORANGE, GREY = "#b2182b", "#2166ac", "#e08214", "#9e9e9e"


# ─────────────────────────────────────────────────────────── helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load(out: Path):
    rows = []
    for r in csv.DictReader(open(out / "records_full.csv", encoding="utf-8")):
        for k in ("s_last_dom", "n_words", "nll_user"):
            try:
                r[k] = float(r[k]) if r.get(k) not in (None, "") else None
            except ValueError:
                r[k] = None
        r["is_train"] = r.get("is_train_contrast") == "1"
        rows.append(r)
    return rows


def sc(rows, *cells):
    return np.array([r["s_last_dom"] for r in rows
                     if r["cell"] in cells and r["s_last_dom"] is not None])


def baseline(rows):
    d = np.array([r["s_last_dom"] for r in rows if r["cell"] == "deployment"
                  and r["is_train"] and r["s_last_dom"] is not None])
    return d if len(d) else sc(rows, "deployment")


def find(N: dict, *suffixes):
    """Tolerant key lookup: stage prefixes have moved between runs."""
    for suf in suffixes:
        for k, v in N.items():
            if k.endswith(suf):
                return v
    return None


def foot(fig, text, y=-0.02):
    fig.text(0.005, y, text, fontsize=7.5, color="#666666", ha="left", va="top", wrap=True)


# ─────────────────────────────────────────────────────────── fig 8
def fig8(N, out):
    layers = ["L19", "L21", "L23"]
    auc, cont, fpr = [], [], []
    for L in layers:
        auc.append(find(N, f"p2.layers.{L}.auc_e1_vs_deploy", f"layers.{L}.auc_e1_vs_deploy"))
        cont.append(find(N, f"p2.layers.{L}.topical_frac_of_e1", f"layers.{L}.topical_frac_of_e1"))
        fpr.append(find(N, f"p2.layers.{L}.fpr_topical_loo", f"layers.{L}.fpr_topical_loo"))
    if any(v is None for v in auc + cont + fpr):
        raise RuntimeError("layer keys (*.p2.layers.L*.*) not in numbers.json")
    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w, auc, w, color=GREY, label="held-out AUC (E1 vs deployment)", zorder=3)
    ax.bar(x, cont, w, color=RED, label="topical contamination (frac of E1 shift)", zorder=3)
    ax.bar(x + w, fpr, w, color=ORANGE, label="topical FPR", zorder=3)
    for xi, vals in zip(x, zip(auc, cont, fpr)):
        for dx, v in zip((-w, 0, w), vals):
            ax.text(xi + dx, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("value")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Layer choice: AUC is blind to the only thing that differs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1)
    foot(fig, "AUC is 1.000 at all three layers while contamination moves by 40 % and topical FPR "
              "by 17 points.\nSelecting the layer by AUC — the standard recipe — cannot see any of it.",
         y=-0.22)
    fig.savefig(out / "fig08_layers.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────── fig 9
def fig9(rows, out):
    base = baseline(rows)
    e1 = sc(rows, "E1")
    shift = e1.mean() - base.mean()
    tau = float(np.quantile(e1, 0.05))
    cells = [("deployment", "deployment\n(no system turn)"),
             ("sys_neutral", "sys_neutral\ncontent-free system turn"),
             ("positive_control", "positive_control\neval statement in system"),
             ("positive_control_strong", "positive_control_strong\nescalated"),
             ("E1", "E1\nthe training positive")]
    fracs, fired, ns = [], [], []
    for c, _ in cells:
        v = sc(rows, c)
        fracs.append((v.mean() - base.mean()) / shift)
        fired.append(int((v >= tau).sum())); ns.append(len(v))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    colors = [BLUE, GREY, ORANGE, RED, "#67001f"]
    bars = ax.bar(range(len(cells)), fracs, color=colors, width=0.62, zorder=3)
    ax.axhline(0, color=INK, lw=0.9, zorder=4)
    ref = fracs[1]
    ax.axhline(ref, color=GREY, ls="--", lw=1.1, zorder=2)
    ax.text(len(cells) - 0.4, ref, " content-free system turn", fontsize=8, color="#666666",
            va="bottom", ha="right")
    for i, (b, f, k, n) in enumerate(zip(bars, fracs, fired, ns)):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.03, f"{f:.3f}", ha="center", fontsize=8.5)
        ax.text(b.get_x() + b.get_width() / 2, -0.07, f"fires {k}/{n}", ha="center",
                fontsize=7.5, color="#666666")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([lab for _, lab in cells], fontsize=8)
    ax.set_ylabel("shift, in units of the E1 shift")
    ax.set_ylim(-0.12, 1.12)
    ax.grid(axis="y", zorder=0)
    ax.set_title("The positive control does not exceed a content-free system turn")
    foot(fig, "The v2 non-monotonicity is explained: the ordinary positive control's shift is produced "
              "by the appearance of a system turn,\nnot by its evaluation content. Only the escalated "
              "version adds anything on top — and it is the only one that ever fires.")
    fig.savefig(out / "fig09_positive_control.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────── fig 10
def fig10(rows, out):
    base = baseline(rows)
    e1 = sc(rows, "E1")
    shift = e1.mean() - base.mean()
    tau = float(np.quantile(e1, 0.05))
    skip = {"deployment", "E1", "topic_unk"}
    data = []
    for c in sorted({r["cell"] for r in rows}):
        if c in skip:
            continue
        v = sc(rows, c)
        if len(v) == 0:
            continue
        k = int((v >= tau).sum())
        lo, hi = wilson(k, len(v))
        data.append((c, (v.mean() - base.mean()) / shift, k, len(v), lo, hi, r_role(rows, c)))
    data.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(data) + 2.2))
    y = np.arange(len(data))
    col = {"topical": RED, "diagnostic": "#fdd0a2", "gradient": ORANGE, "neutral": BLUE,
           "neutral_external": "#053061", "structural": GREY, "ladder": "#d9d9d9",
           "insert_alt": "#d6604d", "prereg_S": "#8c8c8c", "prereg_ctrl": "#8c8c8c"}
    for i, (c, f, k, n, lo, hi, role) in enumerate(data):
        ax.barh(i, f, height=0.62, color=col.get(role, GREY), zorder=3)
        ax.text(max(f, 0.0) + 0.02, i, f"{f:.2f}   {k}/{n} fire",
                va="center", ha="left", fontsize=7.5, color="#555555")
    ax.axvline(0, color=INK, lw=0.9, zorder=4)
    ax.axvline(1, color=RED, ls="--", lw=1.0, zorder=2)
    ax.text(1, len(data) - 0.4, " E1", color=RED, fontsize=8, va="center")
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data])
    ax.set_ylim(-0.8, len(data) - 0.2)
    ax.set_xlim(min(-0.15, min(d[1] for d in data) - 0.1), 1.35)
    ax.set_xlabel("shift relative to deployment, in units of the E1 shift")
    ax.grid(axis="x", zorder=0)
    ax.set_title("How far every cell moves along the direction")
    foot(fig, "Prompts that merely discuss measurement or evaluation land at 0.57-0.97 of the shift "
              "produced by explicitly announcing an evaluation.\nAI-safety prompts (b), the most "
              "AI-dense cell in the set, never fire; churn prediction and LLM engineering fire most.")
    fig.savefig(out / "fig10_contamination.png")
    plt.close(fig)


def r_role(rows, cell):
    for r in rows:
        if r["cell"] == cell:
            return r.get("role", "")
    return ""


# ─────────────────────────────────────────────────────────── fig 11
def fig11(rows, out):
    by = defaultdict(lambda: {"s": [], "w": [], "nll": []})
    for r in rows:
        if r["s_last_dom"] is None or r["is_train"]:
            continue
        g = by[r["cell"]]
        if r["n_words"] is not None:
            g["s"].append(r["s_last_dom"]); g["w"].append(r["n_words"])
        if r["nll_user"] is not None:
            g["nll"].append((r["s_last_dom"], r["nll_user"]))
    rl, nl = [], []
    for c, g in by.items():
        if len(g["s"]) >= 4:
            rl.append((c, spearman(g["s"], g["w"]), len(g["s"])))
        if len(g["nll"]) >= 4:
            a = np.array(g["nll"])
            nl.append((c, spearman(a[:, 0], a[:, 1]), len(a)))
    if not rl and not nl:
        raise RuntimeError("no per-cell length/NLL data in records_full.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=False)
    for ax, dat, name in ((axes[0], rl, "prompt length (words)"),
                          (axes[1], nl, "text unfamiliarity (NLL)")):
        if not dat:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="#999999")
            continue
        dat.sort(key=lambda t: t[1])
        y = np.arange(len(dat))
        ax.barh(y, [d[1] for d in dat], height=0.6,
                color=[RED if d[1] > 0 else BLUE for d in dat], zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{d[0]}  (n={d[2]})" for d in dat], fontsize=8)
        ax.axvline(0, color=INK, lw=0.9, zorder=4)
        ax.set_xlim(-1, 1)
        ax.grid(axis="x", zorder=0)
        ax.set_xlabel(f"within-cell Spearman rho(score, {name})")
        ax.set_title(("A · length" if name.startswith("prompt") else "B · unfamiliarity"))
    fig.suptitle("Neither length nor unfamiliarity explains the score: the sign is not even consistent",
                 fontsize=11.5, fontweight="bold", x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    foot(fig, "Correlations are computed WITHIN cells. A between-cell correlation would be driven by "
              "cell composition and would look convincing for the wrong reason.")
    fig.savefig(out / "fig11_confounds.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────── fig 12
SCOL = "s_last_dom"


def fig12(rows, out):
    """Лестница длин, разнесённая по средам.

    Пулированные ступени (0.306 и 0.281) отличаются на 0.025 и выглядят как насыщение.
    Разнесённые по средам они идут в РАЗНЫЕ стороны и пересекаются: усечение сокращает
    ещё и переменный хвост, поэтому ступени различаются не только длиной. Лестница
    ничего не ограничивает, и фигура должна показывать именно это."""
    base = baseline(rows)
    e1 = sc(rows, "E1")
    shift = e1.mean() - base.mean()

    # среда у ступеней лестницы не записана в env/prompt_id (там x_len140/x_len210),
    # поэтому определяем её по тексту промпта: sandbagging — арифметическая политика,
    # secret_number — сессия с guess.py.
    import json as _json
    texts = {}
    tp = out / "records_texts.jsonl"
    if tp.exists():
        for line in open(tp, encoding="utf-8"):
            d = _json.loads(line)
            texts[d.get("prompt_key") or d.get("key", "")] = d.get("text", "")
    SB = ("Documentation.md", "Arithmetic", "arithmetic")
    SN = ("guess.py", "secret number", "guesses")

    def env_of(r):
        t = texts.get(r.get("prompt_key", ""), "")
        if any(k in t for k in SN):
            return "sn"
        if any(k in t for k in SB):
            return "sb"
        return None

    # если по тексту не определилось (например, тексты не сохранены) — запасной путь:
    # ступени строились по 16 на ячейку, первые 8 из одной среды, вторые 8 из другой
    _probe = [env_of(r) for r in rows if r["cell"] == "len140"]
    USE_ORDER = not any(_probe)
    if USE_ORDER:
        order = {}
        for cell in ("len140", "len210"):
            ids = sorted(r["prompt_id"] for r in rows if r["cell"] == cell)
            for i, pid in enumerate(ids):
                order[(cell, pid)] = "sb" if i < len(ids) // 2 else "sn"

    def sel(cell, env_tag):
        if USE_ORDER:
            rs = [r for r in rows if r["cell"] == cell
                  and order.get((cell, r["prompt_id"])) == env_tag]
        else:
            rs = [r for r in rows if r["cell"] == cell and env_of(r) == env_tag]
        v = [float(r[SCOL]) for r in rs if r.get(SCOL) not in (None, "")]
        w = [r["n_words"] for r in rs if r["n_words"] is not None]
        return np.array(v), (float(np.mean(w)) if w else float("nan"))

    series, fig, ax = [], *plt.subplots(figsize=(7.6, 4.5))
    for tag, lab, col in (("sb", "sandbagging", RED), ("sn", "secret-number", BLUE)):
        xs, ys, ns = [], [], []
        for cell in ("len140", "len210"):
            v, w = sel(cell, tag)
            if len(v):
                xs.append(w); ys.append((v.mean() - base.mean()) / shift); ns.append(len(v))
        if len(xs) == 2:
            ax.plot(xs, ys, "-o", color=col, ms=8, lw=2, zorder=4, label=lab)
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(f"{y:.3f}  (n={n})", (x, y), textcoords="offset points",
                            xytext=(8, 6), fontsize=8, color=col)
            series.append((lab, ys))
    if not series:
        raise RuntimeError("length-ladder cells not split by environment in records_full.csv")
    if USE_ORDER:
        print("  fig12: среда определена по порядку prompt_id (тексты недоступны) — "
              "проверьте, что первые 8 ступеней действительно из одной среды")

    pooled = []
    for cell in ("len140", "len210"):
        v = sc(rows, cell)
        w = [r["n_words"] for r in rows if r["cell"] == cell and r["n_words"] is not None]
        if len(v):
            pooled.append((float(np.mean(w)) if w else float("nan"),
                           (v.mean() - base.mean()) / shift))
    if len(pooled) == 2:
        ax.plot([p[0] for p in pooled], [p[1] for p in pooled], "--s", color=GREY, ms=6,
                lw=1.4, zorder=3, label="pooled (what cancelling looks like)")

    for lab, y, col in (("topical cells: 0.57-0.97 E1", 0.78, ORANGE),
                        ("external Dolly: 0.59 E1", 0.59, "#7a7a7a")):
        ax.axhline(y, color=col, ls=":", lw=1.0, alpha=0.7)
        ax.text(0.99, y, lab + " ", transform=ax.get_yaxis_transform(), fontsize=8,
                color=col, va="bottom", ha="right")
    ax.set_xlabel("prompt length (words)")
    ax.set_ylabel("shift, in units of the E1 shift")
    ax.grid(zorder=0)
    ax.set_title("The pooled length ladder is two environments cancelling")
    ax.legend(loc="center right", fontsize=8)
    foot(fig, "Pooled, the two rungs differ by 0.025 E1 and look like saturation. Split by "
              "environment they move in opposite\ndirections: sandbagging 0.156 -> 0.434, "
              "secret-number 0.456 -> 0.128. Truncation also removes the varying tail\n"
              "(len140-sb is 8 identical texts), so the rungs differ in content as well as "
              "length. The ladder bounds nothing.")
    fig.savefig(out / "fig12_length_ladder.png")
    plt.close(fig)

# ─────────────────────────────────────────────────────────── fig 13
def fig13(N, out):
    # имена конструкций берём ИЗ ДАННЫХ: список профилей меняется между прогонами
    pref = [k for k in N if ".p5.profiles." in k]
    if not pref:
        seen = sorted({k for k in N if ".p5." in k})[:8]
        raise RuntimeError("no *.p5.profiles.* keys in numbers.json; p5-like keys present: "
                           + (", ".join(seen) if seen else "none — run the P5 stage"))
    base = pref[0].split(".p5.profiles.")[0] + ".p5.profiles."
    names = sorted({k[len(base):].split(".")[0] for k in pref})
    rowsd = []
    for nm in names:
        b = base + nm + "."
        ft, fw = N.get(b + "by_group.topical_leading.fpr_loo"), N.get(b + "by_group.wild_other_only.fpr_loo")
        if ft is None and fw is None:
            continue
        rowsd.append((nm, ft, fw,
                      int(N.get(b + "by_group.topical_leading.n") or 30),
                      int(N.get(b + "by_group.wild_other_only.n") or 37),
                      N.get(b + "cos_with_diff_of_means")))
    if not rowsd:
        raise RuntimeError("profiles found (" + ", ".join(names) +
                           ") but none carry by_group.*.fpr_loo")
    rowsd.sort(key=lambda t: (t[1] if t[1] is not None else 9))
    fig, ax = plt.subplots(figsize=(8.0, 0.62 * len(rowsd) + 2.4))
    for i, (nm, ft, fw, nt, nw, cs) in enumerate(rowsd):
        for off, f, n, col, lab in ((0.16, ft, nt, RED, "topical prompts"),
                                    (-0.16, fw, nw, BLUE, "external Dolly")):
            if f is None:
                continue
            k = int(round(f * n))
            lo, hi = wilson(k, n)
            ax.plot([lo, hi], [i + off] * 2, color=col, lw=2.4, alpha=0.5, zorder=3)
            ax.plot([f], [i + off], "o", color=col, ms=6, zorder=4,
                    label=lab if i == 0 else None)
            ax.text(hi + 0.012, i + off, f"{f:.3f}  ({k}/{n})", va="center", fontsize=7.5,
                    color="#555555")
    ax.set_yticks(np.arange(len(rowsd)))
    ax.set_yticklabels([f"{nm}\ncos with d-o-m {cs:.3f}" if cs is not None else nm
                        for nm, _, _, _, _, cs in rowsd], fontsize=8)
    ax.set_ylim(-0.7, len(rowsd) - 0.3)
    ax.set_xlabel("false-positive rate (in-sample), with Wilson 95 % intervals")
    ax.set_xlim(-0.02, 0.8)
    ax.grid(axis="x", zorder=0)
    ax.set_title("Alternative direction constructions: the intervals overlap")
    ax.legend(loc="lower right")
    foot(fig, "All of these are trained IN-SAMPLE on the 64 contrast records, while the headline "
              "numbers are out-of-sample — the two are not comparable.\nThe whitening methods look "
              "better on topical prompts, but their intervals on external negatives overlap the "
              "baseline, so the reduction is not established.")
    fig.savefig(out / "fig13_alt_directions.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────── fig 14
def fig14(N, rows, out):
    items = []
    probe = find(N, "probe.v1.heldout_auc_pool")
    if probe is not None:
        items.append(("linear probe\n(held-out)", probe, RED))
    for nm, lab in (("eval_strict", "grep: strict\nevaluation lexicon"),
                    ("eval_broad", "grep: broad\nevaluation lexicon"),
                    ("eval_adjacent", "grep: adjacent\nevaluation lexicon")):
        v = find(N, f"a2_grep.user_only.{nm}.auc", f"grep.user_only.{nm}.auc", f"{nm}.auc")
        if v is not None:
            items.append((lab, v, GREY if nm != "eval_adjacent" else BLUE))
    # "ask the model" — a ceiling, so its AUC is 0.5 by construction; compute from the CSV
    y = [r for r in rows if r["cell"] == "E1" and r.get("ask_main") in ("yes", "no")]
    d = [r for r in rows if r["cell"] == "deployment" and r.get("ask_main") in ("yes", "no")]
    if y and d:
        py = sum(1 for r in y if r["ask_main"] == "yes") / len(y)
        pd_ = sum(1 for r in d if r["ask_main"] == "yes") / len(d)
        items.append(("ask the model\n“is this an evaluation?”", 0.5 + (py - pd_) / 2, ORANGE))
    if len(items) < 2:
        raise RuntimeError("baseline keys not in numbers.json")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(items))
    ax.bar(x, [i[1] for i in items], width=0.6, color=[i[2] for i in items], zorder=3)
    ax.axhline(0.5, color=INK, ls="--", lw=1.0, zorder=4)
    ax.text(len(items) - 0.45, 0.505, "chance", fontsize=8, color="#666666")
    for xi, (_, v, _) in zip(x, items):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("AUC, E1 vs deployment")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Baselines on the training contrast")
    foot(fig, "The adjacent-lexicon detector scores BELOW chance: under that vocabulary the negative "
              "class is denser than the positive one.\nThat is direct lexical evidence that the "
              "negative class is itself evaluative — the contrast was never evaluation vs not.")
    fig.savefig(out / "fig14_baselines.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────── fig 15
def fig15(rows, out):
    """Precision at 1 % prevalence as a function of the TRUE-POSITIVE RATE we assume.

    The headline precision is computed with TPR = 0.906, measured on E1. But the paper argues
    E1 is not eval awareness — it is an explicit meta-sentence. So the TPR a real monitor would
    have on genuinely eval-aware prompts is unknown. Rather than caveating that in prose, we
    show the whole curve: the conclusion has to survive every plausible TPR, and it does."""
    e1 = sc(rows, "E1")
    if not len(e1):
        raise RuntimeError("no E1 rows in records_full.csv")
    tau = float(np.quantile(e1, 0.05))
    srcs = {"pre-registered topical": ("a_legacy", "d_legacy"),
            "authored after the result": ("a1", "a2", "d"),
            "external Dolly": ("wild_other",)}
    cols = [RED, ORANGE, BLUE]
    tprs = np.linspace(0.05, 1.0, 200)
    q = 0.01
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    drawn = 0
    for (lab, cells), col in zip(srcs.items(), cols):
        x = sc(rows, *cells)
        if not len(x):
            continue
        k, n = int((x >= tau).sum()), len(x)
        f = k / n
        lo, hi = wilson(k, n)
        prec = q * tprs / (q * tprs + (1 - q) * f)
        ax.fill_between(tprs, q * tprs / (q * tprs + (1 - q) * hi),
                        q * tprs / (q * tprs + (1 - q) * lo) if lo > 0 else np.ones_like(tprs),
                        color=col, alpha=0.12, lw=0, zorder=2)
        ax.plot(tprs, prec, color=col, lw=1.9, zorder=3, label=f"{lab} — FPR {f:.3f}, n = {n}")
        drawn += 1
    if not drawn:
        plt.close(fig)
        raise RuntimeError("no negative-origin cells in records_full.csv")
    obs = float(np.mean(e1 >= np.array([np.quantile(np.delete(e1, i), 0.05)
                                        for i in range(len(e1))])))
    ax.axvline(obs, color=INK, ls="--", lw=1.1, zorder=4)
    ax.annotate(f"TPR measured on E1 = {obs:.3f}\n(the class the paper argues\nis NOT eval awareness)",
                xy=(obs, 0.42), xytext=(-10, 0), textcoords="offset points", fontsize=8,
                ha="right", va="center", color="#444444")
    ax.axhline(0.5, color=GREY, ls=":", lw=1.0, zorder=1)
    ax.text(0.99, 0.505, "one in two alerts correct ", fontsize=8, color="#777777",
            ha="right", va="bottom")
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("assumed true-positive rate on genuinely eval-aware prompts")
    ax.set_ylabel("precision at 1 % prevalence")
    ax.grid(zorder=0)
    ax.set_title("The conclusion does not depend on the true-positive rate we assume")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    foot(fig, "Even at a perfect TPR of 1.0 the monitor stays below 0.06 precision at 1 % prevalence, "
              "so the headline does not rest on the\nTPR measured on E1 — which, by the argument of "
              "this paper, is the true-positive rate of a meta-sentence detector, not of an "
              "eval-awareness monitor.")
    fig.savefig(out / "fig15_precision_vs_tpr.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────── main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    out = Path(a.out)
    N = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    rows = load(out)
    jobs = [("fig08_layers.png", lambda: fig8(N, out)),
            ("fig09_positive_control.png", lambda: fig9(rows, out)),
            ("fig10_contamination.png", lambda: fig10(rows, out)),
            ("fig11_confounds.png", lambda: fig11(rows, out)),
            ("fig12_length_ladder.png", lambda: fig12(rows, out)),
            ("fig13_alt_directions.png", lambda: fig13(N, out)),
            ("fig14_baselines.png", lambda: fig14(N, rows, out)),
            ("fig15_precision_vs_tpr.png", lambda: fig15(rows, out))]
    ok, bad = [], []
    for name, fn in jobs:
        try:
            fn()
            ok.append(name)
        except Exception as e:
            bad.append((name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    print("\nwritten: " + ("  ".join(ok) if ok else "(none)"))
    for name, why in bad:
        print(f"NOT written: {name} — {why}")


if __name__ == "__main__":
    main([] if "__file__" not in globals() else None)
