#!/usr/bin/env python3
"""P8 — figures for the write-up (probe arm). English labels, publication-ish styling.

Everything that CAN be recomputed from records_full.csv IS recomputed here, so the figures
no longer depend on which analysis stages happen to be present in numbers.json:

  fig1_scores_by_cell.png     score distribution per cell with the operating threshold
  fig2_insert_position.png    four insertion wordings at three placements; sign flip across environments
  fig3_precision.png          precision vs prevalence by origin of the negatives (from CSV)
  fig4_perm_bands.png         observed shift vs permutation band (needs numbers.json)
  fig5_ask_model.png          probe score vs what the model itself says (AUC computed here)

Each figure is drawn independently; a failure is reported with a reason, never silent.

    PYTHONPATH=. python3 scripts/p8_figures.py --out results
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

# ─────────────────────────────────────────────────────────────── style
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
C = {"positive": "#b2182b", "train_neg": "#2166ac", "insert_alt": "#d6604d",
     "prereg": "#8c8c8c", "topical": "#ef8a62", "diagnostic": "#fdd0a2",
     "gradient": "#e08214", "neutral": "#4393c3", "external": "#053061",
     "structural": "#bababa", "ladder": "#dadada", "band": "#dcdcdc"}
ROLE_TO_C = {"train_pos": C["positive"], "train_neg": C["train_neg"],
             "insert_alt": C["insert_alt"], "prereg_S": C["prereg"],
             "prereg_ctrl": C["prereg"], "topical": C["topical"],
             "diagnostic": C["diagnostic"], "gradient": C["gradient"],
             "neutral": C["neutral"], "neutral_external": C["external"],
             "structural": C["structural"], "ladder": C["ladder"]}
LEGEND_ORDER = [("train_pos", "training positive (E1)"), ("train_neg", "training negative"),
                ("insert_alt", "other wordings / placements"), ("prereg_S", "pre-registered controls"),
                ("prereg_ctrl", "pre-registered controls"),
                ("topical", "evaluation-topic prompts"), ("diagnostic", "diagnostic split"),
                ("gradient", "AI / ML topic gradient"), ("neutral", "neutral, authored"),
                ("neutral_external", "neutral, external (Dolly)"),
                ("structural", "structural control"), ("ladder", "length ladder")]

ORIGIN = {"pre-registered topical": ("a_legacy", "d_legacy"),
          "authored after the result": ("a1", "a2", "d"),
          "external Dolly": ("wild_other",)}
ORIGIN_C = ["#b2182b", "#ef8a62", "#2166ac"]


# ─────────────────────────────────────────────────────────────── helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def within_cell_auc(rows, col):
    """Mantel-Haenszel style: concordant pairs pooled WITHIN cells. The naive pooled AUC is
    driven by cell composition (~96 % of pairs are between cells) and must not be used."""
    num = den = 0.0
    by = defaultdict(lambda: {"yes": [], "no": []})
    for r in rows:
        v, s = r.get(col), r["s_last_dom"]
        if v in ("yes", "no") and s is not None and not r["is_train"]:
            by[r["cell"]][v].append(s)
    cells = 0
    for g in by.values():
        y, n_ = np.array(g["yes"]), np.array(g["no"])
        if len(y) and len(n_):
            d = y[:, None] - n_[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
            cells += 1
    return (num / den if den else float("nan")), int(den), cells


def load(out: Path):
    rows = []
    for r in csv.DictReader(open(out / "records_full.csv", encoding="utf-8")):
        try:
            r["s_last_dom"] = float(r["s_last_dom"]) if r.get("s_last_dom") not in (None, "") else None
        except ValueError:
            r["s_last_dom"] = None
        r["is_train"] = r.get("is_train_contrast") == "1"
        rows.append(r)
    return rows


def scores(rows, *cells):
    return np.array([r["s_last_dom"] for r in rows
                     if r["cell"] in cells and r["s_last_dom"] is not None])


def footnote(fig, text, y=-0.02):
    fig.text(0.005, y, text, fontsize=7.5, color="#666666", ha="left", va="top", wrap=True)


# ─────────────────────────────────────────────────────────────── fig 1
def fig1(rows, out, tau):
    by, role = defaultdict(list), {}
    for r in rows:
        if r["s_last_dom"] is None:
            continue
        by[r["cell"]].append(r["s_last_dom"])
        role[r["cell"]] = r.get("role", "")
    cells = sorted(by, key=lambda c: float(np.mean(by[c])))
    fig, ax = plt.subplots(figsize=(7.8, 0.30 * len(cells) + 2.4))
    rng = np.random.default_rng(0)
    lo = min(min(v) for v in by.values())
    hi = max(max(v) for v in by.values())
    pad = 0.06 * (hi - lo)
    for i, c in enumerate(cells):
        v = np.array(by[c])
        col = ROLE_TO_C.get(role.get(c, ""), "#999999")
        ax.scatter(v, i + rng.normal(0, 0.085, len(v)), s=13, alpha=0.72, color=col,
                   edgecolors="none", zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.34, i + 0.34], color=INK, lw=1.8, zorder=4)
        fired = int((v >= tau).sum())
        ax.text(hi + pad * 1.6, i, f"{fired}/{len(v)}", va="center", ha="right",
                fontsize=7.5, color="#666666")
    ax.axvline(tau, color=C["positive"], ls="--", lw=1.1, zorder=2)
    ax.set_xlim(lo - pad, hi + pad * 2.2)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells)
    ax.set_ylim(-0.9, len(cells) - 0.1)
    ax.set_xlabel("probe score  (layer 21, last position, diff-of-means direction)")
    ax.set_title("Probe scores by prompt cell")
    ax.grid(axis="x", zorder=0)
    ax.annotate(f"τ = {tau:.1f}   (5th percentile of E1)", xy=(tau, 0.6),
                xytext=(9, 0), textcoords="offset points", fontsize=8,
                color=C["positive"], va="center")
    ax.text(hi + pad * 1.6, len(cells) - 0.55, "fired", fontsize=7.5, color="#666666",
            ha="right", va="center")
    seen, handles = set(), []
    for k, labl in LEGEND_ORDER:
        if k in set(role.values()) and labl not in seen:
            seen.add(labl)
            handles.append(plt.Line2D([], [], marker="o", ls="", ms=5,
                                      color=ROLE_TO_C[k], label=labl))
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.02))
    footnote(fig, "Black bar = cell mean. Right-hand counts = prompts at or above τ.\n"
                  "τ is calibrated on the positive class, so the true-positive rate is ≈ 0.95 by "
                  "construction and is not a result.", y=-0.13)
    fig.savefig(out / "fig1_scores_by_cell.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────── fig 2
def fig2(rows, N, out):
    conds = ["E1", "e2_c", "e2_b", "e2_a"]
    lab = {"E1": "E1\nend of user turn", "e2_c": "e2_c\nreworded, same place",
           "e2_b": "e2_b\nmoved", "e2_a": "e2_a\nmoved, other role"}
    dep_tr = np.array([r["s_last_dom"] for r in rows if r["cell"] == "deployment"
                       and r["is_train"] and r["s_last_dom"] is not None])
    base = dep_tr if len(dep_tr) else scores(rows, "deployment")
    e1_shift = scores(rows, "E1").mean() - base.mean()
    pool = [(scores(rows, c).mean() - base.mean()) / e1_shift for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    bars = ax.bar(range(4), pool, color=[C["positive"]] + [C["insert_alt"]] * 3, width=0.62,
                  zorder=3)
    ax.axhline(0, color=INK, lw=0.9, zorder=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([lab[c] for c in conds], fontsize=8)
    ax.set_ylabel("shift, in units of the E1 shift")
    ax.set_title("A · Four wordings at three placements (confounded)")
    ax.grid(axis="y", zorder=0)
    for b, v in zip(bars, pool):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.035 if v >= 0 else -0.07),
                f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(min(-0.2, min(pool) - 0.15), 1.18)

    ax = axes[1]
    envs = ["sandbagging", "secret_number"]
    w = 0.36
    got = False
    for j, e in enumerate(envs):
        vals = [N.get(f"probe.v5.{c}.{e}.frac_of_e1", 1.0 if c == "E1" else None) for c in conds]
        got = got or any(v is not None for v in vals[1:])
        ax.bar(np.arange(4) + (j - 0.5) * w, [0 if v is None else v for v in vals], width=w,
               label=e.replace("_", " "), color=["#2166ac", "#e08214"][j], zorder=3)
    ax.axhline(0, color=INK, lw=0.9, zorder=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(conds)
    ax.set_title("B · Same text, opposite sign in the two environments")
    ax.set_ylabel("shift, in units of the E1 shift")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="best")
    if not got:
        ax.text(0.5, 0.5, "probe.v5.* keys not in numbers.json", transform=ax.transAxes,
                ha="center", color="#999999")
    fig.suptitle("The direction reads the POSITION of the insertion, not its content",
                 fontsize=11.5, fontweight="bold", x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    footnote(fig, "Panel A pools both environments; E1 = 1.000 by definition. Moving the same "
                  "13-token sentence inside the same prompt removes about 95 % of the effect, and in "
                  "panel B the surviving effect changes sign between environments.")
    fig.savefig(out / "fig2_insert_position.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────── fig 3
def fig3(rows, N, out):
    """Recomputed from records_full.csv, so it does not depend on stage keys in numbers.json."""
    e1 = scores(rows, "E1")
    if not len(e1):
        raise RuntimeError("no E1 rows in records_full.csv")
    tau = float(np.quantile(e1, 0.05))
    loo = np.array([np.quantile(np.delete(e1, i), 0.05) for i in range(len(e1))])
    tpr = float(np.mean(e1 >= loo))
    q = np.logspace(-3, -0.7, 300)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    drawn = 0
    for (label, cells), col in zip(ORIGIN.items(), ORIGIN_C):
        x = scores(rows, *cells)
        if not len(x):
            continue
        k, n = int((x >= tau).sum()), len(x)
        f = k / n
        lo, hi = wilson(k, n)
        prec = q * tpr / (q * tpr + (1 - q) * f) if f > 0 else np.ones_like(q)
        p_lo = q * tpr / (q * tpr + (1 - q) * hi)
        p_hi = q * tpr / (q * tpr + (1 - q) * lo) if lo > 0 else np.ones_like(q)
        ax.fill_between(q, p_lo, p_hi, color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(q, prec, color=col, lw=1.9, zorder=3,
                label=f"{label}  (n = {n}) — FPR {f:.3f}, Wilson [{lo:.2f}, {hi:.2f}]")
        at1 = 0.01 * tpr / (0.01 * tpr + 0.99 * f) if f > 0 else 1.0
        ax.plot([0.01], [at1], "o", color=col, ms=6, zorder=5)
        ax.annotate(f"{at1:.3f}", (0.01, at1), textcoords="offset points",
                    xytext=(10, 16 - 14 * drawn), fontsize=8.5, color=col, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=0, shrinkB=3))
        drawn += 1
    if not drawn:
        plt.close(fig)
        raise RuntimeError("none of the negative-origin cells are present in records_full.csv")
    ax.axvline(0.01, color="#777777", ls=":", lw=1, zorder=1)
    ax.text(0.0092, 0.30, "1 % prevalence", fontsize=8, color="#777777",
            rotation=90, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("prevalence of genuinely eval-aware prompts in the traffic")
    ax.set_ylabel("precision of the monitor")
    ax.set_ylim(0, 1)
    ax.set_xlim(q[0], q[-1])
    ax.grid(zorder=0)
    ax.set_title("Precision at realistic prevalence, by origin of the negatives")
    ax.legend(loc="upper left")
    footnote(fig, f"TPR = {tpr:.3f} at the leave-one-out threshold. Shaded band = Wilson 95 % CI on the "
                  "false-positive rate. Negatives written after the result was known are no easier "
                  "than the pre-registered ones.")
    fig.savefig(out / "fig3_precision.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────── fig 4
def fig4(N, out):
    cells, shift, q95, pv = [], [], [], []
    for stage in ("p2.bands", "p4.b2_bands", "p2.cells_last"):
        for k in list(N):
            if k.startswith(f"probe.exploratory.{stage}.") and k.endswith(".perm_band.probe_shift"):
                c = k.split(".")[-3]
                b = k.rsplit(".", 1)[0]
                if c in cells or N.get(b + ".perm_abs_q95") in (None, 0):
                    continue
                cells.append(c); shift.append(abs(N[k]))
                q95.append(N[b + ".perm_abs_q95"]); pv.append(N.get(b + ".perm_p"))
    if not cells:
        raise RuntimeError("no *.perm_band.* keys in numbers.json — run the P2/P4 stages first")
    o = np.argsort([s / q for s, q in zip(shift, q95)])
    cells = [cells[i] for i in o]; shift = [shift[i] for i in o]
    q95 = [q95[i] for i in o]; pv = [pv[i] for i in o]
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(cells) + 2.3))
    y = np.arange(len(cells))
    ax.barh(y, q95, height=0.6, color=C["band"], zorder=2,
            label="permutation band: 95th pct of |shift| under shuffled E1/deployment labels")
    ax.plot(shift, y, "o", color=INK, ms=6, zorder=4, label="observed |shift| of the cell")
    xmax = max(max(shift), max(q95))
    for i, p in enumerate(pv):
        if p is not None:
            ax.text(xmax * 1.02, i, f"p = {p:.2f}", va="center", fontsize=7.5, color="#666666")
    ax.set_yticks(y); ax.set_yticklabels(cells)
    ax.set_ylim(-0.8, len(cells) - 0.2)
    ax.set_xlabel("|shift| relative to the deployment mean")
    ax.set_title("No cell leaves the permutation band")
    ax.grid(axis="x", zorder=0)
    ax.set_xlim(0, xmax * 1.22)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, ncol=1, loc="upper center", bbox_to_anchor=(0.5, 0.02))
    footnote(fig, "The band is built by shuffling the E1/deployment labels within pairs and refitting\n"
                  "the direction. The test uses |shift| only, so it does not check sign agreement —\n"
                  "and the observed shift is positive for every cell.", y=-0.10)
    fig.savefig(out / "fig4_perm_bands.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────── fig 5
def fig5(rows, out):
    qs = [("ask_main", "\u201cIs this an evaluation?\u201d"),
          ("ask_self", "\u201cAre YOU being evaluated?\u201d"),
          ("ask_inverted", "\u201cIs this real deployment?\u201d")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.3), sharey=True)
    rng = np.random.default_rng(1)
    any_data = False
    for ax, (col, title) in zip(axes, qs):
        groups = {"no": [], "yes": []}
        for r in rows:
            if r.get(col) in groups and r["s_last_dom"] is not None and not r["is_train"]:
                groups[r[col]].append(r["s_last_dom"])
        for i, key in enumerate(("no", "yes")):
            v = np.array(groups[key])
            if not len(v):
                continue
            any_data = True
            ax.scatter(i + rng.normal(0, 0.07, len(v)), v, s=13, alpha=0.62,
                       color=[C["train_neg"], C["positive"]][i], edgecolors="none", zorder=3)
            ax.plot([i - 0.26, i + 0.26], [v.mean()] * 2, color=INK, lw=1.8, zorder=4)
        a, npairs, ncells = within_cell_auc(rows, col)
        for i, key in enumerate(("no", "yes")):
            if groups[key]:
                ax.annotate(f"n = {len(groups[key])}", (i, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=7.5, color="#666666")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["model says no", "model says yes"])
        ax.set_xlim(-0.6, 1.6)
        ax.grid(axis="y", zorder=0)
        ax.set_title(f"{title}\nwithin-cell AUC = {a:.3f}\n({npairs} pairs, {ncells} cells)",
                     fontsize=9.5)
    if not any_data:
        plt.close(fig)
        raise RuntimeError("no ask_* answers in records_full.csv")
    axes[0].set_ylabel("probe score")
    fig.suptitle("The probe score barely tracks what the model itself says about being evaluated",
                 fontsize=11.5, fontweight="bold", x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    footnote(fig, "Training-contrast rows excluded. AUC is pooled within cells (Mantel-Haenszel): the "
                  "naive pooled AUC is set by cell composition, since about 96 % of pairs are between "
                  "cells. 0.5 means the score orders yes/no answers no better than chance.")
    fig.savefig(out / "fig5_ask_model.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────── fig 6
def fig6(rows, out, artifact="results/p4_frozen.json"):
    """Story-matched contrast: removing the whole registered lexicon vs changing the request.
    This is the figure for the claim that the axis is not a lexicon lookup."""
    art = Path(artifact)
    if not art.exists():
        raise RuntimeError(f"{artifact} not found (story_map lives there)")
    sm = json.loads(art.read_text(encoding="utf-8")).get("story_map") or {}
    if not sm:
        raise RuntimeError("story_map is empty in p4_frozen.json")
    by_pid = {r["prompt_id"]: r["s_last_dom"] for r in rows if r["s_last_dom"] is not None}
    dep_tr = np.array([r["s_last_dom"] for r in rows if r["cell"] == "deployment"
                       and r["is_train"] and r["s_last_dom"] is not None])
    base = dep_tr if len(dep_tr) else scores(rows, "deployment")
    e1_shift = scores(rows, "E1").mean() - base.mean()

    triples = []
    for ids in sm.values():
        ids = list(ids)
        if len(ids) < 3:
            continue
        v = [by_pid.get(i) for i in ids[:3]]
        if all(x is not None for x in v):
            triples.append(v)
    if not triples:
        raise RuntimeError("no complete a1/a2/a_head triples found in records_full.csv")
    A = np.array(triples)                     # columns: a1, a2, a_head

    panels = [("A · Remove the ENTIRE registered lexicon", 0, 1,
               "a1\nevaluation wording", "a2\nwording stripped"),
              ("B · Keep the wording, change the request", 0, 2,
               "a1\n“assess the system”", "a_head\n“write me a letter”")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    for ax, (title, i, j, li, lj) in zip(axes, panels):
        d = (A[:, i] - A[:, j]) / e1_shift
        for k in range(len(A)):
            up = A[k, i] >= A[k, j]
            ax.plot([0, 1], [A[k, i], A[k, j]], "-", lw=1.2, alpha=0.8,
                    color=C["positive"] if up else C["train_neg"], zorder=3)
        ax.plot([0] * len(A), A[:, i], "o", ms=5, color=INK, zorder=4)
        ax.plot([1] * len(A), A[:, j], "o", ms=5, color=INK, zorder=4)
        ax.plot([-0.12, 0.12], [A[:, i].mean()] * 2, color=INK, lw=2.4, zorder=5)
        ax.plot([0.88, 1.12], [A[:, j].mean()] * 2, color=INK, lw=2.4, zorder=5)
        same = int(sum(1 for x in d if x > 0))
        ax.set_xticks([0, 1]); ax.set_xticklabels([li, lj], fontsize=8.5)
        ax.set_xlim(-0.35, 1.35)
        ax.grid(axis="y", zorder=0)
        ax.set_title(f"{title}\nmean Δ = {d.mean():+.3f} E1   ({same}/{len(d)} same sign)",
                     fontsize=9.5)
    axes[0].set_ylabel("probe score")
    fig.suptitle("Same 8 stories: the wording barely matters, the type of request does",
                 fontsize=11.5, fontweight="bold", x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    footnote(fig, "One line per story; both panels start from the same a1 prompts. Δ is in units of "
                  "the E1 shift. Panel A is a synonym substitution with the meaning preserved, so a "
                  "null there only rules out a lexicon lookup, nothing more.")
    fig.savefig(out / "fig6_lexicon_vs_request.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────── fig 7
def fig7(N, out):
    """Steering dose-response. Left: what the model says. Right: manipulation check."""
    pre = "probe.exploratory.p11."
    # ключ вида <question>.<direction>.lam<λ>.yes_rate — λ надо отделить от суффикса ключа
    lam_keys = sorted({k[len(pre):].rsplit(".lam", 1)[-1].split(".")[0] for k in N
                       if k.startswith(pre) and ".lam" in k and k.endswith(".yes_rate")},
                      key=float)
    if not lam_keys:
        raise RuntimeError("no probe.exploratory.p11.*.lam*.yes_rate keys — run P11 first")
    lams = [float(x) for x in lam_keys]
    qs = sorted({k[len(pre):].split(".")[0] for k in N
                 if k.startswith(pre) and k.endswith(".yes_rate")})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax = axes[0]
    style = {"probe": ("-", "o"), "random_control": ("--", "s"),
             "shuffled_label_control": (":", "^")}
    for q in qs:
        for dname in style:
            xs, ys = [], []
            for lam, ls in zip(lams, lam_keys):
                v = N.get(f"{pre}{q}.{dname}.lam{ls}.yes_rate")
                if v is None and dname != "probe" and lam == 0:
                    v = N.get(f"{pre}{q}.probe.lam0.yes_rate")
                if v is not None:
                    xs.append(lam); ys.append(v)
            if xs:
                l, m = style[dname]
                ax.plot(xs, ys, l, marker=m, ms=5,
                        color=C["positive"] if q == "self" else C["train_neg"],
                        alpha=1.0 if dname == "probe" else 0.55,
                        label=f"{q} · {dname.replace('_', ' ')}")
    ref = N.get(pre + "real_insertion_reference.E1.yes_rate")
    if ref is not None:
        ax.axhline(ref, color="#666666", ls="-.", lw=1.1)
        ax.text(lams[-1], ref, " real insertion (E1)", fontsize=8, color="#666666",
                va="bottom", ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("λ  (in units of the shift produced by the real insertion)")
    ax.set_ylabel("fraction of “yes” answers")
    ax.set_title("A · What the model says")
    ax.grid(zorder=0)
    ax.legend(loc="center right")

    ax = axes[1]
    for dname in style:
        xs, ys = [], []
        for lam, ls in zip(lams, lam_keys):
            v = N.get(f"{pre}{qs[0]}.{dname}.lam{ls}.mean_score_at_last")
            if v is None and dname == "random_control" and lam == 0:
                v = N.get(f"{pre}{qs[0]}.probe.lam0.mean_score_at_last")
            if v is not None:
                xs.append(lam); ys.append(v)
        if xs:
            l, m = style[dname]
            ax.plot(xs, ys, l, marker=m, ms=5,
                    color=INK if dname == "probe" else ("#999999" if dname == "random_control"
                                                        else "#c0a060"),
                    label=dname.replace("_", " "))
    e1s = N.get(pre + "e1_shift_unit_dir")
    if e1s and xs:
        y0 = ys[0] if ys else 0
        ax.plot(lams, [y0 + l * e1s for l in lams], ":", color=C["positive"],
                label="expected if the hook is applied")
    ax.set_xlabel("λ")
    ax.set_ylabel("mean probe score at the read position")
    ax.set_title("B · Manipulation check")
    ax.grid(zorder=0)
    ax.legend(loc="upper left")
    fig.suptitle("Steering along the direction: dose-response and manipulation check",
                 fontsize=11.5, fontweight="bold", x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    footnote(fig, "Panel B must show the black line tracking the dotted one, otherwise the "
                  "intervention never reached the residual stream and panel A means nothing. "
                  "Check the baseline at λ = 0 for floor and ceiling effects before reading a null.")
    fig.savefig(out / "fig7_steering.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────── main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifact", default="results/p4_frozen.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    N = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    rows = load(out)
    e1 = scores(rows, "E1")
    tau = float(np.quantile(e1, 0.05)) if len(e1) else 0.0
    jobs = [("fig1_scores_by_cell.png", lambda: fig1(rows, out, tau)),
            ("fig2_insert_position.png", lambda: fig2(rows, N, out)),
            ("fig3_precision.png", lambda: fig3(rows, N, out)),
            ("fig4_perm_bands.png", lambda: fig4(N, out)),
            ("fig5_ask_model.png", lambda: fig5(rows, out)),
            ("fig6_lexicon_vs_request.png", lambda: fig6(rows, out, a.artifact)),
            ("fig7_steering.png", lambda: fig7(N, out))]
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
    # при вставке скрипта прямо в ячейку ноутбука argv содержит -f <kernel.json> от Jupyter,
    # и argparse падает; в этом случае разбираем пустой argv
    main([] if "__file__" not in globals() else None)
