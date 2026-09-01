#!/usr/bin/env python3
"""P3 — ремонтная стадия: ведущая процедура FPR для ВСЕХ ячеек.

Зачем. Ведущей объявлена процедура (3): leave-one-decision-point-out, где на каждом фолде
заново считаются И направление, И порог. В снимке она покрывает 10 ячеек из 15 — для
`b`, `c1`, `c2`, `len140`, `len210` в разделе 6 дампа стоит прочерк, и сканер аномалий это
отмечает. Здесь она досчитывается для всех ячеек единообразно.

Процедура, повторяющая ведущую:
  для каждой из 32 пар decision point:
      направление  = mean(E1 без этой пары) − mean(deployment без этой пары)
      порог        = 5-й процентиль скоров E1 без этой пары
      доля ячейки выше порога
  итог = среднее по 32 фолдам.

Пишет ТОЛЬКО под probe.exploratory.p3.*, ничего существующего не трогает.

    PYTHONPATH=. python3 scripts/p3_repair.py --out results --artifact results/p4_frozen.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent if _here.name == "scripts" else _here))

POS = "last"
CELLS = ["a_legacy", "d_legacy", "a1", "a2", "a_head", "a_tail", "d", "b", "c1", "c2",
         "n", "dptrunc", "len140", "len210", "wild_other"]
TOPICAL = ("a1", "a2", "d", "d_legacy", "a_legacy")
NEUTRAL = ("n", "wild_other")


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


class D:
    def __init__(self, out: Path, art: dict):
        import torch
        base = torch.load(out / "probe_acts_fp16.pt", weights_only=False)
        extra = torch.load(out / "probe_acts_extra_fp16.pt", weights_only=False)
        self.keys = list(base["keys"]) + list(extra["keys"])
        self.X = np.concatenate([base["acts"][POS][int(art["_layer"])].float().numpy(),
                                 extra["acts"][POS][int(art["_layer"])].float().numpy()]) \
            if "_layer" in art else None
        self._base, self._extra = base, extra
        a_leg = set(art["legacy_reclass"]["a_legacy"])
        d_leg = set(art["legacy_reclass"]["d_legacy"])
        env, pid, cell = [], [], []
        for k in self.keys:
            e, p, c = k.split("/")
            env.append(e); pid.append(p)
            cell.append(("a_legacy" if p in a_leg else "d_legacy" if p in d_leg else "topic_unk")
                        if e == "topic" else (e[2:] if e.startswith("x_") else c))
        self.envA, self.pidA, self.cell = np.array(env), np.array(pid), np.array(cell)

    def acts(self, L: int):
        return np.concatenate([self._base["acts"][POS][L].float().numpy(),
                               self._extra["acts"][POS][L].float().numpy()])

    def m(self, *cs):
        return np.isin(self.cell, cs)

    def dep(self):
        return self.m("deployment") & (self.envA != "topic") & \
            ~np.char.startswith(self.envA.astype(str), "x_")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifact", default="results/p4_frozen.json")
    ap.add_argument("--q", type=float, default=0.05, help="квантиль E1 для порога")
    a = ap.parse_args(argv if argv is not None else
                      ([] if "__file__" not in globals() else None))
    out = Path(a.out)
    nums = out / "numbers.json"
    N = json.loads(nums.read_text(encoding="utf-8"))
    L = int(N["probe.layer"])
    art = json.loads(Path(a.artifact).read_text(encoding="utf-8"))
    d = D(out, art)
    X = d.acts(L)

    me1, mdep = d.m("E1"), d.dep()
    dps = sorted({f"{d.envA[i]}/{d.pidA[i]}" for i in range(len(d.keys)) if me1[i] or mdep[i]})
    dp_of = np.array([f"{d.envA[i]}/{d.pidA[i]}" for i in range(len(d.keys))])

    # ── ведущая процедура: на каждом фолде пересчитываются и направление, и порог
    per_cell = {c: [] for c in CELLS}
    tprs, taus, aucs = [], [], []
    for held in dps:
        keep = dp_of != held
        w = X[me1 & keep].mean(0) - X[mdep & keep].mean(0)
        s = X @ w
        tau = float(np.quantile(s[me1 & keep], a.q))
        taus.append(tau)
        held_e1 = s[me1 & ~keep]
        if len(held_e1):
            tprs.append(float((held_e1 >= tau).mean()))
        held_dep = s[mdep & ~keep]
        if len(held_e1) and len(held_dep):
            aucs.append(float((held_e1[:, None] > held_dep[None, :]).mean()))
        for c in CELLS:
            mc = d.m(c)
            if mc.sum():
                per_cell[c].append(float((s[mc] >= tau).mean()))

    fpr = {c: float(np.mean(v)) for c, v in per_cell.items() if v}
    n_of = {c: int(d.m(c).sum()) for c in CELLS}
    tpr_oos = float(np.mean(tprs)) if tprs else float("nan")
    tau_med = float(np.median(taus))

    # ── группы
    def grp(cells):
        idx = d.m(*cells)
        vals = []
        for held in dps:
            keep = dp_of != held
            w = X[me1 & keep].mean(0) - X[mdep & keep].mean(0)
            s = X @ w
            tau = float(np.quantile(s[me1 & keep], a.q))
            vals.append(float((s[idx] >= tau).mean()))
        return float(np.mean(vals)), int(idx.sum())

    fpr_top, n_top = grp(TOPICAL)
    fpr_neu, n_neu = grp(NEUTRAL)

    R = {
        "tau_median_loo": tau_med,
        "tpr_oos": tpr_oos,
        "auc_oos": float(np.mean(aucs)) if aucs else float("nan"),
        "n_folds": len(dps),
        "q": a.q,
        "fully_oos_fpr_all_cells": fpr,
        "n_by_cell": n_of,
        "wilson95_by_cell": {c: wilson(int(round(fpr[c] * n_of[c])), n_of[c])
                             for c in fpr},
        "groups": {
            "topical_leading": {"fpr": fpr_top, "n": n_top,
                                "precision_at_1pct": precision_at(0.01, tpr_oos, fpr_top)},
            "neutral": {"fpr": fpr_neu, "n": n_neu,
                        "precision_at_1pct": precision_at(0.01, tpr_oos, fpr_neu)},
        },
        "procedure": ("leave-one-decision-point-out; on each of the 32 folds BOTH the direction "
                      "and the threshold are recomputed on the remaining pairs, then every cell "
                      "is scored; the reported value is the mean over folds. This is the leading "
                      "procedure of the project and it now covers all 15 cells, closing the gap "
                      "that earlier snapshots flagged."),
        "caveat": ("tpr_oos is ~0.95 by construction, since the threshold is the "
                   f"{int(a.q*100)}th percentile of the same E1 scores — not a result."),
    }

    from src.contracts import record_numbers

    def san(o):
        if isinstance(o, dict):
            return {str(k): san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [san(x) for x in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return o

    record_numbers(nums, {"probe.exploratory.p3": san(R)})

    print(f"{'cell':12} {'n':>4} {'fully OOS FPR':>14}  Wilson 95 %")
    print("-" * 58)
    for c in CELLS:
        if c not in fpr:
            continue
        w = R["wilson95_by_cell"][c]
        print(f"{c:12} {n_of[c]:>4} {fpr[c]:>14.3f}  [{w[0]:.2f}; {w[1]:.2f}]")
    print(f"\nτ медиана по фолдам {tau_med:.3f} · TPR_oos {tpr_oos:.3f} "
          f"(≈0.95 по построению) · AUC_oos {R['auc_oos']:.3f}")
    print(f"тематические: FPR {fpr_top:.3f} (n={n_top}), "
          f"precision при 1 % = {R['groups']['topical_leading']['precision_at_1pct']:.4f}")
    cur = json.loads(nums.read_text(encoding="utf-8"))
    print("\nмаркер стадии probe.exploratory.p3.tau_median_loo: "
          + ("записан" if "probe.exploratory.p3.tau_median_loo" in cur else "НЕ ЗАПИСАН"))
    print("дальше: пересоберите FINAL_DUMP_EN.md")


if __name__ == "__main__":
    main()
