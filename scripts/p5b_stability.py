#!/usr/bin/env python3
"""P5b — недостающий блок P5: устойчивость направления при разбиении пополам,
эффективный ранг ковариации deployment и наборы сработавших под разными направлениями.

Нынешний p5_lda_and_b3.py считает профили направлений, но не считает split-half —
поэтому раздел 11 дампа неполон, а стадия P5 помечена MISSING (маркер стадии —
ключ probe.exploratory.p5.direction_stability.diff_of_means.cos_mean).

Читает те же замороженные активации, пишет ТОЛЬКО под probe.exploratory.p5.*,
существующие ключи p5.profiles не трогает.

    PYTHONPATH=. python3 scripts/p5b_stability.py --out results --artifact results/p4_frozen.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent if _here.name == "scripts" else _here))

POS = "last"
TOPICAL = ("a1", "a2", "d", "d_legacy", "a_legacy")


def tau_loo(e1, q=0.05):
    return float(np.quantile(e1, q)), np.array(
        [np.quantile(np.delete(e1, i), q) for i in range(len(e1))])


class D:
    def __init__(self, out: Path, art: dict):
        import torch
        base = torch.load(out / "probe_acts_fp16.pt", weights_only=False)
        extra = torch.load(out / "probe_acts_extra_fp16.pt", weights_only=False)
        self.layers = list(base["layers"])
        self.keys = list(base["keys"]) + list(extra["keys"])
        self.X = {p: {Lr: np.concatenate([base["acts"][p][Lr].float().numpy(),
                                          extra["acts"][p][Lr].float().numpy()])
                      for Lr in self.layers} for p in base["positions"]}
        a_leg = set(art["legacy_reclass"]["a_legacy"])
        d_leg = set(art["legacy_reclass"]["d_legacy"])
        self.env, self.pid, self.cell = [], [], []
        for k in self.keys:
            e, p, c = k.split("/")
            self.env.append(e); self.pid.append(p)
            self.cell.append(("a_legacy" if p in a_leg else "d_legacy" if p in d_leg else "topic_unk")
                             if e == "topic" else (e[2:] if e.startswith("x_") else c))
        self.cell = np.array(self.cell); self.envA = np.array(self.env); self.pidA = np.array(self.pid)
        self.N = len(self.keys)

    def m(self, *cs):
        return np.isin(self.cell, cs)

    def dep(self):
        return self.m("deployment") & (self.envA != "topic") & \
            ~np.char.startswith(self.envA.astype(str), "x_")


def fit(name, Xtr, y):
    """Направление по имени метода. diff_of_means — аналитически, остальные — sklearn."""
    if name == "diff_of_means":
        return Xtr[y == 1].mean(0) - Xtr[y == 0].mean(0)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    if name == "lda_shrinkage":
        c = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, y)
        return np.asarray(c.coef_).ravel()
    if name == "linear_svc":
        c = LinearSVC(C=1.0, max_iter=20000).fit(Xtr, y)
        return np.asarray(c.coef_).ravel()
    if name == "pca20_lda_svd":
        p = make_pipeline(PCA(n_components=20),
                          LinearDiscriminantAnalysis(solver="svd")).fit(Xtr, y)
        return p.steps[0][1].components_.T @ np.asarray(p.steps[1][1].coef_).ravel()
    raise ValueError(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifact", default="results/p4_frozen.json")
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260901)
    a = ap.parse_args(argv if argv is not None else
                      ([] if "__file__" not in globals() else None))
    out = Path(a.out)
    nums = out / "numbers.json"
    N = json.loads(nums.read_text(encoding="utf-8"))
    L = int(N["probe.layer"])
    art = json.loads(Path(a.artifact).read_text(encoding="utf-8"))
    d = D(out, art)

    X = d.X[POS][L]
    me1, mdep = d.m("E1"), d.dep()
    contrast = np.where(me1 | mdep)[0]
    y_all = np.array([1.0 if me1[i] else 0.0 for i in contrast])
    pids = np.array([f"{d.env[i]}/{d.pid[i]}" for i in contrast])
    uniq = sorted(set(pids))
    methods = ("diff_of_means", "lda_shrinkage", "linear_svc", "pca20_lda_svd")

    # ── split-half по ПРОМПТАМ: пара half-направлений, косинус между ними
    rng = np.random.default_rng(a.seed)
    stab = {}
    for name in methods:
        cs = []
        for _ in range(a.n_splits):
            perm = list(uniq)
            rng.shuffle(perm)
            half = set(perm[:len(perm) // 2])
            m1 = np.array([p in half for p in pids])
            try:
                w1 = fit(name, X[contrast][m1], y_all[m1])
                w2 = fit(name, X[contrast][~m1], y_all[~m1])
            except Exception as e:
                cs = []
                stab[name] = {"unavailable": f"{type(e).__name__}: {str(e)[:60]}"}
                break
            cs.append(float(w1 @ w2 / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-12)))
        if cs:
            stab[name] = {"cos_mean": float(np.mean(cs)), "cos_sd": float(np.std(cs)),
                          "n_splits": len(cs), "cos_min": float(np.min(cs))}
        print(f"  {name:16} " + (f"cos = {stab[name].get('cos_mean', float('nan')):.3f} ± "
                                 f"{stab[name].get('cos_sd', float('nan')):.3f} "
                                 f"({stab[name].get('n_splits', 0)} разбиений)"
                                 if "cos_mean" in stab[name] else stab[name]["unavailable"]))

    # ── эффективный ранг ковариации deployment
    Xd = X[mdep]
    sv = np.linalg.svd(Xd - Xd.mean(0), compute_uv=False)
    ev = sv ** 2
    p_ = ev / ev.sum()
    eff_rank = float(np.exp(-(p_ * np.log(p_ + 1e-300)).sum()))
    stab["effective_rank_deployment"] = eff_rank
    stab["note"] = ("устойчивость НАПРАВЛЕНИЯ не равна устойчивости НАБОРА сработавших: "
                    "порог стоит на 37 внешних точках, и один сдвиг границы меняет набор. "
                    "Эффективный ранг ковариации deployment близок к 1 → «отбеливание» "
                    "здесь означает «убрать первую главную компоненту».")
    print(f"  эффективный ранг ковариации deployment: {eff_rank:.2f}")

    # ── наборы сработавших под каждым направлением
    Xtr, ytr = X[contrast], y_all
    mw = np.where(d.m("wild_other"))[0]
    fired = {}
    for name in methods:
        try:
            w = fit(name, Xtr, ytr)
        except Exception:
            continue
        s = X @ w
        e1s = s[me1]
        _, loo = tau_loo(e1s)
        tau = float(np.median(loo))
        ids = sorted(d.pid[i] for i in mw if s[i] >= tau)
        fired[name] = {"n_fired": len(ids), "n_total": int(len(mw)), "ids": ids,
                       "tau_median_loo": tau}
        print(f"  {name:16} сработало {len(ids)}/{len(mw)} внешних: {ids}")

    over = {}
    base = set(fired.get("diff_of_means", {}).get("ids", []))
    for name, v in fired.items():
        if name == "diff_of_means":
            continue
        s_ = set(v["ids"])
        over[name] = {"shared_with_diff_of_means": sorted(base & s_),
                      "only_here": sorted(s_ - base), "only_in_diff_of_means": sorted(base - s_)}

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
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    record_numbers(nums, {"probe.exploratory.p5.direction_stability": san(stab),
                          "probe.exploratory.p5.fired_sets": san(fired),
                          "probe.exploratory.p5.fired_overlap": san(over)})

    cur = json.loads(nums.read_text(encoding="utf-8"))
    marker = "probe.exploratory.p5.direction_stability.diff_of_means.cos_mean"
    print(f"\nмаркер стадии {marker}: "
          + ("записан" if marker in cur else "НЕ ЗАПИСАН — дамп по-прежнему пометит P5 как MISSING"))
    print(f"ключей p5.*: {sum(1 for k in cur if k.startswith('probe.exploratory.p5.'))}")
    print("дальше: пересоберите FINAL_DUMP_EN.md")


if __name__ == "__main__":
    main()