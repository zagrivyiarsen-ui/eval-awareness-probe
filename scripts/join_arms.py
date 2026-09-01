"""Связка плеч (план §4.7 шаг 4; addendum §13). probe_scores.csv × behaviour_deltas.csv → results/join_arms.csv (n_dp × 6 строк среды),
промпт-уровневая корреляция сдвигов (Pearson/Spearman) по условиям, кросс-таблица H-dissoc по условиям (поведение значимо? / пробе значимо?).
    PYTHONPATH=. python scripts/join_arms.py --out results --env secret_number"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import join_arms, paired_perm_test_py, pearson, spearman
from src.conditions import MAIN_IDS
from src.contracts import record_numbers


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--env", required=True)
    a = ap.parse_args(argv); out = Path(a.out); nums = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    import os as _os   # R-Launch Д-2: env == pilot.decision.env при n_env=1 (иначе тихий джойн чужой среды)
    if nums.get("pilot.decision.n_env") == 1 and nums.get("pilot.decision.env") and a.env != nums["pilot.decision.env"]:
        if _os.environ.get("EVALAWARE_FORCE_ENV"):
            record_numbers(out / "numbers.json", {"join.deviation.force_env": f"env={a.env} vs decision {nums['pilot.decision.env']}"})
        else:
            raise SystemExit(f"join_arms: env={a.env}, а pilot.decision.env={nums['pilot.decision.env']} — поправь PARAMS.env или EVALAWARE_FORCE_ENV=1 (deviation, пишется в numbers)")
    probe = {r["prompt_key"]: float(r["probe_score"]) for r in csv.DictReader(open(out / "probe_scores.csv", encoding="utf-8")) if r["env"] == a.env}
    beh = {}
    for r in csv.DictReader(open(out / "behaviour_deltas.csv", encoding="utf-8")):
        k = f"{r['env_prompt']}/{r['condition_id']}"
        beh[k] = {"rate": float(r["rate"]) if r["rate"] else None, "lower": float(r["lower"]) if r["lower"] else None, "upper": float(r["upper"]) if r["upper"] else None}
    conds = [c for c in MAIN_IDS if any(k.endswith("/" + c) for k in beh)]
    n_dp = len({k.rsplit("/", 1)[0] for k in probe})
    rows = join_arms(beh, probe, n_prompts=n_dp, conds=conds)   # R-Launch: убран мёртвый фильтр `or True`
    with open(out / "join_arms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by = defaultdict(dict)
    for r in rows: by[f"{r['env']}/{r['prompt_id']}"][r["condition_id"]] = r
    J = {"env": a.env, "n_rows": len(rows), "n_prompts": n_dp}
    for c in conds:
        if c == "deployment": continue
        db, dp = [], []
        for p, d in by.items():
            if c in d and "deployment" in d and d[c]["beh_rate"] is not None and d["deployment"]["beh_rate"] is not None:
                db.append(d[c]["beh_rate"] - d["deployment"]["beh_rate"]); dp.append(d[c]["probe_score"] - d["deployment"]["probe_score"])
        J[f"corr.{c}"] = {"pearson": pearson(db, dp), "spearman": spearman(db, dp), "n": len(db)}
        pb = nums.get(f"beh.delta.{c}.p"); pp = nums.get(f"probe.delta.{a.env}.{c}.p")
        J[f"dissoc.{c}"] = {"beh_p": pb, "probe_p": pp, "beh_sig": (pb is not None and pb <= 0.05), "probe_sig": (pp is not None and pp <= 0.05),
                            "cell": ("both" if (pb or 1) <= 0.05 and (pp or 1) <= 0.05 else "probe_only" if (pp or 1) <= 0.05 else "beh_only" if (pb or 1) <= 0.05 else "neither"),
                            "beh_mde": nums.get(f"beh.mde_actual.{c}")}
    # общая корреляция по всем условиям (пул промпт×условие дельт)
    db, dp = [], []
    for p, d in by.items():
        for c in conds:
            if c != "deployment" and c in d and d[c]["beh_rate"] is not None and d["deployment"]["beh_rate"] is not None:
                db.append(d[c]["beh_rate"] - d["deployment"]["beh_rate"]); dp.append(d[c]["probe_score"] - d["deployment"]["probe_score"])
    J["corr.all_conditions"] = {"pearson": pearson(db, dp), "spearman": spearman(db, dp), "n": len(db)}
    # 2026-08-28 (Р2 3.6, kill-switch, 0 GPU): shuffled-join null — пробе-дельты перемешиваются между промптами (200 перестановок,
    # сид 0) → |corr| должна лечь в нулевую полосу; реальная corr сравнивается с q95. Санити-репорт, не ворота.
    import random as _rnd
    _rng = _rnd.Random(0); null_abs = []
    if len(db) >= 3:
        for _ in range(200):
            perm = dp[:]; _rng.shuffle(perm)
            c0 = pearson(db, perm)
            if c0 is not None: null_abs.append(abs(c0))
        null_abs.sort()
        J["corr.shuffled_null"] = {"q95_abs": null_abs[int(0.95 * len(null_abs)) - 1] if null_abs else None,
                                   "real_abs": abs(J["corr.all_conditions"]["pearson"]) if J["corr.all_conditions"]["pearson"] is not None else None,
                                   "n_perm": len(null_abs)}
    # 2026-08-28 (Р2-В1): confirmatory-семейство (analysis.yaml inference) — Holm по 4 заранее названным p; остальное exploratory.
    from src.scoring import load_analysis
    INF = load_analysis().get("inference") or {}
    fam = list(INF.get("confirmatory_family") or []); alpha = float(INF.get("alpha", 0.05))
    ps = [(k2, nums.get(k2)) for k2 in fam]
    known = [(k2, p2) for k2, p2 in ps if p2 is not None]
    if known:
        order = sorted(known, key=lambda kv: kv[1]); m = len(order); holm = {}
        prev_adj = 0.0
        for i, (k2, p2) in enumerate(order):
            adj = min(1.0, max(prev_adj, (m - i) * p2)); prev_adj = adj
            holm[k2] = {"p": p2, "p_holm": adj, "significant": adj <= alpha}
        J["inference"] = {"alpha": alpha, "correction": "holm", "family_size": len(fam),
                          "missing": [k2 for k2, p2 in ps if p2 is None], **{f"holm.{k2}": v for k2, v in holm.items()}}
    # R-Launch В-4 (analysis.yaml join_arms.dissociation_rule): «диссоциация» — только при gate_v5; иначе «пробе не обобщился»
    gv5 = bool(nums.get("probe.gate_v5"))
    J["dissociation_label_allowed"] = gv5
    record_numbers(out / "numbers.json", {f"join.{k}": v for k, v in J.items()})
    print(json.dumps(J, indent=1))
    print("H-dissoc кросс-таблица (условие → ячейка):", {c: J[f"dissoc.{c}"]["cell"] for c in conds if c != "deployment"})
    if not gv5:
        print("!! probe.gate_v5=false: ячейки probe_only НЕ называются диссоциацией — язык «пробе не обобщился» (dissociation_rule)")


if __name__ == "__main__":
    main()
