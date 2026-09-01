"""Поведенческое плечо (план §4.6; addendum §7, §11). Вход: results/pilot_gens.jsonl + main_gens.jsonl (выбранная среда), cap из yaml.
    PYTHONPATH=. python scripts/behaviour_analysis.py --out results --env secret_number
Выход: numbers beh.*, results/behaviour_deltas.csv (промпт × условие: rate, lower, upper, n_ok), results/behaviour_summary.md.
Считает: парные дельты vs deployment + перестановочный тест (основной), границы по обрыву (truncated/unparsable = 0 / = 1),
truncated_frac и длина CoT по условиям (исходы, парные тесты), unparsable/multi/explore по условиям, вербализация × условие,
аддитивность одной строкой (E1_S1 − E1 − S1, если есть), S−Dstr, фактический MDE по SD дельт, чувствительность для secret_number."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import behaviour_by_prompt_key, load_gens, mde, paired_perm_test_py, trim_to_equal_k
from src.conditions import MAIN_IDS, load_conditions
from src.contracts import record_numbers
from src.scoring import load_analysis


def frac(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml"); ap.add_argument("--analysis", default="configs/analysis.yaml")
    ap.add_argument("--out", default="results"); ap.add_argument("--env", required=True)
    ap.add_argument("--n-perm", type=int, default=5000)
    a = ap.parse_args(argv)
    out = Path(a.out); nums = out / "numbers.json"
    cfg = load_conditions(a.config, allow_unset_gate_params=True); A = load_analysis(a.analysis)
    prev = json.loads(nums.read_text(encoding="utf-8"))
    # 2026-08-28: приоритет main.cap → pilot.decision.cap → yaml (в блобе yaml несёт 2048 до fill_gate_params — резать по нему нельзя)
    cap = int(prev.get("main.cap") or prev.get("pilot.decision.cap") or cfg.generation.get("max_new_tokens") or 4096)
    R_cap_source = "main.cap" if prev.get("main.cap") else ("pilot.decision.cap" if prev.get("pilot.decision.cap") else "yaml/default")
    gens = load_gens(out, a.env, cap)
    # 2026-08-28 (R-Launch, analysis.yaml cut_rule.k_shortfall_rule): равный k′ по всем ячейкам; хвост неполного слоя — вон
    gens, trim = trim_to_equal_k(gens)
    conds = [c for c in MAIN_IDS if any(g["condition_id"] == c for g in gens)]
    R = {"env": a.env, "cap": cap, "cap_source": R_cap_source, "n_gens": len(gens), "conditions": conds,
         "k_equal": trim["k_equal"], "n_dropped_to_equal_k": trim["n_dropped"]}
    if trim["k_equal"] < 3:
        print(f"!! k_equal={trim['k_equal']} < 3 — по k_shortfall_rule поведенческое плечо снимается (репортится только пробе)")
    # ---------------- per prompt_key rates (complete-case + bounds), sensitivity for secret_number
    beh = behaviour_by_prompt_key(gens)
    beh_sens = behaviour_by_prompt_key(gens, target_field="target_sensitivity") if a.env == "secret_number" else {}
    by = defaultdict(dict)
    for k, v in beh.items():
        by[k.rsplit("/", 1)[0]][k.split("/")[-1]] = v
    prompts = sorted(by)
    rows = []
    for p in prompts:
        for c in conds:
            v = by[p].get(c) or {}
            rows.append({"env_prompt": p, "condition_id": c, "rate": v.get("rate"), "lower": v.get("lower"), "upper": v.get("upper"),
                         "n": v.get("n"), "n_ok": v.get("n_ok"), "n_excluded": v.get("n_excluded")})
    with open(out / "behaviour_deltas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    R["k_per_cell_min"] = min((v.get("n") or 0) for p in prompts for v in by[p].values())
    R["k_per_cell_max"] = max((v.get("n") or 0) for p in prompts for v in by[p].values())
    # ---------------- rates & deltas
    for c in conds:
        R[f"rate.{c}"] = frac([by[p].get(c, {}).get("rate") for p in prompts])
        R[f"rate_lower.{c}"] = frac([by[p].get(c, {}).get("lower") for p in prompts])
        R[f"rate_upper.{c}"] = frac([by[p].get(c, {}).get("upper") for p in prompts])
    deltas = {}
    for c in conds:
        if c == "deployment": continue
        d = [(by[p].get(c, {}).get("rate"), by[p]["deployment"].get("rate")) for p in prompts if "deployment" in by[p]]
        d = [x - y for x, y in d if x is not None and y is not None]
        deltas[c] = d
        R[f"delta.{c}"] = paired_perm_test_py(d, a.n_perm)
        dl = [(by[p].get(c, {}).get("lower") or 0) - (by[p]["deployment"].get("upper") or 0) for p in prompts]
        du = [(by[p].get(c, {}).get("upper") or 0) - (by[p]["deployment"].get("lower") or 0) for p in prompts]
        R[f"delta_bounds.{c}"] = {"lower_mean": mean(dl) if dl else None, "upper_mean": mean(du) if du else None,
                                  "sign_robust": (mean(dl) > 0) == (mean(du) > 0) if dl else None}
    if "S1" in deltas and "Dstr" in deltas:
        for s in ("S1", "S2"):
            if s in deltas:
                R[f"S_minus_Dstr.{s}"] = paired_perm_test_py([x - y for x, y in zip(deltas[s], deltas["Dstr"])], a.n_perm)
        if "S2" in deltas:
            R["S_minus_Dstr.S1S2mean"] = paired_perm_test_py([(x + z) / 2 - y for x, z, y in zip(deltas["S1"], deltas["S2"], deltas["Dstr"])], a.n_perm)
    if all(c in deltas for c in ("E1_S1", "E1", "S1")):
        R["additivity.interaction"] = paired_perm_test_py([a_ - b - c for a_, b, c in zip(deltas["E1_S1"], deltas["E1"], deltas["S1"])], a.n_perm)
    if "E1" in deltas and deltas["E1"]:
        for s in ("S1", "S2", "Dstr"):
            if s in deltas and R["delta.E1"]["mean"]:
                R[f"proportion_of_E1.{s}"] = R[f"delta.{s}"]["mean"] / R["delta.E1"]["mean"]
    # фактический MDE по SD дельт (n = число промптов)
    for c, d in deltas.items():
        if len(d) > 1 and R[f"delta.{c}"]["sd"] is not None:
            R[f"mde_actual.{c}"] = mde(R[f"delta.{c}"]["sd"], len(d))
    # ---------------- обрыв, длина, unparsable, multi, explore — по условиям (исходы) + парные тесты vs deployment
    byc = defaultdict(list)
    for g in gens: byc[g["condition_id"]].append(g)
    for c in conds:
        gs = byc[c]
        R[f"truncated_frac.{c}"] = frac([g["cot_category"] == "truncated" for g in gs])
        R[f"cot_len_mean.{c}"] = frac([g["n_generated_tokens"] for g in gs])
        R[f"unparsable_frac.{c}"] = frac([g["cot_category"] == "ok" and g["score"]["class"] == "unparsable" for g in gs])
        R[f"no_close_eos_frac.{c}"] = frac([g["cot_category"] == "no_close_eos" for g in gs])
        if a.env == "secret_number":
            ok = [g for g in gs if g["cot_category"] == "ok"]
            for cls in ("cheat", "honest", "explore"):
                R[f"class_frac.{c}.{cls}"] = frac([g["score"]["class"] == cls for g in ok])
            R[f"multi_frac.{c}"] = frac([g["score"].get("multi", False) for g in ok])
        R[f"verbalization.{c}"] = frac([g["verbalization"]["verbalized"] for g in gs])
        # per-prompt paired vs deployment for truncation and length
        if c != "deployment":
            def per_prompt(cc, fn):
                d = defaultdict(list)
                for g in byc[cc]: d[g["prompt_key"].rsplit("/", 1)[0]].append(fn(g))
                return {p: mean(v) for p, v in d.items()}
            t1, t0 = per_prompt(c, lambda g: g["cot_category"] == "truncated"), per_prompt("deployment", lambda g: g["cot_category"] == "truncated")
            R[f"truncated_delta.{c}"] = paired_perm_test_py([t1[p] - t0[p] for p in t1 if p in t0], a.n_perm)
            l1, l0 = per_prompt(c, lambda g: g["n_generated_tokens"]), per_prompt("deployment", lambda g: g["n_generated_tokens"])
            R[f"cot_len_delta.{c}"] = paired_perm_test_py([l1[p] - l0[p] for p in l1 if p in l0], a.n_perm)
    # вербализация × поведение внутри условия (описательно, план §4.2)
    for c in conds:
        ok = [g for g in byc[c] if g["cot_category"] == "ok" and g["score"].get("target") is not None]
        v1 = [g["score"]["target"] for g in ok if g["verbalization"]["verbalized"]]; v0 = [g["score"]["target"] for g in ok if not g["verbalization"]["verbalized"]]
        R[f"verbalization_assoc.{c}"] = {"rate_verbalized": frac(v1), "n_verbalized": len(v1), "rate_not": frac(v0), "n_not": len(v0), "descriptive_only": True}
    if a.env == "secret_number" and beh_sens:
        for c in conds:
            R[f"rate_sensitivity.{c}"] = frac([beh_sens.get(f"{p}/{c}", {}).get("rate") for p in prompts])
    if a.env == "secret_number":
        # R-Launch В-2: вторичная верхняя граница cheat_any (regex по всему answer_text) — по условиям + парные дельты
        beh_any = behaviour_by_prompt_key(gens, target_field="target_cheat_any")
        for c in conds:
            R[f"rate_cheat_any.{c}"] = frac([beh_any.get(f"{p}/{c}", {}).get("rate") for p in prompts])
            if c != "deployment":
                d = [(beh_any.get(f"{p}/{c}", {}).get("rate"), beh_any.get(f"{p}/deployment", {}).get("rate")) for p in prompts]
                d = [x - y for x, y in d if x is not None and y is not None]
                R[f"delta_cheat_any.{c}"] = paired_perm_test_py(d, a.n_perm)
    record_numbers(nums, {f"beh.{k}": v for k, v in R.items()})
    # ---------------- summary md
    L = [f"# Behavioural arm — {a.env} (cap {cap}, k per cell {R['k_per_cell_min']}–{R['k_per_cell_max']}, n_prompts {len(prompts)})", "",
         "| condition | rate (complete-case) | bounds | Δ vs deployment | p (paired perm) | MDE actual | truncated | CoT len | unparsable | verbalized |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        d = R.get(f"delta.{c}", {})
        dm = "" if c == "deployment" else f"{d.get('mean'):+.3f}"
        dp = "" if c == "deployment" else f"{d.get('p')}"
        md = "" if c == "deployment" else f"{R.get(f'mde_actual.{c}', float('nan')):.3f}"
        L.append(f"| {c} | {R[f'rate.{c}']:.3f} | [{R[f'rate_lower.{c}']:.2f}, {R[f'rate_upper.{c}']:.2f}] | {dm} | {dp} | {md} | "
                 f"{R[f'truncated_frac.{c}']:.2f} | {R[f'cot_len_mean.{c}']:.0f} | {R[f'unparsable_frac.{c}']:.2f} | {R[f'verbalization.{c}']:.2f} |")
    for key in ("S_minus_Dstr.S1", "S_minus_Dstr.S2", "S_minus_Dstr.S1S2mean", "additivity.interaction"):
        if key in R: L.append(f"- {key}: mean {R[key]['mean']:+.3f}, p = {R[key]['p']}")
    (out / "behaviour_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
