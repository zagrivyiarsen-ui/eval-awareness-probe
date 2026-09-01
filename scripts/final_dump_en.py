#!/usr/bin/env python3
"""final_dump.py — FINAL package to hand over for independent analysis (English output).

Self-contained: does not import src/, does not need torch. Reads numbers.json,
records_full.csv, records_texts.jsonl, *_records.jsonl and the reports, and writes
results/FINAL_DUMP.md (that is the file to hand over).

What separates it from the intermediate dumps:
  · SECTION 1 — field completeness per file (catches joins that silently failed);
  · SECTION 2 — numbers.json aggregates RECOMPUTED from records_full.csv and reconciled;
  · SECTION 15 — an explicit list of numbers declared invalid, and why.

    python3 scripts/final_dump_en.py --out results

"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

W = 118
OUT_LINES: list[str] = []


def P(s=""):
    OUT_LINES.append(s)
    print(s)


def hr(t=""):
    P("\n" + "═" * W)
    if t:
        P("  " + t)
        P("═" * W)


def sub(t):
    P("\n── " + t + " " + "─" * max(0, W - len(t) - 5))


def fm(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v != 0 and abs(v) < 10 ** (-nd):
            return f"{v:.2g}"          # НЕ печатать «0» вместо малого ненулевого p
        return (f"{v:.{nd}f}".rstrip("0").rstrip(".") or "0") if abs(v) < 1e4 else f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return "[" + "; ".join(fm(x, nd) for x in v) + "]"
    s = str(v)
    return s if len(s) <= 76 else s[:73] + "…"


def table(rows, head):
    if not rows:
        P("  (no data)")
        return
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    w = [max(len(str(head[i])), *(len(r[i]) for r in rows)) for i in range(len(head))]
    P("  " + " │ ".join(str(head[i]).ljust(w[i]) for i in range(len(head))))
    P("  " + "─┼─".join("─" * x for x in w))
    for r in rows:
        P("  " + " │ ".join(r[i].ljust(w[i]) for i in range(len(head))))


N: dict = {}
R: list[dict] = []
SEEN: set = set()


def g(k, d=None):
    return N.get(k, d)


def mark(*pref):
    for k in N:
        if any(k.startswith(p) for p in pref):
            SEEN.add(k)


def fnum(r, c):
    v = r.get(c)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def wilson(k, n, z=1.96):
    if not n:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


CELLS = ("a_legacy", "d_legacy", "a1", "a2", "a_head", "a_tail", "d", "b", "c1", "c2",
         "n", "dptrunc", "len140", "len210", "wild_ai", "wild_other")
ROLE = {"a1": "topical", "a2": "topical", "d": "topical", "d_legacy": "topical", "a_legacy": "topical",
        "a_head": "diagn.", "a_tail": "diagn.", "b": "gradient", "c1": "gradient", "c2": "gradient",
        "n": "neutral", "wild_other": "neutral(ext)", "wild_ai": "neutral(AI)",
        "dptrunc": "structural", "len140": "ladder", "len210": "ladder"}


# ═══════════════════════════════════════════ 0
def s0(out: Path):
    hr("0. MANIFEST")
    P("  Project: eval-awareness probe, Qwen3.5-9B; the behavioural arm was closed by pivot 4.2,")
    P("  so the work is the probe arm. All numbers below come from results/; nothing is recomputed on GPU.")
    P(f"  keys in numbers.json: {len(N)} | rows in records_full.csv: {len(R)}")
    sub("stages")
    table([[s, "✓" if k in N else "MISSING", fm(g(k))] for s, k in
           [("Gate 0", "gate0.model_name"), ("prereg probe", "probe.layer"),
            ("pilot", "pilot.decision.rule"), ("pivot", "pivot.4_2.closed_20260829"),
            ("G1 (new cells)", "probe.exploratory.g1.artifact_sha256"),
            ("P2 (main analysis)", "probe.exploratory.p2.alt3_gate.outcome"),
            ("P3 (repair)", "probe.exploratory.p3.tau_median_loo"),
            ("P4 (B1–B9)", "probe.exploratory.p4.layer"),
            ("P5 (alt. directions)", "probe.exploratory.p5.direction_stability.diff_of_means.cos_mean")]],
          ["stage", "present", "value"])
    sub("hashes, environment")
    for k in ("prereg.blob_sha256", "prereg.code_sha256", "prereg.config_hash", "prereg.conditions_hash",
              "prereg.analysis_hash", "prereg.dps_sha256", "probe.exploratory.g1.artifact_sha256",
              "probe.exploratory.p4.p2_unchanged.ok", "gate0.model_name", "gate0.model_revision",
              "env.transformers", "env.torch", "env.python", "probe.dtype", "probe.layer", "probe.position"):
        if k in N:
            P(f"  {k:48} {fm(g(k))}")
    P(f"  code_sha256 changes: {len(g('prereg.config_change_log') or [])}")
    sub("files")
    rows = []
    for p in sorted(out.glob("*"), key=lambda x: -x.stat().st_size):
        if not p.is_file():
            continue
        nl = ""
        if p.suffix in (".jsonl", ".csv"):
            try:
                nl = str(sum(1 for _ in open(p, encoding="utf-8", errors="ignore")))
            except OSError:
                nl = "?"
        rows.append([p.name, f"{p.stat().st_size/1e6:.2f}", nl])
    table(rows, ["file", "MB", "rows"])
    mark("prereg.", "env.", "continuity.", "pivot.", "gate0.model", "probe.dtype", "probe.position",
         "probe.n_records", "probe.main_recorded_at", "probe.topic_sha256", "probe.deviation")


# ═══════════════════════════════════════════ 1  ЦЕЛОСТНОСТЬ
def s1_integrity(out: Path):
    hr("1. DATA INTEGRITY — READ THIS FIRST")
    P("  Several numbers in this project turned out to come from joins that silently failed. Below: field")
    P("  completeness per file, so the reader sees the gaps before the conclusions.")
    sub("field completeness in *_records.jsonl")
    rows = []
    for fn in ("prompt_records.jsonl", "new_prompt_records.jsonl"):
        p = out / fn
        if not p.exists():
            rows.append([fn, "FILE MISSING", "", "", "", ""])
            continue
        rr = [json.loads(l) for l in open(p, encoding="utf-8")]
        def cnt(f):
            return sum(1 for x in rr if x.get(f) not in (None, "", [], 0))
        rows.append([fn, str(len(rr)), f"{cnt('messages')}", f"{cnt('input_ids')}",
                     f"{cnt('user_span')}", f"{cnt('n_prompt_tokens')}"])
    table(rows, ["file", "rows", "messages", "input_ids", "user_span", "n_prompt_tokens"])
    P("  ! if input_ids is not filled for every record, the token-level analysis (B9) covers only part of them")
    sub("column completeness in records_full.csv")
    if R:
        rows = []
        for c in R[0]:
            k = sum(1 for r in R if r.get(c) not in (None, "", "None"))
            if k < len(R):
                rows.append([c, f"{k}/{len(R)}", f"{100*k/len(R):.0f}%"])
        table(rows or [["—", "all filled", ""]], ["column", "filled", "%"])
    sub("coverage by cell: how many records have a last_user token")
    if R:
        by = defaultdict(lambda: [0, 0])
        for r in R:
            by[r["cell"]][0] += 1
            if r.get("last_user_token_id") not in (None, "", "None"):
                by[r["cell"]][1] += 1
        table([[c, str(v[0]), str(v[1]), "<- no tokens" if v[1] == 0 else ""]
               for c, v in sorted(by.items())], ["cell", "records", "with token", ""])
    sub("model answers: coverage")
    if R:
        for q in ("ask_main", "ask_self", "ask_inverted"):
            c = Counter(r.get(q) or "—" for r in R)
            P(f"  {q:14} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))


# ═══════════════════════════════════════════ 2  СВЕРКА
def s2_crosscheck():
    hr("2. RECONCILIATION: numbers.json AGGREGATES vs RECOMPUTATION FROM records_full.csv")
    if not R:
        P("  records_full.csv missing — reconciliation impossible")
        return
    e1 = [fnum(r, "s_last_dom") for r in R if r["cell"] == "E1"]
    dep = [fnum(r, "s_last_dom") for r in R if r["cell"] == "deployment" and r["is_train_contrast"] == "1"]
    e1 = [x for x in e1 if x is not None]
    dep = [x for x in dep if x is not None]
    if not e1 or not dep:
        P("  E1/deployment not found in records_full.csv")
        return
    sh = sum(e1) / len(e1) - sum(dep) / len(dep)
    P(f"  recomputed E1 shift: {sh:.4f} | in numbers.json: "
      f"{fm(g('probe.exploratory.p2.cells_last.e1_shift'), 4)}")
    rows, bad = [], 0
    for c in CELLS:
        v = [fnum(r, "s_last_dom") for r in R if r["cell"] == c]
        v = [x for x in v if x is not None]
        if not v:
            continue
        fr = (sum(v) / len(v) - sum(dep) / len(dep)) / sh
        fired = sum(1 for r in R if r["cell"] == c and r.get("fired_dom_tau_insample") == "1")
        n_j = g(f"probe.exploratory.p2.cells_last.{c}.n")
        fr_j = g(f"probe.exploratory.p2.cells_last.{c}.frac_of_e1")
        d_n = "" if n_j == len(v) else f"◄ n {n_j} vs {len(v)}"
        d_f = "" if (fr_j is not None and abs(fr_j - fr) < 0.02) else f"◄ frac {fm(fr_j,2)} vs {fr:.2f}"
        if d_n or d_f:
            bad += 1
        rows.append([c, str(len(v)), f"{fr:.3f}", fm(fr_j, 3), f"{fired}/{len(v)}", (d_n + " " + d_f).strip()])
    table(rows, ["cell", "n (csv)", "frac (recomputed)", "frac (json)", "fired", "mismatch"])
    P(f"  cells with a mismatch: {bad} of {len(rows)}" +
      ("  <- EVERYTHING RECONCILES" if bad == 0 else "  <- INVESTIGATE"))


# ═══════════════════════════════════════════ 3  ЯЧЕЙКИ
def s3_cells():
    hr("3. MAIN TABLE: CELLS (last, L21, direction = diff-of-means)")
    Pp = "probe.exploratory.p2.cells_last."
    e1s = g(Pp + "e1_shift")
    P(f"  E1 shift {fm(e1s)} | τ in-sample {fm(g(Pp+'tau_insample'))} | TPR LOO {fm(g(Pp+'tpr_loo'))}")
    rows = []
    for c in CELLS:
        n_ = g(Pp + c + ".n")
        if not n_:
            continue
        fpr = g(Pp + c + ".fpr_loo")
        ci = g(Pp + c + ".shift_ci95_boot")
        rows.append([c, ROLE.get(c, ""), str(n_), fm(g(Pp + c + ".frac_of_e1"), 2),
                     f"[{ci[0]/e1s:.2f}; {ci[1]/e1s:.2f}]" if (ci and e1s) else "—",
                     fm(fpr, 3), f"{g(Pp+c+'.fp_at_tau')}/{n_}", fm(g(Pp + c + ".fpr_wilson95"), 2),
                     "n/a" if fpr == 0 else fm(g(Pp + c + ".precision_at_1pct_loo"), 3),
                     fm(g(Pp + c + ".auc_vs_deploy"), 2)])
    table(rows, ["cell", "role", "n", "frac", "CI(frac)", "FPR@τ", "FP", "Wilson", "prec@1%", "AUC"])
    P("  READ: prec@1% = n/a when FPR=0 — the 1.0 there is mechanical. At n<10 read the Wilson interval only.")
    sub("firing prompts")
    for c in CELLS:
        if g(Pp + c + ".fired_ids"):
            P(f"  {c:12} {g(Pp+c+'.fired_ids')}")
    sub("score distribution across all conditions (from records_full.csv)")
    if R:
        by = defaultdict(list)
        for r in R:
            v = fnum(r, "s_last_dom")
            if v is not None:
                by[r["cell"]].append(v)
        rows = []
        for c, v in sorted(by.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            fired = sum(1 for r in R if r["cell"] == c and r.get("fired_dom_tau_insample") == "1")
            rows.append([c, str(len(v)), f"{sum(v)/len(v):+.2f}", f"{min(v):+.2f}", f"{max(v):+.2f}",
                         f"{fired}/{len(v)}"])
        table(rows, ["condition", "n", "mean", "min", "max", "fired"])
    mark(Pp)


# ═══════════════════════════════════════════ 4-8
def s4_positions():
    hr("4. POSITIONS, LAYERS, NORM")
    Pp = "probe.exploratory.p2.positions."
    table([[p, fm(g(Pp + p + ".auc_e1_vs_deploy"), 3), fm(g(Pp + p + ".tpr_loo"), 3),
            fm(g(Pp + p + ".fpr_topical_loo"), 3), fm(g(Pp + p + ".precision_at_1pct"), 3),
            fm(g(Pp + p + ".topical_frac_of_e1"), 2), fm(g(Pp + p + ".e1_shift"), 1),
            fm(g(Pp + p + ".w_norm"), 2)] for p in ("last", "last_user", "mean_user")],
          ["position", "AUC", "TPR", "FPR_top", "prec@1%", "top_frac", "E1 shift", "‖w‖"])
    P("  ! top_frac for mean_user (18.6) has a degenerate denominator — do NOT interpret")
    sub("KEY: four insertion wordings at three placements (the e2 conditions) read at last_user")
    D = "probe.diag_positions.last_user."
    rows = []
    for c in ("E1", "e2_a", "e2_b", "e2_c", "S1", "S2", "Dstr"):
        if D + c in N:
            rows.append([c, fm(g(D + c), 2)])
    table(rows, ["condition", "response at last_user"])
    P("  E1, e2_a, e2_b and e2_c are FOUR DIFFERENT sentences at three placements — wording and placement are confounded.")
    P("  e2_c ends the user turn (same last_user token as E1) -> 78 % of the E1 response.")
    P("  e2_a and e2_b do not change the last token -> response is 40-60x smaller.")
    P("  => the \"purity\" of the last_user position (FPR_topical 0, precision 1.0) is an ARTEFACT of a shared token.")
    sub("layers")
    Ls = "probe.exploratory.p2.layers."
    table([[k, fm(g(Ls + k + ".auc_e1_vs_deploy"), 3), fm(g(Ls + k + ".topical_frac_of_e1"), 3),
            fm(g(Ls + k + ".fpr_topical_loo"), 3)] for k in ("L19", "L21", "L23")],
          ["layer", "AUC", "contamination", "FPR_top"])
    P("  READ: AUC is constant where contamination changes by 40 % -> AUC is not a layer-selection criterion")
    sub("activation norm vs prompt length (Gate 0, 320 records of 700-760 tokens)")
    table([[p, fm(g(f"gate0.check5.{p}.L21.pearson_r"), 3), fm(g(f"gate0.check5.{p}.L21.norm_mean"), 1)]
           for p in ("last", "last_user", "mean_user")], ["position (L21)", "r(norm, length)", "‖·‖"])
    P(f"  norm ratio topical/deployment: {fm(g('probe.exploratory.p2.Alt1_cosine.norm_topical_over_deploy'))}")
    mark(Pp, Ls, "gate0.check5.", "probe.diag_positions.")


def s5_bands():
    hr("5. PERMUTATION BANDS — ALL 15 CELLS")
    rows = []
    for src, Pp in (("P2", "probe.exploratory.p2.bands."), ("P4", "probe.exploratory.p4.b2_bands.")):
        for c in CELLS:
            k = Pp + c + ".perm_band."
            if k + "perm_p" not in N:
                continue
            sh, q, pp = g(k + "probe_shift"), g(k + "perm_abs_q95"), g(k + "perm_p")
            rows.append([c, src, fm(sh, 2), fm(q, 2), fm(sh / q, 2) if q else "—", fm(pp, 3),
                         "NOT distinguishable" if (pp or 0) > 0.05 else "distinguishable"])
    table(rows, ["cell", "where", "shift", "perm q95", "shift/q95", "perm_p", "verdict"])
    n_all = len(rows)
    n_weak = sum(1 for r in rows if r[-1] == "NOT distinguishable")
    P(f"  RESULT: {n_weak} of {n_all} cells are NOT distinguishable from a direction fitted on shuffled labels.")
    P("  q95 is the 95th percentile of the NULL distribution. Reading it as")
    P("  \"the shuffled ones give more\" is NOT valid: at perm_p~0.3 that is a tautology.")
    P("  above_band for random_band speaks only to MAGNITUDE (a random axis in d=4096 catches ~3 % of the norm).")
    mark("probe.exploratory.p2.bands.", "probe.exploratory.p4.b2_bands.", "probe.band.")


def s6_fpr():
    hr("6. THREE FPR PROCEDURES, OOS, CROSS-ENVIRONMENT")
    rows = []
    for c in CELLS:
        v = [g(f"probe.exploratory.p2.cells_last.{c}.fpr_loo"),
             g(f"probe.exploratory.p2.oos.fpr_{c}_at_tau_oos"),
             g(f"probe.exploratory.p2.oos.fully_oos_fpr.{c}"),
             g(f"probe.exploratory.p3.fully_oos_fpr_all_cells.{c}")]
        if all(x is None for x in v):
            continue
        rows.append([c] + [fm(x, 3) for x in v] +
                    ["<- absent from the leading procedure" if v[2] is None and v[3] is None else ""])
    table(rows, ["cell", "(1) in-sample", "(2) mixed", "(3) fully OOS", "(3) P3", "note"])
    P("  LEADING procedure is (3). The three are not comparable; never mix them in one paragraph.")
    O = "probe.exploratory.p2.oos."
    sub("OOS leave-one-DP-out")
    for k in ("n", "tau_oos", "tpr_oos", "fpr_deploy_oos", "auc_oos"):
        P(f"  {k:20} {fm(g(O+k))}")
    P("  NB: tpr_oos ~ 0.95 BY CONSTRUCTION (tau = 5th pct of the same scores) — not a result")
    sub("cross-environment")
    table([[t, fm(g(O + f"cross_env.{t}.auc_other"), 3), fm(g(O + f"cross_env.{t}.fpr_deploy_other"), 3),
            fm(g(O + f"cross_env.{t}.tpr_e1_other"), 3)] for t in ("train_sandbagging", "train_secret_number")],
          ["trained on", "AUC on the other", "FPR_dep", "TPR_E1"])
    P("  TPR=0 -> the ordering transfers, the THRESHOLD does not. frac at TPR=0 is degenerate.")
    sub("LOIO — held-out insertion wording")
    A = "probe.exploratory.p2.a1_loio."
    for k in sorted({x[len(A):].split(".")[0] for x in N if x.startswith(A)}):
        if A + k + ".auc_e1_vs_deploy" in N:
            P(f"  {k:26} AUC {fm(g(A+k+'.auc_e1_vs_deploy'),3)} | "
              f"AUC holdout {fm(g(A+k+'.auc_holdout_vs_deploy'),3)} | FPR_topic {fm(g(A+k+'.fpr_topic_loo'),3)}")
    P("  ! topic_frac_of_e1 in this block is degenerate (up to 49.4) — do NOT quote")
    mark(O, A, "probe.exploratory.p3.", "probe.exploratory.cross_env.")


def s7_precision():
    hr("7. PRECISION AND THE ORIGIN OF THE NEGATIVES")
    Pp = "probe.exploratory.p2.precision_surface."
    for k in ("tpr_loo", "fpr_topical", "n_topical", "fpr_neutral", "n_neutral", "fpr_neutral_wilson95",
              "fpr_topical_diag_headtail", "fpr_structural_dptrunc", "precision_at_1pct_q0",
              "precision_at_1pct_q1", "neutral_source"):
        if Pp + k in N:
            P(f"  {k:30} {fm(g(Pp+k))}")
    sub("answer to \"you wrote the negatives yourself\"")
    B = "probe.exploratory.p4.b6_precision_dolly."
    rows = []
    for grp, lab in (("legacy_prereg", "the original 8, BEFORE the result"),
                     ("authored_posthoc", "written AFTER the result"),
                     ("external_dolly", "external, sha/seed recorded before selection"),
                     ("neutral_tn", "neutral, authored")):
        if B + grp + ".n" in N:
            pr = g(B + grp + ".precision_at_1pct")
            rows.append([grp, lab, str(g(B + grp + ".n")), fm(g(B + grp + ".fpr_loo"), 3),
                         fm(g(B + grp + ".fpr_insample_wilson95"), 2),
                         "n/a (FPR=0)" if pr is None else fm(pr, 3)])
    table(rows, ["group", "origin", "n", "FPR", "Wilson", "prec@1%"])
    P("  READ: negatives authored post-hoc give an FPR NO LOWER than the pre-registered ones ->")
    P("  writing negatives after knowing the result did not make the picture more convenient.")
    mark(Pp, B)


def s8_contrasts():
    hr("8. CONTRASTS AND FORKS")
    pp = "probe.exploratory.p2.primary_pairs."
    b1 = "probe.exploratory.p4.b1_story."
    rows = [["a1-a2 (lexicon, P2)", str(g(pp + "n_pairs")), fm(g(pp + "mean_diff_frac_of_e1"), 3),
             fm(g(pp + "frac_ci95_boot"), 3), fm(g(pp + "p_sign_perm_exact"), 4),
             fm(g(pp + "min_achievable_p"), 4), ""]]
    for nm, key in (("a1-a_head (request type, P4)", "a1_minus_a_head"), ("a1-a2 (re-check, P4)", "a1_minus_a2")):
        if b1 + key + ".n" in N:
            rows.append([nm, str(g(b1 + key + ".n")), fm(g(b1 + key + ".mean_frac_of_e1"), 3),
                         fm(g(b1 + key + ".cluster_ci95_by_story"), 3), fm(g(b1 + key + ".p_sign_perm_exact"), 4),
                         fm(g(b1 + key + ".min_achievable_p"), 4),
                         f"{g(b1+key+'.same_sign')}/{g(b1+key+'.n')} of one sign"])
    table(rows, ["contrast", "n", "frac", "CI", "p", "p floor", "signs"])
    P("  The same 8 stories. Removing the ENTIRE registered lexicon moves it by 0.02;")
    P("  changing the request type with the lexicon kept moves it by 0.21. An order of magnitude apart.")
    st = g(b1 + "per_story_frac")
    if isinstance(st, dict):
        sub("by story")
        table([[v.get("story", k), fm(v.get("a1"), 2), fm(v.get("a2"), 2), fm(v.get("a_head"), 2)]
               for k, v in sorted(st.items()) if isinstance(v, dict)], ["story", "a1", "a2", "a_head"])
    sub("secondary")
    for b, nm in (("probe.exploratory.p2.secondary.jt_trend.", "JT trend"),
                  ("probe.exploratory.p2.secondary.b_vs_c.", "b−c"),
                  ("probe.exploratory.p2.secondary.d_vs_c.", "d−c")):
        ks = {k[len(b):]: v for k, v in N.items() if k.startswith(b) and not k.endswith("note")}
        if ks:
            P(f"  {nm:10} " + " · ".join(f"{k}={fm(v)}" for k, v in sorted(ks.items())))
    P("  NB: JT is one-sided; at p=0.923 the reverse order would give ~0.08 — state the direction.")
    P("  b = 8/8 \"write a document\", c = 8/8 \"analyse these numbers\" -> task type is confounded with the cell.")
    sub("head/tail and scenario pairs")
    for b in ("probe.exploratory.p2.head_vs_tail.", "probe.exploratory.p2.scenario_pairs.",
              "probe.exploratory.p2.story_cluster_ci."):
        for k, v in sorted({k[len(b):]: v for k, v in N.items() if k.startswith(b)}.items()):
            if not k.endswith(("note", "reading")) and not k.startswith("pairs.i"):
                P(f"  {b.split('.')[-2]:18} {k:38} {fm(v)}")
    P("  ! a_tail: 8/8 share one request template and the prompts are two-topic (\"Separately\") -> effective n ~ 1;")
    P("  p_perm on 8x8 overstates by orders of magnitude. Reported as a diagnostic, not a test.")
    sub("confirmatory S-Dstr")
    rows = []
    for env in ("sandbagging", "secret_number"):
        for c in ("S1", "S2", "Dstr", "E1_S1"):
            b = f"probe.exploratory.p2.confirmatory.{env}.{c}."
            if b + "mean" in N:
                pv = g(b + "p_sign_perm")
                rows.append([env, c, fm(g(b + "mean"), 2), fm(g(b + "frac_of_e1"), 3), fm(pv, 6),
                             "FLOOR" if (pv and abs(pv - 2 ** -15) < 1e-9) else ""])
    table(rows, ["environment", "condition", "mean", "frac", "p", ""])
    P("  At n=16 the exact-test floor is 2/2^16 = 3.05e-5. Every \"significant\" p equals the floor ->")
    P("  write \"16/16 deltas of one sign, p at the test floor\", not as an exact value.")
    P(f"  interaction E1xS: mean {fm(g('probe.interaction.pool.mean'))}, "
      f"sd {fm(g('probe.interaction.pool.sd'))}, n {g('probe.interaction.pool.n')} -> subadditive")
    mark(pp, b1, "probe.exploratory.p2.secondary.", "probe.exploratory.p2.head_vs_tail.",
         "probe.exploratory.p2.scenario_pairs.", "probe.exploratory.p2.story_cluster_ci.",
         "probe.exploratory.p2.confirmatory.", "probe.S_minus_Dstr.", "probe.interaction.", "probe.delta.")


def s9_baselines():
    hr("9. BASELINES")
    sub("\"ask the model\" — three questions")
    B = "baseline.ask_model."
    grps = sorted({k.split(".")[3] for k in N if k.startswith(B + "main.") and k.count(".") >= 4})
    rows = []
    for grp in grps:
        cells = []
        for q in ("main", "self", "inverted"):
            b = f"{B}{q}.{grp}."
            yr = g(b + "yes_rate")
            cells.append(f"{fm(yr,2)} ({g(b+'yes')}/{g(b+'n_parsable')})" if yr is not None else "—")
        rows.append([grp] + cells + [fm(g(f"{B}main.{grp}.wilson95"), 2)])
    table(rows, ["group", "main: is this an evaluation?", "self: are YOU being evaluated?", "inverted: is this deployment?", "Wilson(main)"])
    P("  E1 and deployment answer identically on the topical and inverted questions (32/32 vs 31/32; 0/32 vs 0/32) — a CEILING, meaning they cannot be distinguished, not that there is no difference. On the self-referential question they DO differ: 20/32 against 16/32, and all four extra yes answers are in sandbagging (Fisher p = 0.051).")
    P("  This is a CEILING: it means \"cannot be distinguished\", not \"there is no difference\".")
    P("  The main question admits a TOPICAL reading — the deployment prompt literally describes an evaluation procedure.")
    for k in ("calibration.accuracy", "calibration.n", "calibration.yes_bias_rate", "calibration.limitation",
              "mode.prefill", "mode.cap", "temperature", "n_targets"):
        if B + k in N:
            P(f"  {k:28} {fm(g(B+k))}")
    P("  ! all 8 calibration questions are surface facts about the first user turn on 8 sandbagging dialogues")
    P("  (German? recipe? gardening? phone? poem? digits? English? the word \"the\"?).")
    P("  This is NOT calibration on the meta-question; the transfer is unjustified.")
    sub("score vs the explicit answer (stratified by cell)")
    I = "probe.exploratory.p4.b7_item_level."
    table([[q, fm(g(I + q + ".auc_within_cell"), 3), str(g(I + q + ".n_pairs_within")),
            str(g(I + q + ".n_cells_with_variation")), fm(g(I + q + ".auc_naive_pooled_DO_NOT_USE"), 3)]
           for q in ("main", "self", "inverted") if I + q + ".n_pairs_within" in N],
          ["question", "within-cell AUC", "pairs", "cells", "naive pooled (DO NOT USE)"])
    P("  The naive pooled ~0.5 is set by cell composition: 96 % of pairs are between cells.")
    S = "probe.exploratory.p4.b4_selfyes_scores.sandbagging."
    P(f"  scores yes vs no on self-reference (sandbagging E1): AUC {fm(g(S+'auc_yes_vs_no'),3)}, "
      f"yes {g(S+'n_yes')} / no {g(S+'n_no')}, Fisher {fm(g(S+'e1_yes_vs_deployment_yes.p_fisher_one_sided'),4)}")
    sub("self by environment: does truncation change the answer")
    E = "probe.exploratory.p4.b5_self_by_env."
    rows = []
    for c in sorted({k.split(".")[0] for k in N if k.startswith(E) and k.count(".") >= 2}):
        for s_ in sorted({k.split(".")[1] for k in N if k.startswith(E + c + ".") and k.count(".") >= 3}):
            b = f"{E}{c}.{s_}."
            if b + "yes_rate" in N:
                rows.append([c, s_, str(g(b + "yes")), str(g(b + "no")), fm(g(b + "yes_rate"), 2)])
    table(rows, ["cell", "environment", "yes", "no", "yes rate"])
    P("  len140/len210: halving the prompt does NOT change the self-reference answer — it is set by the environment.")
    sub("lexical (grep)")
    G = "probe.exploratory.p2.a2_grep."
    rows = []
    for chan in ("user_only", "full_prompt"):
        for lex in ("eval_strict", "eval_broad", "eval_adjacent"):
            b = f"{G}{chan}.{lex}.excl_checklist."
            if b + "auc_e1_vs_deploy" in N:
                rows.append([chan, lex, fm(g(b + "auc_e1_vs_deploy"), 3), fm(g(b + "tau"), 2),
                             fm(g(b + "fpr_a1"), 2), fm(g(b + "fpr_a2"), 2), fm(g(b + "fpr_n"), 2)])
    table(rows, ["channel", "lexicon", "AUC", "τ", "FPR a1", "a2", "n"])
    P("  ! these cells have exactly one turn -> texts_full == texts_user, the channel axis is DEGENERATE.")
    P("  ! eval_adjacent gives AUC 0.375 (<0.5): under the adjacent lexicon deployment is DENSER than E1.")
    P("  This is direct lexical evidence that the negative class is itself evaluative.")
    mark(B, I, S, E, G)


def s10_densities():
    hr("10. LEXICAL DENSITY, LENGTH, NLL, STRUCTURE")
    D = "probe.exploratory.p2.cell_densities."
    V = "probe.exploratory.p2.objections.V3_length_nll."
    rows = []
    for c in CELLS:
        if D + c + ".n" not in N:
            continue
        rows.append([c, str(g(D + c + ".n")), fm(g(D + c + ".strict_median"), 2),
                     fm(g(D + c + ".ai_narrow_median"), 2), fm(g(D + c + ".ai_broad_median"), 2),
                     fm(g(V + f"len_median_{c}"), 0), fm(g(V + f"spearman_len_{c}"), 2),
                     fm(g(V + f"nll_median_{c}"), 2), fm(g(V + f"spearman_nll_{c}"), 2)])
    table(rows, ["cell", "n", "eval_strict", "ai_narrow", "ai_broad", "words", "rho(score,length)",
                 "NLL", "rho(score,NLL)"])
    P("  ! ai_narrow=0 for c1 is a property of the regex (literals ai/llm/language model), NOT absence of AI:")
    P("  c1 = «300-million-parameter transformer», «speculative decoding», «tokenizer», «16 GPUs».")
    P("  READ: within-cell rho(score, NLL) ranges -0.57 to +0.60 with no pattern -> \"text unfamiliarity\"")
    P("  is not the dominant factor. The between-cell rho=0.36 over 16 dependent points does not prove it.")
    sub("length ladder")
    L = "probe.exploratory.p4.length_substitutes."
    for c in ("len140", "len210"):
        b = L + f"saturation_vs_foreign.ladder.{c}."
        if b + "frac" in N:
            P(f"  {c:10} frac {fm(g(b+'frac'),3)}  words {fm(g(b+'len_median'),0)}")
    b = L + "saturation_vs_foreign.structural_reference.dptrunc."
    if b + "frac" in N:
        P(f"  dptrunc    frac {fm(g(b+'frac'),3)}  <- STRUCTURAL (MCQ format), NOT a rung of the length ladder")
    P(f"  rho(score, length) within Dolly: {fm(g(L+'within_external_length.spearman_score_vs_len'),3)} "
      f"(n={g(L+'within_external_length.n')})")
    P("  NO BOUND: the pooled rungs (0.306, 0.281) differ by 0.025 only because the two environments")
    P("  cancel. Split: sandbagging 0.156 -> 0.434, secret_number 0.456 -> 0.128 from 140 to 210 words.")
    P("  Truncation also removes the varying tail, so len140-sb is 8 identical texts and each sn rung has")
    P("  2 distinct texts out of 8 -> the rungs differ in content as well as length. The ladder therefore")
    P("  bounds nothing. A direct 'foreign text of ~300 words' control was NOT run — a limitation.")
    sub("structure of the conditions (pre-registered, lengths matched)")
    table([[c, fm(g(f"prereg.density.{c}.n_prompt_tokens"), 1), fm(g(f"prereg.density.{c}.headers"), 2),
            fm(g(f"prereg.density.{c}.colon_lines"), 2), fm(g(f"prereg.density.{c}.blank_lines"), 2),
            fm(g(f"prereg.density.{c}.numbering"), 2),
            fm(g(f"prereg.density_similarity.{c}.cos_mean_user_vs_deployment_mean"), 5)]
           for c in ("deployment", "E1", "E1_S1", "S1", "S2", "Dstr") if f"prereg.density.{c}.headers" in N],
          ["condition", "tokens", "headers", "colon", "blank", "number", "cos to deployment"])
    P("  S1 +0.13 / S2 -0.01 / Dstr -0.13 E1 at matched lengths -> the read at last is LOCAL.")
    mark(D, V, L, "prereg.density")


def s11_p4p5():
    hr("11. ALTERNATIVE DIRECTION CONSTRUCTIONS")
    Pp = "probe.exploratory.p5.profiles."
    ms = sorted({k[len(Pp):].split(".")[0] for k in N if k.startswith(Pp)})
    rows = []
    for m in ms:
        b = Pp + m + "."
        if b + "auc_e1_vs_deploy" not in N:
            continue
        pr = g(b + "precision_at_1pct.q_topic_1")
        rows.append([m, fm(g(b + "cos_with_diff_of_means"), 3), fm(g(b + "auc_e1_vs_deploy"), 3),
                     fm(g(b + "tpr_loo"), 3),
                     f"{fm(g(b+'by_group.topical_leading.fpr_loo'),3)} "
                     f"({g(b+'by_group.topical_leading.fp_insample')}/{g(b+'by_group.topical_leading.n')})",
                     fm(g(b + "by_group.topical_leading.wilson95_insample"), 2),
                     f"{fm(g(b+'by_group.wild_other_only.fpr_loo'),3)} "
                     f"({g(b+'by_group.wild_other_only.fp_insample')}/{g(b+'by_group.wild_other_only.n')})",
                     fm(g(b + "by_group.wild_other_only.wilson95_insample"), 2),
                     "n/a" if pr is None else fm(pr, 3)])
    table(rows, ["direction", "cos", "AUC", "TPR", "FPR topical", "Wilson", "FPR wild", "Wilson", "prec@1%"])
    P("  ! ALL are trained IN-SAMPLE on the 64 contrast records; the headline numbers are OOS. Do not mix.")
    P("  COMPARE VIA WILSON: the intervals on wild overlap -> the reduction is NOT established.")
    sub("direction stability (split-half by prompt)")
    S = "probe.exploratory.p5.direction_stability."
    for m in ms:
        if S + m + ".cos_mean" in N:
            P(f"  {m:16} cos = {fm(g(S+m+'.cos_mean'),3)} ± {fm(g(S+m+'.cos_sd'),3)} "
              f"({g(S+m+'.n_splits')} splits)")
    P(f"  effective rank of the deployment covariance: {fm(g(S+'effective_rank_deployment'),2)} "
      f"-> 'whitening' ~ 'remove PC1' (PC1 holds {fm(g('probe.exploratory.p2.Alt2_anisotropy.pc1_var_frac'))})")
    P("  NB: stability of the DIRECTION is not stability of the FIRING SET (threshold on 37 points).")
    sub("firing sets under the different directions")
    F = "probe.exploratory.p5.fired_sets."
    for m in ms + ["p2_fired_ids"]:
        if F + m + ".wild_fired_ids" in N:
            P(f"  {m:16} {g(F+m+'.wild_fp')}/{g(F+m+'.wild_n')} "
              f"Wilson {fm(g(F+m+'.wild_wilson95'),3)}  {g(F+m+'.wild_fired_ids')}")
        elif F + m in N:
            P(f"  {m:16} {fm(g(F+m))}")
    sub("B8 (same topic, separate run)")
    B = "probe.exploratory.p4.b8_lr_lda."
    table([[m, fm(g(B + m + ".cos_with_diff_of_means"), 3), fm(g(B + m + ".heldout_auc_mean"), 3),
            fm(g(B + m + ".fpr_topical_loo"), 3), fm(g(B + m + ".topical_frac_of_e1"), 2)]
           for m in sorted({k[len(B):].split(".")[0] for k in N if k.startswith(B)})
           if B + m + ".cos_with_diff_of_means" in N],
          ["method", "cos with d-o-m", "held-out AUC", "FPR_top", "top frac"])
    P("  At n=64, d=4096 the L2 methods CONVERGE to the difference of means (cos >= 0.98) — that is mathematics,")
    P("  not a comparison of methods. Correct conclusion: the pre-registered choice was not a limitation.")
    sub("anisotropy")
    A = "probe.exploratory.p2.Alt2_anisotropy."
    for k in ("cos_w_pc1", "pc1_var_frac", "after_proj_out_pc1_2.auc_e1_vs_deploy",
              "after_proj_out_pc1_2.fpr_topical_loo", "after_proj_out_pc1_2.topical_frac_of_e1"):
        if A + k in N:
            P(f"  {k:44} {fm(g(A+k))}")
    P("  ! mu and the PCs are taken from deployment ONLY (n=32), where rho(score,length)=0.78 -> PC1 is poorly")
    P("  separable from the length axis. The FPR rise 0.20->0.428 after projecting out PC1-2 reads rather as")
    P("  \"PC1 was suppressing the contamination\" than as \"anisotropy is irrelevant\".")
    sub("B3 — enrichment of the external set")
    E = "probe.exploratory.p4.b3_dolly_enrichment."
    if g(E + "pending"):
        P("  AWAITING blind coding (results/dolly_labels.json). Observation that needs checking:")
        P("  five of the 6 firing Dolly prompts are \"interpret a person's measured numbers\"")
        P("  (heart rate by test stage, two participants of different experience, 4-month progress,")
        P("  baseline fitness assessment, gym supervisor); the sixth is navigation with 2439 miles.")
        P("  The category was defined AFTER looking at the scores -> do not claim it without blind recoding.")
    else:
        t = E + "table."
        P(f"  labelled {g(E+'n_labelled')} | category {g(t+'cat_fired')}/{g(t+'cat_not_fired')} · "
          f"others {g(t+'other_fired')}/{g(t+'other_not_fired')} | Fisher p = {fm(g(E+'p_fisher_one_sided'),6)}")
        P(f"  {fm(g(E+'caveat'))}")
    mark("probe.exploratory.p5.", "probe.exploratory.p4.", A)


def s12_gates():
    hr("12. MEASUREMENT VALIDITY, GATE 0, PILOT")
    sub("activation-extraction controls")
    for k, lab in (("gate0.check1.right.last.L21.rel_l2_max", "batched vs single (right pad)"),
                   ("gate0.check1.naive_last_right_pad.cos_min", "NAIVE index — negative control"),
                   ("gate0.check2.uniform", "last-token homogeneity"),
                   ("gate0.check2.distinct_last_token_ids", "id of the last token (position last)"),
                   ("gate0.check3.L21.max_abs_diff_vs_hidden_states", "hooks vs hidden_states"),
                   ("gate0.check6.hooks_clean", "hooks removed"),
                   ("gate0.check6.bitwise_repeatable", "bitwise reproducibility"),
                   ("probe.exploratory.g1.anchor.rel_l2_max", "cross-session anchor rel_l2"),
                   ("probe.exploratory.g1.anchor.cos_min", "anchor cos")):
        if k in N:
            P(f"  {lab:44} {fm(g(k))}")
    sub("fp16 (accepted with a recorded deviation)")
    for k in ("kl_last_bf16_vs_fp16.median", "kl_last_bf16_vs_fp16.argmax_top1_agree",
              "overflow.nonfinite_count", "overflow.max_abs_h", "greedy_prefix_single_match",
              "verdict.prefix_single"):
        if "gate0.fp16." + k in N:
            P(f"  {k:44} {fm(g('gate0.fp16.'+k))}")
    sub("pre-registered probe validations")
    for k in ("probe.layer_rule.layer", "probe.layer_rule.tie", "probe.gate_v1_v2_v4", "probe.gate_v5",
              "probe.gate_v3.pool", "probe.v2.pass", "probe.v2.means.E1", "probe.v2.means.deployment",
              "probe.v2.means.positive_control"):
        if k in N:
            P(f"  {k:44} {fm(g(k))}")
    P("  => v2/v4/v5 and v3 NOT passed: the work follows the frozen branch \"the direction did not validate\".")
    sub("pilot and pivot")
    for k in ("pilot.rate.sandbagging.deployment", "pilot.rate.secret_number.deployment",
              "pilot.positive_control.alive", "pilot.positive_control.next", "pilot.decision.fits",
              "pilot.decision.hours", "pilot.decision.k", "pilot.decision.env",
              "pilot.truncated_frac.sandbagging.deployment", "pilot.sanity.keyshuffle_accuracy_mean",
              "pivot.4_2.closed_20260829"):
        if k in N:
            P(f"  {k:48} {fm(g(k))}")
    sub("rescue analysis of truncated generations")
    for pre in ("secondary.rescue.", "probe.exploratory.rescue."):
        for k, v in sorted({k[len(pre):]: v for k, v in N.items() if k.startswith(pre)}.items()):
            P(f"  {k:48} {fm(v)}")
    mark("gate0.", "pilot.", "probe.layer", "probe.gate", "probe.v2", "probe.exploratory.g1.",
         "secondary.rescue.", "probe.exploratory.rescue.")


# ═══════════════════════════════════════════ 13-15
def s13_anomalies():
    hr("13. AUTOMATED ANOMALY SCAN")
    A = []
    def add(s, m):
        A.append((s, m))
    for k, v in N.items():
        if k.endswith(("topical_frac_of_e1", "frac_of_e1", "topic_frac_of_e1")) and isinstance(v, (int, float)):
            if v == v and abs(v) > 3:
                add("!!", f"|frac| = {fm(v,2)} for {k} — degenerate denominator")
    for c in CELLS:
        b = f"probe.exploratory.p2.cells_last.{c}."
        if g(b + "fpr_loo") == 0 and g(b + "precision_at_1pct_loo") == 1.0:
            add("!", f"precision@1% = 1.0 for {c} — a consequence of FPR=0 at n={g(b+'n')}")
        n_ = g(b + "n")
        if n_ is not None and n_ < 5:
            add("!", f"cell {c}: n = {n_}")
    weak = sorted({c for c in CELLS for Pp in ("probe.exploratory.p2.bands.", "probe.exploratory.p4.b2_bands.")
                   if (g(Pp + c + ".perm_band.perm_p") or 0) > 0.05})
    if weak:
        add("!!", f"perm_p > 0.05 for {len(weak)} cells: {weak}")
    for k, v in N.items():
        if k.endswith(("p_sign_perm", "p_sign_perm_exact")) and isinstance(v, float) and v == v:
            for n_ in (4, 8, 16):
                if abs(v - 2 ** (1 - n_)) < 1e-9:
                    add("!", f"{k} = {v:.2e} — FLOOR of the exact test at n={n_}")
    au = [g(f"probe.exploratory.p2.layers.L{L}.auc_e1_vs_deploy") for L in (19, 21, 23)]
    fr = [g(f"probe.exploratory.p2.layers.L{L}.topical_frac_of_e1") for L in (19, 21, 23)]
    if all(x is not None for x in au + fr) and max(au) - min(au) < 0.02:
        add("!!", f"AUC across layers {[fm(x,3) for x in au]} is constant, contamination "
                  f"{[fm(x,3) for x in fr]} changes by {100*(max(fr)-min(fr))/max(fr):.0f}%")
    for t in ("train_sandbagging", "train_secret_number"):
        if g(f"probe.exploratory.p2.oos.cross_env.{t}.tpr_e1_other") == 0:
            add("!!", f"cross-environment {t}: TPR=0 at AUC=1 -> the threshold does not transfer")
    for q in ("main", "self", "inverted"):
        b = f"probe.exploratory.p4.b7_item_level.{q}."
        w_, nn = g(b + "auc_within_cell"), g(b + "auc_naive_pooled_DO_NOT_USE")
        if w_ is not None and nn is not None and w_ == w_ and nn == nn and abs(w_ - nn) > 0.05:
            add("!!", f"B7 {q}: within-cell {fm(w_,3)} vs naive pooled {fm(nn,3)}")
    Pp = "probe.exploratory.p5.profiles."
    ms = sorted({k[len(Pp):].split(".")[0] for k in N if k.startswith(Pp)})
    base = g(Pp + "diff_of_means.by_group.wild_other_only.wilson95_insample")
    for m in ms:
        if m == "diff_of_means":
            continue
        w_ = g(Pp + m + ".by_group.wild_other_only.wilson95_insample")
        if base and w_ and not (w_[1] < base[0] or base[1] < w_[0]):
            add("!!", f"P5 {m}: Wilson on wild {fm(w_,3)} overlaps diff_of_means {fm(base,3)}")
    tf = g("env.transformers")
    for s_ in {v for k, v in N.items() if k.endswith("_source") and isinstance(v, str) and "transformers=" in v}:
        if tf and tf not in s_:
            add("!!", f"env.transformers = {tf}, records refer to {s_} — a recorded deviation is required")
    if g("probe.exploratory.p4.p2_unchanged.ok") is False:
        add("!!", "p2_unchanged.ok = False")
    if "probe.exploratory.p3.tau_median_loo" not in N:
        add("!!", "P3 results are absent from the snapshot: the leading fully_oos_fpr procedure does not cover every cell")
    if g("probe.exploratory.p4.b9_last_tokens.scored_position") != "last_user":
        add("!!", "B9 was computed off the last_user position")
    if R:
        no_tok = sum(1 for r in R if r.get("last_user_token_id") in (None, "", "None"))
        if no_tok:
            add("!!", f"last_user_token_id is missing for {no_tok} of {len(R)} records — "
                      f"the token-level analysis does not cover everything")
    if g("probe.exploratory.p2.a2_grep.user_only.eval_adjacent.excl_checklist.auc_e1_vs_deploy", 1) < 0.5:
        add("i", "eval_adjacent AUC < 0.5: under the adjacent lexicon deployment is denser than E1 — "
                 "lexical evidence that the negative class is itself evaluative")
    if g("prereg.density_similarity.note") == "probe_acts not found":
        add("!", "prereg.density_similarity was computed without activations")
    if g("gate0.fp16.verdict.prefix_single") is False:
        add("i", "fp16 accepted despite the prefix_single failure — deviation recorded")
    if g("probe.exploratory.p2.multiplicity.correction") == "none":
        add("i", f"no multiplicity correction was applied across "
                 f"{g('probe.exploratory.p2.multiplicity.declared_forks')} forks")
    if g("probe.gate_v1_v2_v4") is False or g("probe.gate_v5") is False:
        add("i", "pre-registered validations v2/v4/v5 not passed — branch \"the direction did not validate\"")
    o = {"!!": 0, "!": 1, "i": 2}
    for s_, m in sorted(A, key=lambda x: o.get(x[0], 3)):
        P(f"  {s_:2} {m}")
    P(f"\n  critical {sum(1 for s_,_ in A if s_=='!!')} · warnings "
      f"{sum(1 for s_,_ in A if s_=='!')} · notes {sum(1 for s_,_ in A if s_=='i')}")


def s14_invalid():
    hr("14. NUMBERS DECLARED INVALID — DO NOT USE")
    items = [
        ("probe.exploratory.p4.b9_last_tokens.token_only_predictor.*",
         "FPR_topical=0 was obtained on records without input_ids (the 137 new records have input_ids=None); "
         "the last_user artefact conclusion rests on probe.diag_positions (e2_a/e2_b/e2_c), not on B9"),
        ("probe.exploratory.p4.b9_last_tokens.global_eta2_*",
         "covers 384 of the 529 pre-registered records; the topical cells are not included"),
        ("probe.exploratory.p2.a1_loio.*.topic_frac_of_e1",
         "degenerate denominator (up to 49.4) at AUC 0.5; the meaningful number there is the AUC"),
        ("probe.exploratory.cross_env.*.topic_frac_of_e1_in_*",
         "TPR_E1=0 in both directions -> denominator ~0; the sign flip (-2.667/+4.862) is an artefact"),
        ("probe.exploratory.p2.positions.mean_user.topical_frac_of_e1",
         "18.6 at ||w||=0.51 and AUC 0.75 — degenerate"),
        ("*.precision_at_1pct when FPR=0",
         "the 1.0 is mechanical; report the rate with a Wilson interval"),
        ("confirmatory p_sign_perm = 3.05e-05",
         "the exact-test floor at n=16; write \"16/16 deltas of one sign\""),
        ("a2_grep: the \"channel\" axis (user_only vs full_prompt)",
         "every cell has a single turn -> the rows are twins by construction"),
        ("P5/B8: every FPR and frac of the alternative directions",
         "trained in-sample on 64 records; the headline numbers are OOS — not comparable"),
    ]
    for k, why in items:
        P(f"  ✗ {k}")
        P(f"      {why}")


def s15_rest():
    hr("15. KEYS NOT COVERED BY ANY SECTION")
    rest = sorted(k for k in N if k not in SEEN)
    P(f"  total {len(N)}, shown {len(SEEN)}, here {len(rest)}")
    grp = defaultdict(list)
    for k in rest:
        parts = k.split(".")
        grp[".".join(parts[:3]) if len(parts) > 3 else ".".join(parts[:2])].append(k)
    for pre, ks in sorted(grp.items(), key=lambda x: -len(x[1])):
        P(f"\n  ── {pre}* ({len(ks)})")
        for k in ks[:30]:
            P(f"    {k:74} {fm(N[k])}")
        if len(ks) > 30:
            P(f"    … and {len(ks)-30}")


def main(argv=None):
    global N, R
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--md", default=None, help="output path (default results/FINAL_DUMP.md)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    N = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    p = out / "records_full.csv"
    R = list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []

    P("═" * W)
    P(f"  FINAL DUMP FOR INDEPENDENT ANALYSIS  |  {out.resolve()}")
    P("═" * W)
    for fn in (lambda: s0(out), lambda: s1_integrity(out), s2_crosscheck, s3_cells, s4_positions,
               s5_bands, s6_fpr, s7_precision, s8_contrasts, s9_baselines, s10_densities,
               s11_p4p5, s12_gates, s13_anomalies, s14_invalid, s15_rest):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            P(f"\n  !! section failed: {type(e).__name__}: {e}")
    hr("END")

    md = Path(a.md) if a.md else out / "FINAL_DUMP.md"
    md.write_text("```\n" + "\n".join(OUT_LINES) + "\n```\n", encoding="utf-8")
    print(f"\n>>> written: {md}  ({md.stat().st_size/1024:.0f} KB) — this is the file to hand over")


if __name__ == "__main__":
    # при вставке скрипта прямо в ячейку ноутбука argv содержит -f <kernel.json> от Jupyter,
    # и argparse паyesет; в этом случае разбираем пустой argv
    main([] if "__file__" not in globals() else None)
