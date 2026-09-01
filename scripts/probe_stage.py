"""Этап 5 — плечо пробе (план §4.7, addendum §9–10). Только prefill в fp16 (dtype генерации), ~15 мин на T4.
    PYTHONPATH=. python scripts/probe_stage.py --dps data/decision_points.json --topic data/topic_prompts.json --out results
Порядок ЖЁСТКИЙ — S-скоры не считаются, пока валидации 1–4 и слой не записаны в numbers (probe.validations_recorded_at).
2026-08-28 (ревью R-Pipeline А.1): два запуска — `--stage validate` (A–C, до пилота) и `--stage main` (D–E, только после
fill_gate_params: prereg.config_change_log непуст; иначе --force-main с записью отклонения). Ворота: gate_v1_v2_v4, gate_v5
(= e2_a/e2_b, ДРУГАЯ позиция; e2_c — репорт локальности), gate_v3.{scope} — после D.
  A. активации last (+ last_user/mean_user как диагностика) на L∈candidate_layers для всех версий 32 DP + topic (deployment-шаблон)
  B. правило слоя: кросс-средовой held-out AUC (deployment vs E1); тай → середина — БЕЗ E2 и S
  C. валидации: 1 held-out AUC (пул + по средам); 2 монотонность PC > E1 > deployment; 3 полосы (random — репорт, perm — ворота, машинерия);
     4 topic: сдвиг topic vs deployment (доля E1-сдвига, perm p; unpaired); 5 E2 по вставкам (e2_a/e2_b/e2_c) + sys_neutral (D3)
  D. ГЛАВНЫЙ ВОПРОС: проекции 6 условий × 32 промпта; парные дельты vs deployment; парная перестановка S1−Dstr, S2−Dstr, S−deploy;
     perm-полоса и random-полоса; знак = знак E1; пул и по средам. Таблица probe_scores.csv (192 строк = единица джойна).
  E. фигура: ось X = 6 условий, Y = нормированный скор (E1 − deployment = 1), две панели (среды), точки = промпты.
Активации → results/probe_acts_fp16.pt (dtype в payload; ассерт на равенство cfg.model.dtype/pilot.sampling.dtype)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gate0_kaggle import load_model, release
from src.activations import extract
from src.conditions import MAIN_IDS, S_IDS, load_conditions
from src.contracts import assert_matches_prereg, dps_sha256, make_records, record_numbers
from src.probe_toy import auc, choose_layer_cross_env, diff_of_means, heldout_by_prompt, perm_band, random_band, shift_frac_and_sign
from src.prompts import load_tokenizer
from src.scoring import load_analysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("probe")
POSITIONS = ("last", "last_user", "mean_user")


def paired_perm_test(d: torch.Tensor, n_perm: int = 5000, seed: int = 0) -> dict[str, float]:
    """Парная перестановка: знак каждой дельты случаен под H0. Возвращает mean, p (двусторонний), CI по бутстрапу знака не нужен."""
    g = torch.Generator().manual_seed(seed)
    obs = d.mean().abs().item()
    signs = (torch.randint(0, 2, (n_perm, len(d)), generator=g) * 2 - 1).float()
    null = (signs * d[None, :]).mean(1).abs()
    return {"mean": d.mean().item(), "sd": d.std().item() if len(d) > 1 else float("nan"), "n": len(d),
            "p": ((null >= obs).sum().item() + 1) / (n_perm + 1)}


def unpaired_perm_test(a: torch.Tensor, b: torch.Tensor, n_perm: int = 5000, seed: int = 0) -> dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    obs = (a.mean() - b.mean()).abs().item(); z = torch.cat([a, b]); na = len(a)
    null = []
    for _ in range(n_perm):
        p = z[torch.randperm(len(z), generator=g)]
        null.append((p[:na].mean() - p[na:].mean()).abs().item())
    return {"diff": (a.mean() - b.mean()).item(), "p": (sum(x >= obs for x in null) + 1) / (n_perm + 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--analysis", default="configs/analysis.yaml")
    ap.add_argument("--dps", required=True)
    ap.add_argument("--topic", default="data/topic_prompts.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dtype", default=None, help="default: pilot.sampling.dtype / gate0.budget.dtype")
    ap.add_argument("--layers", default=None, help="override candidate layers, e.g. 19,21,23")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--skip-extract", action="store_true", help="использовать results/probe_acts_fp16.pt")
    ap.add_argument("--stage", default="all", choices=["validate", "main", "all"])
    ap.add_argument("--force-main", action="store_true", help="D–E без config_change_log (записывается как отклонение)")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.stage == "main":
        a.skip_extract = True
    out = Path(a.out); nums = out / "numbers.json"
    cfg = load_conditions(a.config, allow_unset_gate_params=True)
    A = load_analysis(a.analysis); thr = cfg.thresholds
    dsha = dps_sha256(a.dps); assert_matches_prereg(cfg, nums, dsha)
    prev = json.loads(nums.read_text(encoding="utf-8"))
    if a.stage in ("main", "all") and not prev.get("prereg.config_change_log"):
        assert a.force_main, "главный вопрос (D) — только после fill_gate_params/verify (prereg.config_change_log); иначе --force-main"
        record_numbers(nums, {"probe.deviation.main_before_gate1": True})
    dtype = a.dtype or prev.get("pilot.sampling.dtype") or ("float16" if prev.get("gate0.budget.dtype") == "fp16" else None)
    assert dtype, "dtype не определён: нужен pilot.sampling.dtype или --dtype"
    layers = [int(x) for x in a.layers.split(",")] if a.layers else (cfg.probe.get("candidate_layers") or prev.get("gate0.candidate_layers") or [19, 21, 23])
    topic_raw = json.loads(Path(a.topic).read_text(encoding="utf-8"))
    topic_sha = hashlib.sha256(json.dumps(topic_raw["prompts"], sort_keys=True).encode()).hexdigest()[:16]

    tok = load_tokenizer(cfg)
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8"))
    recs = make_records(dps, cfg.conditions, tok, cfg, dps_sha=dsha)                        # 32 × 11
    dep_cond = cfg.get("deployment")
    trecs = make_records(topic_raw["prompts"], [dep_cond], tok, cfg, dps_sha=topic_sha)      # 8 × deployment-шаблон
    allrecs = recs + trecs
    acts_path = out / f"probe_acts_{'fp16' if dtype == 'float16' else 'bf16'}.pt"

    # ---------------- A. активации
    if a.skip_extract and acts_path.exists():
        payload = torch.load(acts_path, weights_only=False)
        assert payload["dtype"] == dtype and payload["keys"] == [r.prompt_key for r in allrecs], "acts.pt не соответствует текущим записям"
        acts = payload["acts"]
    else:
        model = load_model(cfg.tokenizer["name"], cfg.model["revision"], getattr(torch, dtype))
        t0 = time.time()
        acts = extract(model, tok, allrecs, layers, list(POSITIONS), batch_size=a.batch_size, padding_side="right")
        log.info("extract: %d records × %d layers in %.0fs", len(allrecs), len(layers), time.time() - t0)
        for p in POSITIONS:
            for L in layers:
                assert torch.isfinite(acts[p][L]).all(), f"non-finite acts {p} L{L}"
        payload = {"dtype": dtype, "layers": layers, "positions": list(POSITIONS), "keys": [r.prompt_key for r in allrecs],
                   "acts": {p: {L: acts[p][L].float() for L in layers} for p in POSITIONS}, "config_hash": cfg.config_hash,
                   "dps_sha256": dsha, "topic_sha256": topic_sha, "model_revision": cfg.model["revision"]}
        torch.save(payload, acts_path); acts = payload["acts"]
        del model; release()
    keys = [r.prompt_key for r in allrecs]
    cond = [r.condition_id for r in allrecs]; env = [r.env for r in allrecs]; ep = [r.env_prompt for r in allrecs]
    def m(pred):
        return torch.tensor([pred(i) for i in range(len(allrecs))])
    is_dp = m(lambda i: env[i] != "topic")
    md, me1 = m(lambda i: cond[i] == "deployment" and env[i] != "topic"), m(lambda i: cond[i] == "E1")
    X = {L: acts["last"][L] for L in layers}
    contrast = md | me1
    y_c = torch.tensor([1.0 if cond[i] == "E1" else 0.0 for i in range(len(allrecs))])[contrast]
    env_c = [env[i] for i in range(len(allrecs)) if contrast[i]]; ep_c = [ep[i] for i in range(len(allrecs)) if contrast[i]]

    # ---------------- B. слой (без E2/S)
    lr = choose_layer_cross_env({L: X[L][contrast] for L in layers}, y_c, env_c)
    L = lr["layer"]; XL = X[L]
    w = diff_of_means(XL[contrast], y_c)
    proj = XL @ w
    record_numbers(nums, {"probe.dtype": dtype, "probe.layers_candidates": layers, "probe.layer": L, "probe.layer_rule": lr,
                          "probe.position": "last", "probe.topic_sha256": topic_sha, "probe.n_records": len(allrecs)})

    # ---------------- C. валидации 1–5 (S не трогаются)
    V = {}
    ho = heldout_by_prompt(XL[contrast], y_c, ep_c)
    V["v1.heldout_auc_pool"] = ho["heldout_auc_mean"]; V["v1.heldout_aucs"] = ho["heldout_aucs"]
    for e in sorted(set(env_c)):
        sel = torch.tensor([x == e for x in env_c])
        V[f"v1.heldout_auc.{e}"] = heldout_by_prompt(XL[contrast][sel], y_c[sel], [p for p, x in zip(ep_c, env_c) if x == e])["heldout_auc_mean"]
    V["v1.pass"] = V["v1.heldout_auc_pool"] >= thr["v1_heldout_auc_min"]
    mpc = m(lambda i: cond[i] == "positive_control")
    means = {c: proj[m(lambda i, c=c: cond[i] == c and env[i] != "topic")].mean().item() for c in ("deployment", "E1", "positive_control")}
    V["v2.means"] = means; V["v2.pass"] = means["positive_control"] > means["E1"] > means["deployment"]
    for e in ("sandbagging", "secret_number"):
        V[f"v2.means.{e}"] = {c: proj[m(lambda i, c=c, e=e: cond[i] == c and env[i] == e)].mean().item() for c in ("deployment", "E1", "positive_control")}
    # 4 topic (unpaired: topic-промпты не спарены с DP)
    mt = m(lambda i: env[i] == "topic")
    sf = shift_frac_and_sign(w, XL[md], XL[me1], XL[mt])
    V["v4.topic"] = {**sf, "unpaired_perm": unpaired_perm_test(proj[mt], proj[md], a.n_perm), "auc_topic_vs_deploy": auc(torch.cat([proj[mt], proj[md]]), torch.cat([torch.ones(int(mt.sum())), torch.zeros(int(md.sum()))])),
                     "norm_topic_mean": XL[mt].norm(dim=1).mean().item(), "norm_deploy_mean": XL[md].norm(dim=1).mean().item()}
    # 2026-08-28 (ревью R-Pipeline А.3): эквивалентность, а не «p>0.05»: бутстрап-CI (90 %, = TOST при alpha=0.05) для frac_of_e1
    # topic vs deployment должен лежать в (−m, m), m = v4_topic_equivalence_margin_sd (порог на долю E1-сдвига). Старый флаг — репорт.
    gb = torch.Generator().manual_seed(1)
    pt, pd_ = proj[mt], proj[md]; e1_shift = (proj[me1].mean() - pd_.mean()).item()
    fr = []
    for _ in range(a.n_boot):
        it = torch.randint(0, len(pt), (len(pt),), generator=gb); idd = torch.randint(0, len(pd_), (len(pd_),), generator=gb)
        fr.append(((pt[it].mean() - pd_[idd].mean()) / e1_shift).item() if e1_shift else float("nan"))
    fr.sort(); m_eq = float(thr["v4_topic_equivalence_margin_sd"]); alpha = float(thr.get("v4_topic_tost_alpha", 0.05))
    lo, hi = fr[int(alpha * len(fr))], fr[int((1 - alpha) * len(fr)) - 1]
    V["v4.tost"] = {"frac_ci_lo": lo, "frac_ci_hi": hi, "margin": m_eq, "alpha": alpha, "n_boot": a.n_boot}
    V["v4.pass"] = (-m_eq < lo) and (hi < m_eq)
    V["v4.pass_legacy_absence_of_evidence"] = (abs(sf["frac_of_e1"]) < m_eq) and (V["v4.topic"]["unpaired_perm"]["p"] > 0.05 or not sf["sign_matches_e1"])
    # 5 E2 по вставкам + sys_neutral; пул и по средам; perm-полоса
    for c in ("e2_a", "e2_b", "e2_c", "sys_neutral"):
        for e in ("pool", "sandbagging", "secret_number"):
            sel_t = m(lambda i, c=c, e=e: cond[i] == c and (e == "pool" or env[i] == e))
            sel_d = m(lambda i, e=e: cond[i] == "deployment" and env[i] != "topic" and (e == "pool" or env[i] == e))
            sel_1 = m(lambda i, e=e: cond[i] == "E1" and (e == "pool" or env[i] == e))
            r = shift_frac_and_sign(w, XL[sel_d], XL[sel_1], XL[sel_t])
            r["auc_vs_deploy"] = auc(torch.cat([proj[sel_t], proj[sel_d]]), torch.cat([torch.ones(int(sel_t.sum())), torch.zeros(int(sel_d.sum()))]))
            r["paired_perm"] = paired_perm_test(proj[sel_t] - proj[sel_d], a.n_perm)
            V[f"v5.{c}.{e}"] = r
    e2_pool = [V[f"v5.{c}.pool"] for c in ("e2_a", "e2_b", "e2_c")]
    def _ok(r):
        return r["sign_matches_e1"] and (r["frac_of_e1"] >= thr["v5_e2_shift_frac_of_e1_min"] or r["paired_perm"]["p"] <= thr["v5_e2_perm_p_max"])
    # 2026-08-28 (ревью R-Pipeline А.3): ОДИН флаг-ворота = e2_a (user-prefix) и e2_b (system) — обобщение на ДРУГУЮ позицию/роль
    # (addendum §10: доля ≥0.5 ИЛИ perm-p ≤0.05, знак = E1). e2_c (та же позиция и роль, что E1) — диагностика перефразировки/локальности, репорт.
    V["v5.pass"] = all(_ok(r) for r in e2_pool[:2])
    V["v5.per_insert_ok"] = {c: _ok(V[f"v5.{c}.pool"]) for c in ("e2_a", "e2_b", "e2_c")}
    V["v5.report_all_three"] = all(_ok(r) for r in e2_pool)
    V["v5.note"] = "gate = e2_a ∧ e2_b (другая позиция/роль); e2_c = перефразировка в позиции E1 (addendum §2) — репорт локальности"
    record_numbers(nums, {f"probe.{k}": v for k, v in V.items()})
    record_numbers(nums, {"probe.validations_recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "probe.gate_v1_v2_v4": bool(V["v1.pass"] and V["v2.pass"] and V["v4.pass"]), "probe.gate_v5": bool(V["v5.pass"]),
                          "probe.gate_v3_note": "v3 (perm-полоса по S) считается на стадии main: probe.gate_v3.{scope}"})
    log.info("валидации 1–5 записаны; слой L%d; v1=%.3f v2=%s v4=%s v5=%s", L, V["v1.heldout_auc_pool"], V["v2.pass"], V["v4.pass"], V["v5.pass"])
    if a.stage == "validate":
        print(json.dumps({"stage": "validate", "layer": L, "layer_rule": lr["scores"], "v1": V["v1.heldout_auc_pool"], "v2": V["v2.means"],
                          "v4": V["v4.tost"], "v5": V["v5.per_insert_ok"], "gate_v1_v2_v4": bool(V["v1.pass"] and V["v2.pass"] and V["v4.pass"]), "gate_v5": V["v5.pass"]}, indent=1, default=str))
        print(f"→ валидации/слой записаны; {acts_path.name}. Главный вопрос: --stage main после fill_gate_params (B4).")
        return

    # ---------------- D. главный вопрос: 6 условий × 32 промпта
    rows = []
    for i, r in enumerate(allrecs):
        if r.condition_id in MAIN_IDS and r.env != "topic":
            rows.append({"prompt_key": r.prompt_key, "env": r.env, "prompt_id": r.prompt_id, "condition_id": r.condition_id, "probe_score": proj[i].item(),
                         "norm": XL[i].norm().item(), "n_prompt_tokens": r.n_prompt_tokens})
    n_dp = len({r.env_prompt for r in allrecs if r.env != "topic"})
    assert len(rows) == n_dp * 6, (len(rows), n_dp)
    with open(out / "probe_scores.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0])); wcsv.writeheader(); wcsv.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(f"{r['env']}/{r['prompt_id']}", {})[r["condition_id"]] = r["probe_score"]
    R = {}
    for scope in ("pool", "sandbagging", "secret_number"):
        eps = [k for k in by if scope == "pool" or k.startswith(scope + "/")]
        d = {c: torch.tensor([by[k][c] - by[k]["deployment"] for k in eps]) for c in MAIN_IDS if c != "deployment"}
        e1 = d["E1"].mean().item()
        for c, v in d.items():
            R[f"delta.{scope}.{c}"] = {**paired_perm_test(v, a.n_perm), "frac_of_e1": v.mean().item() / e1 if e1 else float("nan"), "sign_matches_e1": v.mean().item() * e1 > 0}
        for s in ("S1", "S2"):
            R[f"S_minus_Dstr.{scope}.{s}"] = paired_perm_test(d[s] - d["Dstr"], a.n_perm)
        R[f"S_minus_Dstr.{scope}.S1S2mean"] = paired_perm_test((d["S1"] + d["S2"]) / 2 - d["Dstr"], a.n_perm)
        R[f"interaction.{scope}"] = paired_perm_test(d["E1_S1"] - d["E1"] - d["S1"], a.n_perm)
        # полосы (v3): perm — ворота; random — репорт
        sel_scope = m(lambda i, s=scope: env[i] != "topic" and (s == "pool" or env[i] == s))
        cs = contrast & sel_scope
        yc = torch.tensor([1.0 if cond[i] == "E1" else 0.0 for i in range(len(allrecs)) if cs[i]])
        pairs = [ep[i] for i in range(len(allrecs)) if cs[i]]
        for c in ("S1", "S2", "Dstr", "E1_S1"):
            tsel = m(lambda i, c=c, s=scope: cond[i] == c and env[i] != "topic" and (s == "pool" or env[i] == s))
            dsel = m(lambda i, s=scope: cond[i] == "deployment" and env[i] != "topic" and (s == "pool" or env[i] == s))
            R[f"band.{scope}.{c}.perm"] = perm_band(XL[cs], yc, pairs, XL[tsel], XL[dsel], n_perm=int(thr.get("v3_perm_n", 200)))
            R[f"band.{scope}.{c}.random"] = random_band(XL[cs], yc, XL[tsel], XL[dsel], n_dirs=int(thr.get("v3_random_directions_n", 10)))
        R[f"strong.{scope}"] = {
            "S_minus_Dstr_significant": min(R[f"S_minus_Dstr.{scope}.S1"]["p"], R[f"S_minus_Dstr.{scope}.S2"]["p"]) <= 0.05,
            "Dstr_within_perm_band": R[f"band.{scope}.Dstr.perm"]["perm_p"] > thr.get("v3_perm_alpha", 0.05),
            "S_above_perm_band": all(R[f"band.{scope}.{s}.perm"]["perm_p"] <= thr.get("v3_perm_alpha", 0.05) and R[f"delta.{scope}.{s}"]["sign_matches_e1"] for s in ("S1", "S2")),
            "reading": "§8: S−Dstr значим И Dstr в полосе → Strong-кандидат; оба двигают → ответ про false positives пробе-монитора"}
    for scope in ("pool", "sandbagging", "secret_number"):
        R[f"gate_v3.{scope}"] = bool(R[f"strong.{scope}"]["S_above_perm_band"])
    record_numbers(nums, {f"probe.{k}": v for k, v in R.items()})
    record_numbers(nums, {"probe.main_recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    # диагностика позиций (secondary; position_change_rule=none)
    diag = {}
    for p in ("last_user", "mean_user"):
        Xp = acts[p][L]; wp = diff_of_means(Xp[contrast], y_c); pp = Xp @ wp
        diag[p] = {c: (pp[m(lambda i, c=c: cond[i] == c and env[i] != "topic")].mean() - pp[md].mean()).item() for c in ("E1", "S1", "S2", "Dstr", "e2_a", "e2_b", "e2_c")}
    record_numbers(nums, {"probe.diag_positions": diag})

    # ---------------- E. фигура
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, e in zip(axes, ("sandbagging", "secret_number")):
            eps = [k for k in by if k.startswith(e + "/")]
            dep = torch.tensor([by[k]["deployment"] for k in eps]); e1 = torch.tensor([by[k]["E1"] for k in eps])
            scale = (e1 - dep).mean().item() or 1.0
            for xi, c in enumerate(MAIN_IDS):
                v = (torch.tensor([by[k][c] for k in eps]) - dep) / scale
                ax.scatter((xi + (torch.rand(len(v)) - 0.5) * 0.3).tolist(), v.tolist(), s=12, alpha=0.5)
                ax.plot([xi - 0.25, xi + 0.25], [v.mean().item()] * 2, color="k", lw=2)
            ax.axhline(0, color="gray", lw=0.8); ax.axhline(1, color="gray", lw=0.8, ls="--")
            ax.set_xticks(range(len(MAIN_IDS))); ax.set_xticklabels(MAIN_IDS); ax.set_title(f"{e} — probe score along deploy→E1 (L{L}), E1 = 1")
        axes[0].set_ylabel("Δ projection / mean Δ(E1)")
        fig.tight_layout(); fig.savefig(out / "fig_probe_by_condition.png", dpi=150)
    except Exception as ex:  # noqa: BLE001
        log.warning("figure failed: %s", ex)

    print(json.dumps({"layer": L, "layer_rule": lr["scores"], "v1": V["v1.heldout_auc_pool"], "v2": V["v2.means"], "v4": {k: V["v4.topic"][k] for k in ("frac_of_e1", "sign_matches_e1")},
                      "v5_pool": {c: (round(V[f"v5.{c}.pool"]["frac_of_e1"], 3), round(V[f"v5.{c}.pool"]["auc_vs_deploy"], 3)) for c in ("e2_a", "e2_b", "e2_c", "sys_neutral")},
                      "deltas_pool_frac_of_E1": {c: round(R[f"delta.pool.{c}"]["frac_of_e1"], 3) for c in ("S1", "S2", "Dstr", "E1_S1")},
                      "S_minus_Dstr_pool_p": {s: R[f"S_minus_Dstr.pool.{s}"]["p"] for s in ("S1", "S2", "S1S2mean")},
                      "strong": {s: R[f"strong.{s}"] for s in ("pool", "sandbagging", "secret_number")}}, indent=1, default=str))
    print(f"→ results/probe_scores.csv (192 строк), results/fig_probe_by_condition.png, {acts_path.name}")


if __name__ == "__main__":
    main()
