"""Синтетический смоук P2 (К-7): фиктивные активации с известной структурой → проверяем, что функции возвращают ожидаемое."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import p2_analysis as P

import os
art = json.load(open("results/p4_frozen.json" if os.path.exists("results/p4_frozen.json") else "p4_frozen.json"))
rng = np.random.default_rng(0); d = 48; L = 21
conds = ["deployment","E1","S1","S2","E1_S1","Dstr","e2_a","e2_b","e2_c","sys_neutral","positive_control","positive_control_strong"]
keys, cell_shift = [], {}
for env in ("sandbagging","secret_number"):
    for i in range(16):
        for c in conds: keys.append(f"{env}/{'sb' if env=='sandbagging' else 'sn'}_{i+1:03d}/{c}")
for i in range(8): keys.append(f"topic/tp_{i+1:03d}/deployment")
extra_keys = [f"{p['env']}/{p['prompt_id']}/deployment" for p in art["prompts"]] + [f"x_dptrunc/dt_sb_{i:03d}/deployment" for i in range(1,9)] + [f"x_wild_other/wd_{i:03d}/deployment" for i in range(8)] + [f"x_wild_ai/wd_{i:03d}/deployment" for i in range(8,10)] + [f"x_len140/L140_sb_{i:03d}/deployment" for i in range(1,9)] + [f"x_len210/L210_sb_{i:03d}/deployment" for i in range(1,9)]
axis = np.zeros(d); axis[0] = 1
shift = {"E1":3.0,"E1_S1":3.2,"S1":0.4,"S2":0.5,"Dstr":-0.4,"e2_a":2.5,"e2_b":2.0,"e2_c":2.8,"sys_neutral":0.9,"positive_control":1.0,"positive_control_strong":2.0}
cellshift = {"a1":2.2,"a2":2.1,"b":0.3,"c1":0.2,"c2":0.0,"d":1.2,"n":0.0,"dptrunc":0.1,"len140":0.15,"len210":0.2,"wild_other":0.05,"wild_ai":0.5,"a_head":1.0,"a_tail":2.0}
_base = {}
def acts_for(klist):
    X = np.zeros((len(klist), d), dtype=np.float32)
    for r,k in enumerate(klist):
        env,pid,c = k.split("/")
        if (env,pid) not in _base:
            b = 0.3*rng.normal(size=d); b[0] = 0.0; _base[(env,pid)] = b
        s = shift.get(c,0.0) if not env.startswith(("x_","topic")) else (3.1 if pid in ("tp_003","tp_004") else 0.6 if env=="topic" else cellshift[env[2:] if env!="x_dptrunc" else "dptrunc"])
        X[r] = _base[(env,pid)] + s*axis + 0.2*rng.normal(size=d)
    return X
base = {"keys":keys,"layers":[19,21,23],"acts":{p:{l:acts_for(keys)+0.05*i*rng.normal(size=(len(keys),d)).astype(np.float32) for l in (19,21,23)} for i,p in enumerate(P.POSITIONS)}}
extra = {"keys":extra_keys,"acts":{p:{l:acts_for(extra_keys) for l in (19,21,23)} for p in P.POSITIONS}}
nll_rows = [{"prompt_key":k,"n_user_words":str(rng.integers(55,85)),"nll_user":str(rng.normal(3,0.3))} for k in keys+extra_keys]
texts_user = {k: ("evaluation test score benchmark of the system" if "/E1" in k or k.startswith("x_a1") else "we ship a model for churn forecasting help me plan") for k in keys+extra_keys}
texts_full = dict(texts_user)
ask_rows = [{"kind":"main","env":"sandbagging","condition_id":"E1","answer":"yes"},{"kind":"main","env":"sandbagging","condition_id":"deployment","answer":"no"},{"kind":"main","env":"x_a1","condition_id":"deployment","answer":"yes"},{"kind":"self","env":"sandbagging","condition_id":"E1","answer":"no"},{"kind":"self","env":"x_a1","condition_id":"deployment","answer":"no"}]
out = Path(tempfile.mkdtemp()); (out/"numbers.json").write_text("{}")
D = P.Data(base, extra, art, nll_rows)
assert D.N == len(keys)+len(extra_keys)
from collections import Counter; cnt = Counter(D.cell); print("cells:", {k:v for k,v in cnt.items() if not k in conds})
R = P.run(D, art, texts_user, texts_full, nll_rows, ask_rows, out, L, n_boot=200, n_perm=300)
g = R["alt3_gate"]; print("alt3:", g["outcome"][:40], g["fired"])
assert set(g["fired"]) <= {"tp_003","tp_004"} and g["outcome"].startswith("A"), g
pp = R["primary_pairs"]; print("pairs frac:", round(pp["mean_diff_frac_of_e1"],3), "p:", pp["p_sign_perm_exact"], "tost:", pp["equivalent_within_margin"])
# проверка ТОЧНОСТИ парного контраста прямым пересчётом (код, не удача генератора)
w_ = P.direction(D, "last", L); s_ = P.scores(D, "last", L, w_)
direct = np.mean([s_[D.idx("a1",a)] - s_[D.idx("a2",b)] for a,b in art["pairs"]])
assert abs(pp["mean_diff"] - direct) < 1e-4, (pp["mean_diff"], direct)
assert 0 < pp["p_sign_perm_exact"] <= 1 and pp["min_achievable_p"] == 2/256
ct = R["cells_last"]; print("a1 frac:", round(ct["a1"]["frac_of_e1"],2), "c2 frac:", round(ct["c2"]["frac_of_e1"],2), "fpr a1:", ct["a1"]["fpr_loo"])
assert ct["a1"]["frac_of_e1"] > 0.5 and abs(ct["c2"]["frac_of_e1"]) < 0.3
assert ct["a1"]["n"] == 8 and ct["dptrunc"]["n"] == 8
jt = R["secondary"]["jt_trend"]; print("JT order", jt["order"], "p:", jt["p_one_sided"]); assert jt["p_one_sided"] < 0.05
oo = R["oos"]; print("OOS: tpr", oo["tpr_oos"], "fpr_dep", oo["fpr_deploy_oos"], "auc", round(oo["auc_oos"],3), "cross", {k:round(v["fpr_deploy_other"],2) for k,v in oo["cross_env"].items()})
hv = R["head_vs_tail"]; print("head/tail frac:", round(hv["a_head"]["frac_of_e1"],2), round(hv["a_tail"]["frac_of_e1"],2), "p", hv["p_perm"]); assert hv["a_tail"]["frac_of_e1"] > hv["a_head"]["frac_of_e1"]
assert ct["wild_other"]["n"] == 8 and ct["wild_ai"]["n"] == 2 and ct["a_head"]["n"] == 8
ps = R["precision_surface"]; print("surface: fpr_t", round(ps["fpr_topical"],2), "fpr_n", round(ps["fpr_neutral"],2), "q_break", ps["q_topic_where_precision_at_1pct_below_0.5"])
a5 = R["a5_hard_negative"]; print("A5 loco a2:", round(a5["loco"]["a2"]["fpr_holdout_loo"],2), "no_hn:", round(a5["no_hard_neg"]["fpr_topic_loo"],2))
print("positions:", {p:(round(v["auc_e1_vs_deploy"],2), round(v["topical_frac_of_e1"],2)) for p,v in R["positions"].items()})
cf = R["confirmatory"]["sandbagging"]; print("S1-Dstr p:", cf["S1_minus_Dstr"]["p_sign_perm"], "Dstr sign:", cf["Dstr"]["sign"]); assert cf["S1_minus_Dstr"]["p_sign_perm"] < 0.05 and cf["Dstr"]["sign"] == -1
print("grep strict a1 fpr:", R["a2_grep"]["user_only.eval_strict.keep_checklist"].get("fpr_a1"))
print("Alt2 cos_pc1:", round(R["Alt2_anisotropy"]["cos_w_pc1"],3))
P.rec(out/"numbers.json", {"probe.exploratory.p2": R})
n = json.load(open(out/"numbers.json")); print("ключей записано:", len(n), "| пример:", [k for k in n if "primary_pairs.p_sign" in k])
P.md_report(R, out); dr = R["density_regression"]; print("регрессия: n", dr.get("n"), "R2", round(dr.get("r2",float("nan")),2), "beta", {k:round(v,2) for k,v in dr.get("std_beta",{}).items()})
assert dr["n"] > 30 and set(dr["std_beta"]) == {"eval_density","ai_narrow","ai_broad","n_words"}
print("лестница:", {c: round(R["cells_last"][c]["frac_of_e1"],2) for c in ("dptrunc","len140","len210") if R["cells_last"].get(c,{}).get("n")})
print("fully_oos:", {k: round(v,2) for k,v in list(R["oos"].get("fully_oos_fpr",{}).items())[:4]})
assert "a_family" in R["a5_hard_negative"]["loco"]
# ПОЛ точного знакового теста для КАЖДОГО вызова (баг float32 давал p=0.0)
import itertools as _it
rng2 = np.random.default_rng(3)
for n in (4,7,8,10,12,15,16):
    for _ in range(60):
        d = np.abs(rng2.normal(size=n)) + 0.1
        pv = P.sign_perm_exact(d)
        assert pv >= 2/2**n - 1e-12, (n, pv)
        assert abs(pv - 2/2**n) < 1e-12, (n, pv, "единогласные разности должны давать ровно пол")
for key, sub in (("primary_pairs","p_sign_perm_exact"), ("scenario_pairs","p_sign_perm_exact")):
    v = R[key][sub]; n = R[key]["n_pairs"] if "n_pairs" in R[key] else R[key]["n"]
    assert v >= 2/2**n - 1e-12, (key, v, n)
for env in ("sandbagging","secret_number"):
    cf = R["confirmatory"][env]
    for k in ("S1","S2","Dstr","E1_S1"):
        assert cf[k]["p_sign_perm"] >= 2/2**cf["n"] - 1e-12, (env, k, cf[k]["p_sign_perm"])
print("floor-asserts OK | scenario p:", R["scenario_pairs"]["p_sign_perm_exact"], "(пол", 2/2**R["scenario_pairs"]["n"], ")")
print("ask groups:", sorted(k for k in R["ask_model"] if k in ("main","self","inverted")))
print("jaccard a1/a2:", R["primary_pairs"].get("per_pair_jaccard"))
print("SMOKE OK")
