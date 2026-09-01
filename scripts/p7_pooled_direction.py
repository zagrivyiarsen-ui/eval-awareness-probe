#!/usr/bin/env python3
"""P7 — направление на ОБЪЕДИНЁННОМ позитивном классе.

Зачем. Головной негативный результат работы получен на направлении, обученном на ОДНОЙ
вставке в ОДНОЙ позиции (E1). Первое возражение рецензента: «вы опровергли соломенное
чучело — так пробе никто не строит». P7 закрывает это возражение БЕЗ новых генераций:
то же самое считается для направления, обученного на четырёх формулировках вставки сразу
(E1 ∪ e2_a ∪ e2_b ∪ e2_c) и, отдельно, с усиленным позитивным контролем.

Плюс LOIO по конструкции: направление учится на трёх формулировках, проверяется на
четвёртой. Это прямой ответ на «а если обучить на разнообразии — заработает?».

Читает замороженные активации, пишет ТОЛЬКО под probe.exploratory.p7.* и results/p7_report.md.
P2/P4/P5 не трогает.

    PYTHONPATH=. python3 scripts/p7_pooled_direction.py --out results --artifact results/p4_frozen.json
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
INSERTS = ("E1", "e2_a", "e2_b", "e2_c")
TOPICAL = ("a1", "a2", "d", "d_legacy", "a_legacy")
DIAG = ("a_head", "a_tail")
GRADIENT = ("b", "c1", "c2")
NEUTRAL = ("n", "wild_other")
GROUPS = {"topical_leading": TOPICAL, "diagnostic": DIAG, "gradient": GRADIENT,
          "neutral": NEUTRAL, "wild_other_only": ("wild_other",)}


# ─────────────────────────────────────────────── утилиты (копии p4, чтобы скрипт был автономен)
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def precision_at(prev: float, tpr: float, fpr: float) -> float | None:
    den = prev * tpr + (1 - prev) * fpr
    return (prev * tpr / den) if den > 0 else None


def tau_loo(pos: np.ndarray, q: float = 0.05):
    return float(np.quantile(pos, q)), np.array([np.quantile(np.delete(pos, i), q) for i in range(len(pos))])


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


# ─────────────────────────────────────────────── загрузка (та же схема, что в p4_analysis.D)
class D:
    def __init__(self, out: Path, art: dict):
        import torch
        base = torch.load(out / "probe_acts_fp16.pt", weights_only=False)
        extra = torch.load(out / "probe_acts_extra_fp16.pt", weights_only=False)
        self.layers = list(base["layers"])
        self.keys = list(base["keys"]) + list(extra["keys"])
        self.X = {p: {L: np.concatenate([base["acts"][p][L].float().numpy(),
                                         extra["acts"][p][L].float().numpy()])
                      for L in self.layers} for p in base["positions"]}
        a_leg = set(art["legacy_reclass"]["a_legacy"])
        d_leg = set(art["legacy_reclass"]["d_legacy"])
        self.env, self.pid, self.cond, cell = [], [], [], []
        for k in self.keys:
            e, p, c = k.split("/")
            self.env.append(e); self.pid.append(p); self.cond.append(c)
            cell.append(("a_legacy" if p in a_leg else "d_legacy" if p in d_leg else "topic_unk")
                        if e == "topic" else (e[2:] if e.startswith("x_") else c))
        self.cell = np.array(cell)
        self.envA = np.array(self.env)
        self.pidA = np.array(self.pid)
        self.N = len(self.keys)

    def m(self, *cs):
        return np.isin(self.cell, cs)

    def dep(self):
        return self.m("deployment") & (self.envA != "topic") & ~np.char.startswith(self.envA.astype(str), "x_")

    def env_prompt(self, mask):
        return np.array([f"{self.env[i]}/{self.pid[i]}" for i in range(self.N)])[mask]


def direction(X: np.ndarray, mpos: np.ndarray, mneg: np.ndarray) -> np.ndarray:
    """diff-of-means, НОРМИРОВАННОЕ (как probe_toy.diff_of_means). Все числа P7 — в этой шкале."""
    return unit(X[mpos].mean(0) - X[mneg].mean(0))


# ─────────────────────────────────────────────── профиль одного направления
def profile(d: D, X: np.ndarray, w: np.ndarray, mpos: np.ndarray, name: str) -> dict:
    s = X @ w
    mdep = d.dep()
    me1 = d.m("E1")
    dep_mean = float(s[mdep].mean())
    e1_shift = float(s[me1].mean() - dep_mean)
    pos_scores = s[mpos]
    tau_in, loo = tau_loo(pos_scores)
    tpr_loo = float(np.mean(pos_scores >= loo))
    # ── ВТОРОЙ ПОРОГ, привязанный к НЕГАТИВАМ, а не к позитивам.
    # τ = 5 % квантиль позитивов работает, только если позитивы отделены. У объединённого
    # класса внутри есть e2_a/e2_b, которые почти не отличаются от deployment, поэтому такой
    # τ проваливается НИЖЕ распределения deployment и FPR = 1 у всех ячеек. Это артефакт
    # правила выбора порога, а не свойство направления. Порог по 95 % квантилю deployment
    # фиксирует FPR_deploy = 0.05 и делает конструкции сравнимыми.
    s_dep = s[mdep]
    tau_dep95 = float(np.quantile(s_dep, 0.95))
    fpr_dep_at_tau_pos = float((s_dep >= tau_in).mean())
    out = {"name": name, "n_positive_train": int(mpos.sum()),
           "e1_shift_normed": e1_shift, "tau_insample": tau_in,
           "tau_median_loo": float(np.median(loo)), "tpr_loo": tpr_loo,
           "tpr_insample": float(np.mean(pos_scores >= tau_in)),
           "tau_dep95": tau_dep95,
           "tpr_at_tau_dep95": float((pos_scores >= tau_dep95).mean()),
           "tpr_E1_at_tau_dep95": float((s[me1] >= tau_dep95).mean()),
           "fpr_deploy_at_tau_positive": fpr_dep_at_tau_pos,
           "tau_rule_degenerate": bool(fpr_dep_at_tau_pos > 0.2),
           "tau_rule_note": ("если fpr_deploy_at_tau_positive велик, τ по 5 % квантилю позитивов "
                             "лежит внутри распределения deployment → все FPR при этом пороге "
                             "недействительны; читать колонки *_at_tau_dep95"),
           "auc_pos_vs_deploy": auc(pos_scores, s[mdep]),
           "auc_E1_vs_deploy": auc(s[me1], s[mdep]),
           "heldout_auc_by_prompt": heldout_auc(d, X, mpos)}
    for g, cells in GROUPS.items():
        mg = d.m(*cells)
        if not mg.sum():
            continue
        x = s[mg]
        k_in = int((x >= tau_in).sum())
        fpr_loo = float(np.mean([(x >= t).mean() for t in loo]))
        k_dep = int((x >= tau_dep95).sum())
        out[f"group.{g}"] = {
            "n": int(mg.sum()),
            "fpr_insample": k_in / int(mg.sum()), "fp_at_tau_insample": k_in,
            "wilson95_insample": wilson(k_in, int(mg.sum())),
            "fpr_loo": fpr_loo,
            "fpr_at_tau_dep95": k_dep / int(mg.sum()),
            "fp_at_tau_dep95": k_dep,
            "wilson95_at_tau_dep95": wilson(k_dep, int(mg.sum())),
            "frac_of_e1": float((x.mean() - dep_mean) / e1_shift) if e1_shift else float("nan"),
            "frac_unreliable": bool(abs(e1_shift) < 0.5 * float(s[mdep].std())),
            "precision_at_1pct": precision_at(0.01, tpr_loo, fpr_loo),
            "precision_at_1pct_tau_dep95": precision_at(0.01, out["tpr_E1_at_tau_dep95"],
                                                        k_dep / int(mg.sum())),
            "precision_note": None if fpr_loo > 0 else f"n/a: FPR=0 при n={int(mg.sum())} → единица механическая",
        }
    return out


def heldout_auc(d: D, X: np.ndarray, mpos: np.ndarray, n_folds: int = 4) -> dict:
    """Held-out по env/prompt_id: направление переобучается на трёх фолдах, AUC на четвёртом.
    Позитивы — тот же класс, на котором направление обучено (для pool — все вставки)."""
    mdep = d.dep()
    m = mpos | mdep
    keys = np.array([f"{d.env[i]}/{d.pid[i]}" for i in range(d.N)])
    uniq = sorted(set(keys[m]))
    aucs = []
    for f in range(n_folds):
        te_keys = set(uniq[f::n_folds])
        te = m & np.array([k in te_keys for k in keys])
        tr = m & ~te
        if not te.sum() or not tr.sum():
            continue
        if not (mpos & tr).sum() or not (mdep & tr).sum():
            continue
        w = direction(X, mpos & tr, mdep & tr)
        s = X @ w
        aucs.append(auc(s[mpos & te], s[mdep & te]))
    return {"mean": float(np.nanmean(aucs)) if aucs else float("nan"), "folds": aucs}


# ─────────────────────────────────────────────── LOIO по конструкции направления
def loio(d: D, X: np.ndarray) -> dict:
    """Обучаем на трёх формулировках вставки, проверяем на четвёртой.
    Если pooled-направление переносится — «провал специфичен E1». Если нет — провал общий."""
    mdep = d.dep()
    res = {}
    for held in INSERTS:
        train_cells = [c for c in INSERTS if c != held]
        mtr = d.m(*train_cells)
        mte = d.m(held)
        if not mte.sum():
            continue
        w = direction(X, mtr, mdep)
        s = X @ w
        tau_in, loo_t = tau_loo(s[mtr])
        tau_dep95 = float(np.quantile(s[mdep], 0.95))
        mt = d.m(*TOPICAL)
        res[held] = {
            "auc_heldout_vs_deploy": auc(s[mte], s[mdep]),
            "tpr_heldout_at_tau_train": float((s[mte] >= tau_in).mean()),
            "tpr_heldout_at_tau_dep95": float((s[mte] >= tau_dep95).mean()),
            "fpr_deploy_at_tau_train": float((s[mdep] >= tau_in).mean()),
            "frac_of_e1_heldout": float((s[mte].mean() - s[mdep].mean()) /
                                        (s[mtr].mean() - s[mdep].mean()))
            if s[mtr].mean() != s[mdep].mean() else float("nan"),
            "fpr_topical_at_tau_train": float((s[mt] >= tau_in).mean()) if mt.sum() else None,
            "fpr_topical_at_tau_dep95": float((s[mt] >= tau_dep95).mean()) if mt.sum() else None,
            "n_heldout": int(mte.sum()),
        }
    res["reading"] = ("если auc_heldout ≈ 1 и fpr_topical низкий — направление, обученное на "
                      "разнообразии формулировок, переносится, и головной вывод надо сузить до "
                      "«рецепта с одной вставкой». если auc_heldout ≈ 0.5 — провал НЕ специфичен "
                      "конструкции обучающего контраста.")
    return res


# ─────────────────────────────────────────────── main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifact", default="results/p4_frozen.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    nums_path = out / "numbers.json"
    N = json.loads(nums_path.read_text(encoding="utf-8"))
    L = int(N["probe.layer"])
    art = json.loads(Path(a.artifact).read_text(encoding="utf-8"))
    d = D(out, art)
    X = d.X[POS][L]
    mdep = d.dep()

    defs = {
        "dom_E1_only": d.m("E1"),
        "pool_4_inserts": d.m(*INSERTS),
    }
    if d.m("positive_control_strong").sum():
        defs["pool_4_inserts_plus_pcstrong"] = d.m(*INSERTS) | d.m("positive_control_strong")

    w_ref = direction(X, defs["dom_E1_only"], mdep)
    R = {"layer": L, "position": POS, "n_records": d.N,
         "note": ("направление на объединённом позитивном классе; ответ на возражение «пробе "
                  "обучен на одной вставке — соломенное чучело». Все числа нормированы (‖w‖=1), "
                  "сравнивать с probe.delta.* / probe.v5.*, НЕ с probe.exploratory.p2.cells_last.*"),
         "positive_class_definitions": {k: sorted(set(d.cell[v])) for k, v in defs.items()},
         "profiles": {}}
    for name, mpos in defs.items():
        w = direction(X, mpos, mdep)
        p = profile(d, X, w, mpos, name)
        p["cos_with_dom_E1_only"] = float(np.dot(unit(w), unit(w_ref)))
        R["profiles"][name] = p
    R["loio_by_insert"] = loio(d, X)

    # ── сравнение по Wilson: отличается ли FPR на внешних негативах?
    base = R["profiles"]["dom_E1_only"].get("group.wild_other_only", {})
    cmp = {}
    for name, p in R["profiles"].items():
        g = p.get("group.wild_other_only")
        if not g or not base:
            continue
        cmp[name] = {"fpr_insample": g["fpr_insample"], "wilson95": g["wilson95_insample"],
                     "overlaps_dom": not (g["wilson95_insample"][0] > base["wilson95_insample"][1]
                                          or g["wilson95_insample"][1] < base["wilson95_insample"][0])}
    R["wild_comparison"] = cmp
    R["wild_comparison"]["reading"] = ("перекрывающиеся Wilson → различие между конструкциями "
                                       "позитивного класса НЕ установлено (n=37 мало; см. P6)")

    from src.contracts import record_numbers

    def san(o):
        if isinstance(o, dict):
            return {str(k): san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [san(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    record_numbers(nums_path, {"probe.exploratory.p7": san(R)})

    # ── отчёт
    Lr = ["# P7 — направление на объединённом позитивном классе", "",
          "Вопрос: головной провал специфичен рецепту «одна вставка в одной позиции» "
          "или сохраняется, если обучать на разнообразии формулировок?", "",
          "## Порог по 5 % квантилю ПОЗИТИВОВ (правило работы)", "",
          "| направление | n поз. | cos с E1-only | held-out AUC | AUC поз vs dep | TPR_loo | "
          "FPR_deploy при этом τ | FPR topical | FPR wild | правило τ выродилось |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, p in R["profiles"].items():
        t = p.get("group.topical_leading", {})
        wl = p.get("group.wild_other_only", {})
        Lr.append(f"| {name} | {p['n_positive_train']} | {p['cos_with_dom_E1_only']:.3f} | "
                  f"{p['heldout_auc_by_prompt']['mean']:.3f} | {p['auc_pos_vs_deploy']:.3f} | "
                  f"{p['tpr_loo']:.3f} | {p['fpr_deploy_at_tau_positive']:.3f} | "
                  f"{t.get('fpr_loo', float('nan')):.3f} | {wl.get('fpr_loo', float('nan')):.3f} | "
                  f"{'ДА' if p['tau_rule_degenerate'] else 'нет'} |")
    Lr += ["", "Если в последней колонке «ДА» — τ по квантилю позитивов лежит ВНУТРИ "
               "распределения deployment (в объединённом классе есть e2_a/e2_b, почти не "
               "отличимые от негатива), и все FPR при нём равны единице механически. "
               "Сравнивать конструкции по этой таблице нельзя.", "",
           "## Порог, привязанный к НЕГАТИВАМ: τ = 95 % квантиль deployment (FPR_deploy = 0.05)", "",
           "| направление | TPR E1 | FPR topical | Wilson | FPR wild | Wilson | prec@1 % |",
           "|---|---|---|---|---|---|---|"]
    for name, p in R["profiles"].items():
        t = p.get("group.topical_leading", {})
        wl = p.get("group.wild_other_only", {})
        pr = wl.get("precision_at_1pct_tau_dep95")
        tw = t.get("wilson95_at_tau_dep95", [float("nan")] * 2)
        ww = wl.get("wilson95_at_tau_dep95", [float("nan")] * 2)
        Lr.append(f"| {name} | {p['tpr_E1_at_tau_dep95']:.3f} | "
                  f"{t.get('fpr_at_tau_dep95', float('nan')):.3f} | [{tw[0]:.2f}; {tw[1]:.2f}] | "
                  f"{wl.get('fpr_at_tau_dep95', float('nan')):.3f} | [{ww[0]:.2f}; {ww[1]:.2f}] | "
                  f"{'n/a' if pr is None else f'{pr:.3f}'} |")
    Lr += ["", "Эта таблица — сравнимая: порог у всех конструкций даёт одинаковый FPR на "
               "deployment, поэтому различия в FPR на чужом трафике относятся к направлению, "
               "а не к правилу выбора порога.", ""]
    Lr += ["## LOIO по конструкции: обучение на трёх вставках, проверка на четвёртой", "",
           "| отложена | AUC held-out vs deploy | TPR при τ_dep95 | frac от сдвига обучения | FPR topical при τ_dep95 |",
           "|---|---|---|---|---|"]
    for h in INSERTS:
        v = R["loio_by_insert"].get(h)
        if not v:
            continue
        ft = v["fpr_topical_at_tau_dep95"]
        ft_s = "—" if ft is None else f"{ft:.3f}"
        Lr.append(f"| {h} | {v['auc_heldout_vs_deploy']:.3f} | {v['tpr_heldout_at_tau_dep95']:.3f} | "
                  f"{v['frac_of_e1_heldout']:.3f} | {ft_s} |")
    Lr += ["", "## ЧИТАТЬ ТАК", "",
           "* pooled AUC ≈ 1 и FPR topical по-прежнему высокий → расширение позитивного класса "
           "НЕ лечит загрязнение; заголовок работы остаётся, но теперь он про класс конструкций, "
           "а не про одну.",
           "* pooled FPR topical заметно ниже при непересекающихся Wilson → заголовок сузить: "
           "«провал специфичен минимальному контрасту с одной вставкой».",
           "* LOIO AUC ≈ 0.5 на отложенной формулировке → направление не кодирует «вставку вообще», "
           "а привязано к конкретной позиции/токену — это усиливает раздел про позицию.",
           "", "ВСЁ IN-SAMPLE по negatives и OOS только по held-out фолдам — не смешивать с "
           "ведущими OOS-числами P2."]
    (out / "p7_report.md").write_text("\n".join(Lr), encoding="utf-8")
    print("\n".join(Lr))
    print("\nЗАПИСАНО под probe.exploratory.p7.* ; отчёт: results/p7_report.md")


if __name__ == "__main__":
    main()
