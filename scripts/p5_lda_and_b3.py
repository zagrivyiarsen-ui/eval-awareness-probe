#!/usr/bin/env python3
"""P5 — (A) диагностика lda_shrinkage, (B) пакет для слепой разметки Dolly.

Читает замороженные активации, пишет ТОЛЬКО под probe.exploratory.p5.*.
P2/P3/P4 не трогает. Запуск в той же CPU-сессии, results/ уже разложен:

    python3 scripts/p5_lda_and_b3.py --out results --artifact results/p4_frozen.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent if _here.name == "scripts" else _here))

POS = "last"
TOPICAL = ("a1", "a2", "d", "d_legacy", "a_legacy")
NEUTRAL = ("n", "wild_other")
DIAG = ("a_head", "a_tail")

CODING_RULE = """ПРАВИЛО КОДИРОВКИ (зафиксировано до подсчёта).

Поставьте 1, если промпт ОДНОВРЕМЕННО:
  (а) сообщает измеренные показатели конкретного человека или людей
      (числа, результаты теста, замеры, стаж, динамика за период), И
  (б) просит их интерпретировать, оценить, сравнить или сделать по ним вывод.

Поставьте 0 во всех остальных случаях, включая:
  - просьбы что-то сгенерировать, написать, спланировать, составить список;
  - классификацию/извлечение из текста;
  - расчётные задачи без человека в роли объекта оценки;
  - советы без приведённых измерений.

Числа сами по себе не дают 1: важно, что измеряют ЧЕЛОВЕКА и просят оценку.
Категория выделена post-hoc; вы оцениваете тексты, НЕ видя никаких результатов.
"""


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def tau_loo(e1, q=0.05):
    return float(np.quantile(e1, q)), np.array([np.quantile(np.delete(e1, i), q) for i in range(len(e1))])


def precision_at(p, tpr, fpr):
    den = p * tpr + (1 - p) * fpr
    return (p * tpr / den) if den > 0 else float("nan")


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
        a_leg = set(art["legacy_reclass"]["a_legacy"]); d_leg = set(art["legacy_reclass"]["d_legacy"])
        self.env, self.pid, self.cell = [], [], []
        for k in self.keys:
            e, p, c = k.split("/")
            self.env.append(e); self.pid.append(p)
            self.cell.append(("a_legacy" if p in a_leg else "d_legacy" if p in d_leg else "topic_unk") if e == "topic"
                             else (e[2:] if e.startswith("x_") else c))
        self.cell = np.array(self.cell); self.envA = np.array(self.env); self.pidA = np.array(self.pid)
        self.N = len(self.keys)
        self.texts = {}
        for fn in ("prompt_records.jsonl", "new_prompt_records.jsonl"):
            p = out / fn
            if p.exists():
                for line in open(p, encoding="utf-8"):
                    d = json.loads(line)
                    m = d.get("messages") or []
                    if m:
                        self.texts[d["prompt_key"]] = m[-1]["content"]

    def m(self, *cs):
        return np.isin(self.cell, cs)

    def dep(self):
        return self.m("deployment") & (self.envA != "topic") & ~np.char.startswith(self.envA.astype(str), "x_")


def profile(d: D, s: np.ndarray, name: str) -> dict:
    """Полный профиль направления: TPR, FPR по всем группам, precision, вырожденность."""
    e1 = s[d.m("E1")]; dep = s[d.dep()]
    shift = float(e1.mean() - dep.mean()); sd = float(np.std(dep))
    tau_in, loo = tau_loo(e1)
    tpr = float(np.mean(e1 >= loo))
    out = {"name": name, "e1_shift": shift, "sd_deployment": sd,
           "degenerate_denominator": bool(abs(shift) < 0.5 * sd),
           "auc_e1_vs_deploy": auc(e1, dep), "tpr_loo": tpr, "tau_insample": tau_in}
    groups = {"topical_leading": TOPICAL, "neutral": NEUTRAL, "wild_other_only": ("wild_other",),
              "tn_only": ("n",), "topical_diag": DIAG, "structural_dptrunc": ("dptrunc",),
              "ladder": ("len140", "len210"), "b": ("b",), "c": ("c1", "c2"), "deployment": None}
    fprs = {}
    for g, cells in groups.items():
        x = dep if cells is None else s[d.m(*cells)]
        if not len(x):
            continue
        f = float(np.mean([(x >= t).mean() for t in loo]))
        k = int((x >= tau_in).sum())
        fprs[g] = {"n": int(len(x)), "fpr_loo": f, "fp_insample": k, "wilson95_insample": wilson(k, len(x)),
                   "frac_of_e1": float((x.mean() - dep.mean()) / shift) if shift else float("nan")}
    out["by_group"] = fprs
    ft, fn_ = fprs["topical_leading"]["fpr_loo"], fprs["neutral"]["fpr_loo"]
    out["precision_at_1pct"] = {
        "q_topic_1": (precision_at(0.01, tpr, ft) if ft > 0 else None),
        "q_topic_0": (precision_at(0.01, tpr, fn_) if fn_ > 0 else None),
        "note_if_none": "FPR=0 → precision=1.0 механически, приведён Wilson по in-sample вместо числа"}
    for c in ("a1", "a2", "d", "d_legacy", "a_legacy", "a_head", "a_tail", "b", "c1", "c2", "n",
              "wild_other", "dptrunc", "len140", "len210"):
        mc = d.m(c)
        if mc.sum():
            x = s[mc]
            out.setdefault("per_cell_fpr_loo", {})[c] = float(np.mean([(x >= t).mean() for t in loo]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifact", default="results/p4_frozen.json")
    ap.add_argument("--seed", type=int, default=20260901)
    a = ap.parse_args(argv)
    out = Path(a.out); nums = out / "numbers.json"
    n = json.loads(nums.read_text(encoding="utf-8")); L = int(n["probe.layer"])
    art = json.loads(Path(a.artifact).read_text(encoding="utf-8"))
    d = D(out, art)

    X = d.X[POS][L]; me1, mdep = d.m("E1"), d.dep()
    Xtr = np.concatenate([X[me1], X[mdep]])
    y = np.concatenate([np.ones(me1.sum()), np.zeros(mdep.sum())])

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    R = {"layer": L, "position": POS,
         "note": ("P5: (A) профиль lda_shrinkage на ВСЕХ группах негативов — решает, специфичен ли провал "
                  "diff-of-means; (B) пакет для слепой разметки Dolly. P2/P3/P4 не пересчитываются.")}

    # ── A. профили направлений
    w_dm = X[me1].mean(0) - X[mdep].mean(0)
    dirs = {"diff_of_means": w_dm}
    fitted = {"lda_shrinkage": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
              "linear_svc": LinearSVC(C=1.0, max_iter=20000)}
    for nm, clf in fitted.items():
        clf.fit(Xtr, y); dirs[nm] = np.asarray(clf.coef_).ravel()
    pipe = make_pipeline(PCA(n_components=20), LinearDiscriminantAnalysis(solver="svd")).fit(Xtr, y)
    dirs["pca20_lda_svd"] = pipe.steps[0][1].components_.T @ np.asarray(pipe.steps[1][1].coef_).ravel()

    # ковариационный контекст: не является ли LDA просто выбиванием главных компонент deployment
    Xd = X[mdep]; mu = Xd.mean(0)
    U, S, Vt = np.linalg.svd(Xd - mu, full_matrices=False)
    pcs = Vt[:5]
    R["profiles"] = {}
    for nm, w in dirs.items():
        s = X @ w
        p = profile(d, s, nm)
        p["cos_with_diff_of_means"] = float(w @ w_dm / (np.linalg.norm(w) * np.linalg.norm(w_dm) + 1e-9))
        p["cos_with_deployment_pcs"] = [float(abs(w @ v) / (np.linalg.norm(w) * np.linalg.norm(v) + 1e-9)) for v in pcs]
        R["profiles"][nm] = p
    R["pc_var_frac_deployment"] = [float(S[i] ** 2 / (S ** 2).sum()) for i in range(5)]

    # ── B. пакет для слепой разметки
    mw = np.where(d.m("wild_other"))[0]
    items = [{"prompt_id": d.pid[i], "text": d.texts.get(d.keys[i], "")} for i in mw]
    rng = random.Random(a.seed); order = list(range(len(items))); rng.shuffle(order)
    md = ["# Слепая разметка: 37 внешних промптов (Dolly)", "", CODING_RULE, "",
          "Заполните таблицу в конце: prompt_id → 0 или 1. Порядок предъявления перемешан (сид "
          f"{a.seed}). Никаких результатов измерений вам не показывают и показывать нельзя.", "", "---", ""]
    for k, j in enumerate(order, 1):
        md += [f"## {k}. `{items[j]['prompt_id']}`", "", "```", items[j]["text"].strip(), "```", ""]
    md += ["---", "", "## Ответ — вставьте в results/dolly_labels.json", "", "```json",
           "{", ",\n".join(f'  "{items[j]["prompt_id"]}": 0' for j in order), "}", "```"]
    (out / "dolly_for_blind_coding.md").write_text("\n".join(md), encoding="utf-8")
    (out / "dolly_labels_TEMPLATE.json").write_text(
        json.dumps({items[j]["prompt_id"]: 0 for j in order}, indent=1, ensure_ascii=False), encoding="utf-8")

    R["b3_package"] = {
        "n_prompts": len(items), "shuffle_seed": a.seed,
        "rule_sha256": hashlib.sha256(CODING_RULE.encode()).hexdigest()[:16],
        "file": "dolly_for_blind_coding.md", "template": "dolly_labels_TEMPLATE.json",
        "procedure": ("кодировщик НЕ ДОЛЖЕН видеть скоры, список сработавших и текст этой работы. "
                      "Записать: кто кодировал (человек/модель+версия), точный промпт, дату. "
                      "Затем положить results/dolly_labels.json и перезапустить P4."),
        "contamination_note": ("категория выделена post-hoc по просмотру скоров; автор работы и ассистент, "
                               "предложивший категорию, кодировать НЕ МОГУТ")}

    from src.contracts import record_numbers

    def san(o):
        if isinstance(o, dict): return {str(k): san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [san(x) for x in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        return o

    record_numbers(nums, {"probe.exploratory.p5": san(R)})

    # ── отчёт
    Lr = ["# P5 — специфичен ли провал diff-of-means?", "",
          "| направление | cos с d-o-m | AUC | TPR | FPR topical | FPR neutral | FPR wild | FPR c1+c2 | prec@1% (q=1) | вырожден |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, p in R["profiles"].items():
        g = p["by_group"]
        pr = p["precision_at_1pct"]["q_topic_1"]
        Lr.append(f"| {nm} | {p['cos_with_diff_of_means']:.3f} | {p['auc_e1_vs_deploy']:.3f} | "
                  f"{p['tpr_loo']:.3f} | {g['topical_leading']['fpr_loo']:.3f} | {g['neutral']['fpr_loo']:.3f} | "
                  f"{g['wild_other_only']['fpr_loo']:.3f} | {g.get('c',{}).get('fpr_loo', float('nan')):.3f} | "
                  f"{'n/a (FPR=0)' if pr is None else f'{pr:.3f}'} | {p['degenerate_denominator']} |")
    Lr += ["", "Доля дисперсии deployment в PC1..PC5: " +
           ", ".join(f"{v:.3f}" for v in R["pc_var_frac_deployment"]), "",
           "|cos| направления с PC1..PC5 deployment:", ""]
    for nm, p in R["profiles"].items():
        Lr.append(f"  {nm:16} " + " ".join(f"{v:.3f}" for v in p["cos_with_deployment_pcs"]))
    Lr += ["", "## ЧИТАТЬ ТАК", "",
           "Если у lda_shrinkage FPR низкий НА ВСЕХ группах (topical, neutral, wild) при AUC 1.0 и "
           "невырожденном знаменателе — провал специфичен diff-of-means, и заголовок работы меняется на "
           "«провал специфичен рецепту; отбеливание ковариации чинит загрязнение без потери разделения».",
           "Если FPR низкий только на topical, а на wild/neutral высокий — «лечение» не обобщается, "
           "заголовок остаётся прежним.",
           "Если |cos| с PC1..PC2 deployment у lda_shrinkage заметно ниже, чем у diff-of-means — "
           "механизм именно в подавлении высокодисперсных осей.", "",
           f"## B3: пакет на {R['b3_package']['n_prompts']} промптов → results/dolly_for_blind_coding.md"]
    (out / "p5_report.md").write_text("\n".join(Lr), encoding="utf-8")
    print("\n".join(Lr))
    print("\nЗАПИСАНО под probe.exploratory.p5.* ; отчёт: results/p5_report.md")
    print("B3: отдайте results/dolly_for_blind_coding.md слепому кодировщику, ответ положите как "
          "results/dolly_labels.json и перезапустите P4.")


if __name__ == "__main__":
    main()
