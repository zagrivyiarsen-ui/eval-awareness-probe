#!/usr/bin/env python3
"""export_records.py — ОДНА таблица на все записи: скоры под 4 направлениями × 3 позициями,
ответы модели, длины, NLL, last_user-токен, флаги срабатывания, тексты.

То, чего нет в numbers.json (там только агрегаты) и без чего независимый анализ невозможен.
Пишет results/records_full.csv (+ .jsonl с текстами) и результаты НЕ трогает.

    python3 scripts/export_records.py --out results --artifact results/p4_frozen.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])   # запуск файлом
                if "__file__" in globals() else ".")           # вставка в ячейку ноутбука

ROLE = {"a1": "topical", "a2": "topical", "d": "topical", "d_legacy": "topical", "a_legacy": "topical",
        "a_head": "diagnostic", "a_tail": "diagnostic", "b": "gradient", "c1": "gradient", "c2": "gradient",
        "n": "neutral", "wild_other": "neutral_external", "wild_ai": "neutral_external",
        "dptrunc": "structural", "len140": "ladder", "len210": "ladder",
        "E1": "train_pos", "deployment": "train_neg", "S1": "prereg_S", "S2": "prereg_S", "Dstr": "prereg_S",
        "E1_S1": "prereg_S", "sys_neutral": "prereg_ctrl", "positive_control": "prereg_ctrl",
        "positive_control_strong": "prereg_ctrl", "e2_a": "insert_alt", "e2_b": "insert_alt", "e2_c": "insert_alt"}


def tau_loo(e1, q=0.05):
    return float(np.quantile(e1, q)), np.array([np.quantile(np.delete(e1, i), q) for i in range(len(e1))])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifact", default="results/p4_frozen.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    import torch

    N = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    L = int(N["probe.layer"])
    art = json.loads(Path(a.artifact).read_text(encoding="utf-8"))

    base = torch.load(out / "probe_acts_fp16.pt", weights_only=False)
    extra = torch.load(out / "probe_acts_extra_fp16.pt", weights_only=False)
    keys = list(base["keys"]) + list(extra["keys"])
    POS = list(base["positions"])
    X = {p: np.concatenate([base["acts"][p][L].float().numpy(), extra["acts"][p][L].float().numpy()])
         for p in POS}

    a_leg = set(art["legacy_reclass"]["a_legacy"]); d_leg = set(art["legacy_reclass"]["d_legacy"])
    env, pid, cond, cell = [], [], [], []
    for k in keys:
        e, p, c = k.split("/")
        env.append(e); pid.append(p); cond.append(c)
        cell.append(("a_legacy" if p in a_leg else "d_legacy" if p in d_leg else "topic_unk") if e == "topic"
                    else (e[2:] if e.startswith("x_") else c))
    cell = np.array(cell); envA = np.array(env)
    me1 = cell == "E1"
    mdep = (cell == "deployment") & (envA != "topic") & ~np.char.startswith(envA.astype(str), "x_")

    # ── направления
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    Xl = X["last"]; Xtr = np.vstack([Xl[me1], Xl[mdep]])
    y = np.r_[np.ones(me1.sum()), np.zeros(mdep.sum())]
    dirs = {"dom": Xl[me1].mean(0) - Xl[mdep].mean(0)}
    try:
        dirs["lda"] = np.asarray(LDA(solver="lsqr", shrinkage="auto").fit(Xtr, y).coef_).ravel()
        dirs["svc"] = np.asarray(LinearSVC(C=1.0, max_iter=20000).fit(Xtr, y).coef_).ravel()
        pp = make_pipeline(PCA(20), LDA(solver="svd")).fit(Xtr, y)
        dirs["pca20"] = pp.steps[0][1].components_.T @ np.asarray(pp.steps[1][1].coef_).ravel()
    except Exception as e:  # noqa: BLE001
        print(f"  альтернативные направления пропущены: {type(e).__name__}: {e}")

    scores, taus = {}, {}
    for nm, w in dirs.items():
        s = Xl @ w; scores[f"s_last_{nm}"] = s
        e1 = s[me1]; taus[nm] = (float(np.quantile(e1, 0.05)), float(np.median(tau_loo(e1)[1])))
    for p in POS:
        if p == "last":
            continue
        w = X[p][me1].mean(0) - X[p][mdep].mean(0)
        scores[f"s_{p}_dom"] = X[p] @ w

    # ── тексты, токены, длины, NLL
    texts, ids_span, ntok = {}, {}, {}
    for fn in ("prompt_records.jsonl", "new_prompt_records.jsonl"):
        p = out / fn
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            m = d.get("messages") or []
            if m:
                texts[d["prompt_key"]] = m[-1]["content"]
            if d.get("input_ids") and d.get("user_span"):
                ids_span[d["prompt_key"]] = (d["input_ids"], d["user_span"])
            ntok[d["prompt_key"]] = d.get("n_prompt_tokens")
    nll = {}
    p = out / "nll.csv"
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            nll[r["prompt_key"]] = r
    ask = {}
    p = out / "ask_model.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("kind") in ("main", "self", "inverted"):
                ask.setdefault(r["prompt_key"], {})[r["kind"]] = r["answer"]
    labels = {}
    p = out / "dolly_labels.json"
    if p.exists():
        labels = json.loads(p.read_text(encoding="utf-8"))

    # ── сборка
    cols = (["prompt_key", "env", "prompt_id", "condition_id", "cell", "role"]
            + sorted(scores) + [f"fired_{nm}_tau_insample" for nm in dirs] + [f"fired_{nm}_tau_medloo" for nm in dirs]
            + ["n_words", "n_prompt_tokens", "n_user_tokens", "nll_user", "nll_full",
               "last_user_token_id", "last_token_id", "ask_main", "ask_self", "ask_inverted",
               "dolly_label", "is_train_contrast"])
    rows = []
    for i, k in enumerate(keys):
        seq, span = ids_span.get(k, (None, None))
        lu = seq[span[1] - 1] if (seq and span and 0 <= span[1] - 1 < len(seq)) else None
        nr = nll.get(k, {})
        r = {"prompt_key": k, "env": env[i], "prompt_id": pid[i], "condition_id": cond[i],
             "cell": cell[i], "role": ROLE.get(cell[i], ""),
             "n_words": len(texts[k].split()) if k in texts else None,
             "n_prompt_tokens": ntok.get(k), "n_user_tokens": nr.get("n_user_tokens"),
             "nll_user": nr.get("nll_user"), "nll_full": nr.get("nll_full"),
             "last_user_token_id": lu, "last_token_id": seq[-1] if seq else None,
             "ask_main": ask.get(k, {}).get("main"), "ask_self": ask.get(k, {}).get("self"),
             "ask_inverted": ask.get(k, {}).get("inverted"),
             "dolly_label": labels.get(pid[i]),
             "is_train_contrast": int(bool(me1[i] or mdep[i]))}
        for c_, v in scores.items():
            r[c_] = round(float(v[i]), 6)
        for nm in dirs:
            r[f"fired_{nm}_tau_insample"] = int(scores[f"s_last_{nm}"][i] >= taus[nm][0])
            r[f"fired_{nm}_tau_medloo"] = int(scores[f"s_last_{nm}"][i] >= taus[nm][1])
        rows.append(r)

    with open(out / "records_full.csv", "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=cols); w_.writeheader(); w_.writerows(rows)
    with open(out / "records_texts.jsonl", "w", encoding="utf-8") as fh:
        for i, k in enumerate(keys):
            if k in texts:
                fh.write(json.dumps({"prompt_key": k, "cell": cell[i], "text": texts[k]}, ensure_ascii=False) + "\n")

    # ── словарь колонок, чтобы таблицу можно было читать без этого кода
    doc = {
        "n_records": len(rows), "layer": L, "positions": POS,
        "directions": {"dom": "diff-of-means (prereg, ВЕДУЩЕЕ)", "lda": "LDA lsqr shrinkage=auto",
                       "svc": "LinearSVC C=1", "pca20": "PCA(20) → LDA(svd)"},
        "thresholds": {nm: {"tau_insample": t[0], "tau_median_loo": t[1]} for nm, t in taus.items()},
        "caveats": [
            "все альтернативные направления обучены IN-SAMPLE на всех 64 записях контраста; "
            "ведущие числа работы — OOS, смешивать нельзя",
            "is_train_contrast=1 → запись входит в обучающий контраст E1/deployment; исключать при анализе загрязнения",
            "frac_of_e1 в агрегатах = (mean(cell) − mean(deployment)) / (mean(E1) − mean(deployment)); "
            "знаменатель лежит в том же неспецифическом подпространстве",
            "last_user_token_id у E1 = закрывающая скобка вставки → коллинеарен наличию вставки по построению",
            "FPR=0 при малом n не значит ноль: смотреть Wilson",
        ],
        "columns": {c_: "" for c_ in cols},
    }
    (out / "records_schema.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"  records_full.csv     {len(rows)} строк × {len(cols)} колонок")
    print(f"  records_texts.jsonl  {sum(1 for k in keys if k in texts)} текстов")
    print(f"  records_schema.json  словарь колонок, направления, пороги, оговорки")
    print("\n  сводка по ячейкам (по s_last_dom):")
    for c_ in sorted(set(cell)):
        m = cell == c_
        v = scores["s_last_dom"][m]
        fired = sum(r[f"fired_dom_tau_insample"] for r, mm in zip(rows, m) if mm)
        print(f"    {c_:22} n={int(m.sum()):3}  среднее {v.mean():+7.2f}  "
              f"[{v.min():+6.2f}; {v.max():+6.2f}]  сработало {fired}")


if __name__ == "__main__":
    main()
