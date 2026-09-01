#!/usr/bin/env python3
"""P10 — аудит единиц, процедур и полов p. Ничего не пересчитывает по активациям.

Три вещи, на которых работу ловит внимательный рецензент:

1. ЕДИНИЦЫ. В снимке сосуществуют нормированная (‖w‖=1) и ненормированная шкалы, а внутри
   нормированной — сырые сдвиги (`.mean`) и доли (`frac_of_e1`). В сводке блок v2 подан
   вперемешку: «frac 0.307 / 0.277 … PC−sys_neutral −0.147 … PC_strong−sys_neutral +1.210».
   Первые два — доли, вторые два — сырые единицы. Читатель, который вычтет 0.547 − 0.307 = 0.240
   и увидит рядом «+1.210», перестанет доверять остальным числам. Скрипт печатает блок
   в ОБЕИХ шкалах с явными подписями.

2. ПРОЦЕДУРЫ. В главной таблице колонка FPR — это `fpr_loo` (среднее по 32 LOO-порогам),
   а колонка FP и Wilson — in-sample счёт. Отсюда строка `c1: FPR 0.727 при FP 3/4`.
   Арифметика верна, но подпись отсутствует. Скрипт печатает таблицу с разделёнными колонками.

3. ПОЛЫ p. Перестановочный p имеет пол 1/(B+1); точный знаковый — 2^(1−n). Форматирование
   дампа (3 знака) превращает 0.0002 в «0». Скрипт находит все такие ключи и даёт
   формулировку «p < 1/(B+1)» вместо нуля.

    PYTHONPATH=. python3 scripts/p10_audit.py --out results
→ results/AUDIT_units_and_procedures.md, results/NUMBERS_FOR_WRITEUP.md, probe.exploratory.p10.*
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_here.parent if _here.name == "scripts" else _here))

CELLS = ["a_legacy", "d_legacy", "a1", "a2", "a_head", "a_tail", "d", "b", "c1", "c2",
         "n", "dptrunc", "len140", "len210", "wild_other"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def fmt_p(p: float | None, floor: float | None) -> str:
    """Никогда не печатать p = 0. Пол теста — часть утверждения."""
    if p is None or p != p:
        return "—"
    if floor is not None and p <= floor * 1.001 + 1e-9:
        return f"= пол теста ({floor:.3g}); писать «все дельты одного знака, p на полу»"
    if p < 1e-3:
        return f"< {max(p, 1e-4):.1g}"
    return f"= {p:.4g}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-perm", type=int, default=5000, help="B в перестановочных тестах probe_stage")
    a = ap.parse_args(argv)
    out = Path(a.out)
    N = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    A = {}

    # ── 1. единицы: блок v2_deconstruction
    b = "probe.exploratory.v2_deconstruction"
    dE1 = N.get(f"{b}.delta_E1")
    rows_units = []
    if dE1:
        for k in ("sys_neutral", "positive_control", "positive_control_strong",
                  "positive_control_minus_sys_neutral",
                  "positive_control_strong_minus_sys_neutral",
                  "positive_control_strong_minus_positive_control"):
            m = N.get(f"{b}.{k}.mean")
            if m is None:
                continue
            rows_units.append({"key": k, "mean_normed_units": m, "frac_of_e1": m / dE1,
                               "p": N.get(f"{b}.{k}.p"), "n": N.get(f"{b}.{k}.n")})
    A["units.delta_E1_normed"] = dE1
    A["units.v2_deconstruction_in_both_scales"] = {
        r["key"]: {"mean_normed_units": r["mean_normed_units"], "frac_of_e1": r["frac_of_e1"]}
        for r in rows_units}
    A["units.rule"] = ("frac = mean / delta_E1. В тексте работы приводить ТОЛЬКО frac либо "
                       "ТОЛЬКО сырые единицы, но не вперемешку в одном списке.")

    # ── 2. процедуры: FPR_loo vs FPR_insample по ячейкам
    rows_proc = []
    csv_path = out / "records_full.csv"
    counts = {}
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path, encoding="utf-8")):
            c = r["cell"]
            counts.setdefault(c, [0, 0])
            counts[c][1] += 1
            if r.get("fired_dom_tau_insample") == "1":
                counts[c][0] += 1
    for c in CELLS:
        k, n = counts.get(c, (None, None))
        fpr_loo = N.get(f"probe.exploratory.p2.cells_last.{c}.fpr_loo")
        if fpr_loo is None:
            fpr_loo = N.get(f"probe.exploratory.p4.b2_bands.{c}.fpr_loo")
        rows_proc.append({"cell": c, "n": n, "fp_at_tau_insample": k,
                          "fpr_insample": (k / n) if (k is not None and n) else None,
                          "wilson95_insample": wilson(k, n) if (k is not None and n) else None,
                          "fpr_loo_mean_over_32_taus": fpr_loo,
                          "mismatch": (k is not None and n and fpr_loo is not None
                                       and abs(k / n - fpr_loo) > 1e-9)})
    A["procedures.cells"] = {r["cell"]: {k: v for k, v in r.items() if k != "cell"}
                             for r in rows_proc}
    A["procedures.rule"] = ("fpr_loo = среднее доли выше порога по 32 LOO-порогам; "
                            "fp_at_tau_insample и Wilson относятся к ЕДИНСТВЕННОМУ in-sample "
                            "порогу. Это разные процедуры: в одной строке таблицы подписывать обе.")

    # ── 3. полы p
    floors = []
    for k, v in N.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        leaf = k.rsplit(".", 1)[-1]
        if leaf not in ("p", "p_perm", "p_sign_perm", "p_sign_perm_exact", "p_one_sided", "p_fisher_one_sided"):
            continue
        n = N.get(k.rsplit(".", 1)[0] + ".n")
        if leaf in ("p_sign_perm", "p_sign_perm_exact") and isinstance(n, (int, float)) and n:
            fl = 2.0 ** (1 - int(n))
        elif leaf in ("p", "p_perm"):
            fl = 1.0 / (a.n_perm + 1)
        else:
            fl = None
        floors.append({"key": k, "p": v, "floor": fl, "at_floor": bool(fl and v <= fl * (1 + 1e-9)),
                       "prints_as_zero_at_3dp": bool(v < 5e-4), "report_as": fmt_p(v, fl)})
    A["p_floors.keys_at_floor"] = sorted(x["key"] for x in floors if x["at_floor"])
    A["p_floors.keys_printed_as_zero"] = sorted(x["key"] for x in floors
                                                if x["prints_as_zero_at_3dp"])
    A["p_floors.summary"] = {
        "n_keys": len(floors),
        "n_at_floor": sum(1 for x in floors if x["at_floor"]),
        "n_printed_as_zero": sum(1 for x in floors if x["prints_as_zero_at_3dp"]),
        "rule": ("перестановочный p считается как (k+1)/(B+1) — нуля там быть не может; "
                 f"«0» в дампе — округление до 3 знаков. Пол при B={a.n_perm}: {1/(a.n_perm+1):.3g}. "
                 "В тексте писать «p < пол», а не «p = 0».")}

    # ── запись
    from src.contracts import record_numbers
    record_numbers(out / "numbers.json", {"probe.exploratory.p10": A})

    # ── отчёт-аудит
    L = ["# AUDIT — единицы, процедуры, полы p", "",
         "Три места, где текст работы сейчас уязвим, и готовые исправленные формулировки.", "",
         "## 1. Единицы в блоке деконструкции позитивного контроля", ""]
    if rows_units:
        L += [f"delta_E1 (нормированная шкала) = **{dE1:.4g}** — это знаменатель frac.", "",
              "| величина | сырые нормированные единицы | доли E1 (frac) | n | p |",
              "|---|---|---|---|---|"]
        for r in rows_units:
            L.append(f"| {r['key']} | {r['mean_normed_units']:+.4g} | {r['frac_of_e1']:+.4g} | "
                     f"{r['n']} | {fmt_p(r['p'], 1.0/(a.n_perm+1))} |")
        L += ["", "**Как писать в тексте (одна шкала на список):**", "",
              "> Обычный позитивный контроль не превышает нейтральный system-ход: разность "
              f"{[r for r in rows_units if r['key']=='positive_control_minus_sys_neutral'][0]['frac_of_e1']:+.3f} "
              "в долях сдвига E1. Содержание добавляет только усиленная версия: "
              f"{[r for r in rows_units if r['key']=='positive_control_strong_minus_sys_neutral'][0]['frac_of_e1']:+.3f} E1.",
              "", "**НЕ писать** «PC_strong − sys_neutral = +1.210» рядом с «frac 0.547»: "
              "это одна и та же величина в двух шкалах, и читатель поймает противоречие "
              "(0.547 − 0.307 = 0.240, а не 1.210).", ""]
    else:
        L += ["(ключей блока v2_deconstruction нет в снимке)", ""]

    L += ["## 2. Две процедуры FPR в одной таблице", "",
          "| ячейка | n | FP при in-sample τ | FPR in-sample | Wilson (in-sample) | FPR LOO (среднее по 32 τ) | расходятся |",
          "|---|---|---|---|---|---|---|"]
    for r in rows_proc:
        w = r["wilson95_insample"]
        n_s = "—" if r["n"] is None else str(r["n"])
        fp_s = "—" if r["fp_at_tau_insample"] is None else str(r["fp_at_tau_insample"])
        fi_s = "—" if r["fpr_insample"] is None else f"{r['fpr_insample']:.3f}"
        w_s = "—" if w is None else f"[{w[0]:.2f}; {w[1]:.2f}]"
        fl_s = "—" if r["fpr_loo_mean_over_32_taus"] is None else f"{r['fpr_loo_mean_over_32_taus']:.3f}"
        L.append(f"| {r['cell']} | {n_s} | {fp_s} | {fi_s} | {w_s} | {fl_s} | "
                 f"{'да' if r['mismatch'] else ''} |")
    L += ["", "Строки с «да» — те, где рецензент вычтет и не сойдётся (например c1: 0.727 против 3/4). "
              "Это не ошибка счёта, а отсутствующая подпись: усреднение по 32 LOO-порогам даёт "
              "значения, не кратные 1/n.", ""]

    L += ["## 3. Полы точных и перестановочных тестов", "",
          f"Перестановочный p = (k+1)/(B+1), B = {a.n_perm} → пол {1/(a.n_perm+1):.3g}. "
          "Нуля там быть не может; «0» в дампе — округление до трёх знаков.", "",
          "| ключ | p в снимке | пол | как писать |", "|---|---|---|---|"]
    for r in sorted(floors, key=lambda x: x["p"])[:40]:
        fl_s = "—" if r["floor"] is None else f"{r['floor']:.3g}"
        L.append(f"| `{r['key']}` | {r['p']:.3g} | {fl_s} | {r['report_as']} |")
    L += ["", f"Всего ключей с p: {len(floors)}; на полу: {A['p_floors.summary']['n_at_floor']}; "
              f"печатаются как «0» при 3 знаках: {A['p_floors.summary']['n_printed_as_zero']}.", ""]
    (out / "AUDIT_units_and_procedures.md").write_text("\n".join(L), encoding="utf-8")

    # ── короткий список чисел, безопасных для цитирования
    W = ["# Числа, безопасные для цитирования в тексте", "",
         "Сгенерировано p10_audit.py. Каждая строка — величина, шкала, процедура и оговорка.", ""]
    def g(k, default=None):
        return N.get(k, default)
    W += ["| число | значение | шкала / процедура | оговорка |", "|---|---|---|---|",
          f"| held-out AUC E1 vs deployment | {g('probe.v1.heldout_auc_pool')} | 4 фолда по промптам | единственная пройденная валидация |",
          f"| TPR при τ (LOO) | {g('probe.exploratory.precision.tpr_loo')} | τ = 5 % квантиль E1 | ≈0.95 ПО ПОСТРОЕНИЮ, не результат |",
          f"| FPR тематических (LOO) | {g('probe.exploratory.p2.precision.fpr_topical')} | ведущая OOS | n=30 |",
          f"| FPR внешних Dolly | {g('probe.exploratory.p4.b6_precision_dolly.external_dolly.fpr_insample')} | in-sample τ | n=37, Wilson обязателен |",
          f"| precision при 1 % | {g('probe.exploratory.p2.precision.precision_at_1pct_q1', g('probe.exploratory.precision.precision_at_1pct_fpr_topic'))} | q=1 %, FPR тематических | bootstrap CI широкий |",
          f"| frac e2_c | {g('probe.v5.e2_c.pool.frac_of_e1')} | нормированная, доли E1 | перефразировка в позиции E1 |",
          f"| frac e2_b | {g('probe.v5.e2_b.pool.frac_of_e1')} | нормированная, доли E1 | тот же текст, другое место |",
          f"| frac e2_a | {g('probe.v5.e2_a.pool.frac_of_e1')} | нормированная, доли E1 | тот же текст, другое место/роль |",
          f"| AUC скор vs ответ модели (self) | {g('probe.exploratory.p4.b7_item_level.self.auc_within_cell')} | within-cell (Mantel–Haenszel) | наивный пул вводит в заблуждение |",
          f"| AUC скор vs ответ модели (main) | {g('probe.exploratory.p4.b7_item_level.main.auc_within_cell')} | within-cell | 183 пары, 7 ячеек |",
          ]
    (out / "NUMBERS_FOR_WRITEUP.md").write_text("\n".join(W), encoding="utf-8")

    print("\n".join(L[:60]))
    print("\n→ results/AUDIT_units_and_procedures.md")
    print("→ results/NUMBERS_FOR_WRITEUP.md")
    print("→ probe.exploratory.p10.*")


if __name__ == "__main__":
    main()
