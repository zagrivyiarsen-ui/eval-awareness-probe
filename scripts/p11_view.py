#!/usr/bin/env python3
"""P11 VIEW — разобрать результат стиринга. CPU, ничего не считает на модели.

Главный вопрос, на который отвечает этот просмотр: при λ = 8 модель ПЕРЕКЛЮЧИЛАСЬ на «да»
или РАЗВАЛИЛАСЬ, а слово «yes» попало в мусор? Парсер берёт первое yes/no в тексте и
различить это не умеет — различить может только глаз. Поэтому скрипт печатает сырые
генерации по ступеням λ рядом друг с другом.

    PYTHONPATH=. python3 scripts/p11_view.py --out results
    PYTHONPATH=. python3 scripts/p11_view.py --out results --question self --show 6
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

PRE = "probe.exploratory.p11."


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def clean(t: str) -> str:
    t = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", t)
    return re.sub(r"\s+", " ", t).strip()


def repeat_ratio(t: str) -> float:
    """Грубый индикатор вырождения: доля повторов среди слов. Аккуратный ответ ≈ 0.0–0.3,
    зацикленная генерация → к 1.0."""
    w = [x.lower() for x in re.findall(r"[a-z']+", t, flags=re.I)]
    return 0.0 if len(w) < 4 else 1 - len(set(w)) / len(w)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--question", default="self", help="self | main | all")
    ap.add_argument("--show", type=int, default=3, help="сколько генераций печатать на конфигурацию")
    ap.add_argument("--width", type=int, default=150)
    a = ap.parse_args(argv)
    out = Path(a.out)
    N = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    p11 = {k[len(PRE):]: v for k, v in N.items() if k.startswith(PRE)}
    if not p11:
        raise SystemExit("в numbers.json нет ключей probe.exploratory.p11.* — сначала запустите P11")

    print("=" * 100)
    print("1. ПРОВЕРКА ВМЕШАТЕЛЬСТВА")
    print("=" * 100)
    e1s = p11.get("e1_shift_unit_dir")
    mc = {k[len("manipulation_check."):]: v for k, v in p11.items()
          if k.startswith("manipulation_check.")}
    if mc:
        print(f"  хук доехал: {mc.get('hook_applied')} | макс. сдвиг скора {mc.get('max_score_shift')} "
              f"| макс. доля изменившихся текстов {mc.get('max_text_changed')}")
    else:
        print("  блок manipulation_check не сохранён (старая версия p11_steer) — считаю из таблицы")
    print(f"  сдвиг E1 вдоль единичного направления: {e1s}")
    print("  Проверка: у probe скор должен расти как λ·сдвиг, у random_control — стоять на месте.")

    # ── таблица
    lam_set = sorted({float(k.rsplit(".lam", 1)[1].split(".")[0])
                      for k in p11 if ".lam" in k and k.endswith(".yes_rate")})
    qs = sorted({k.split(".")[0] for k in p11 if k.endswith(".yes_rate")})
    show_q = qs if a.question == "all" else [a.question]
    print("\n" + "=" * 100)
    print("2. ДОЗА-ОТВЕТ  (доля «да» с Wilson 95 %; ожидаемый скор = скор при λ=0 + λ·сдвиг)")
    print("=" * 100)
    hdr = f"  {'вопрос':6} {'направление':15} {'λ':>4} {'да/n':>7} {'доля':>6} {'Wilson':>14} " \
          f"{'скор':>8} {'ожид.':>8} {'текст≠λ0':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for q in show_q:
        for d in ("probe", "random_control"):
            s0 = p11.get(f"{q}.probe.lam0.mean_score_at_last")
            for lam in lam_set:
                ls = f"{lam:g}"
                base = f"{q}.{d}.lam{ls}"
                if f"{base}.yes_rate" not in p11:
                    continue
                y, n = p11[f"{base}.yes"], p11[f"{base}.n"]
                unp = p11.get(f"{base}.unparsed", 0)
                den = y + p11[f"{base}.no"]
                lo, hi = wilson(y, den) if den else (float("nan"),) * 2
                sc = p11.get(f"{base}.mean_score_at_last")
                exp = (s0 + lam * e1s) if (s0 is not None and e1s and d == "probe") else s0
                ch = p11.get(f"{base}.text_changed_frac_vs_lam0")
                flag = "  ← НЕ РАЗОБРАНО" if unp == n else ""
                print(f"  {q:6} {d:15} {lam:>4g} {y:>3}/{den:<3} {(y/den if den else float('nan')):>6.3f} "
                      f"[{lo:.2f};{hi:.2f}]".ljust(0) +
                      f" {sc:>8.2f} {('—' if exp is None else f'{exp:>8.2f}')} "
                      f"{('—' if ch is None else f'{ch:>8.0%}')}{flag}")

    # ── сырые генерации
    sp = out / "p11_samples.json"
    print("\n" + "=" * 100)
    print("3. СЫРЫЕ ГЕНЕРАЦИИ — СМОТРЕТЬ ГЛАЗАМИ")
    print("=" * 100)
    if not sp.exists():
        print("  results/p11_samples.json не найден")
        return
    samples = json.loads(sp.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for s in samples:
        by[(s["question"], s["direction"], float(s["lam"]))].append(s)
    for q in show_q:
        for d in ("probe", "random_control"):
            print(f"\n── {q} · {d} " + "─" * 60)
            for lam in lam_set:
                items = by.get((q, d, lam), [])
                if not items:
                    continue
                rr = [repeat_ratio(clean(i["text"])) for i in items]
                mark = "  ⚠ похоже на вырождение" if (rr and max(rr) > 0.5) else ""
                print(f"  λ = {lam:g}   (повторяемость слов max {max(rr) if rr else 0:.2f}){mark}")
                for i in items[:a.show]:
                    t = clean(i["text"])[:a.width]
                    print(f"     [{i['parsed'] or '—'}] {i['prompt_id']}: {t}")

    print("\n" + "=" * 100)
    print("4. ЧТО РЕШИТЬ, ГЛЯДЯ НА ЭТО")
    print("=" * 100)
    print("""  a) при λ, где доля «да» переключилась: ответы связные и по делу — или это уже мусор,
     в котором парсер нашёл слово yes? Второе НЕ является каузальным эффектом.
  b) random_control при том же λ: если он ломается (не разобрано / повторы), а probe даёт
     связное «да» — специфичность направления показана; если ломаются оба, это эффект нормы.
  c) при каком λ эффект появляется. λ = 1 означает «столько же, сколько даёт настоящая
     вставка». Переключение только при λ ≫ 1 — это НЕ «направление кодирует понятие»,
     а «достаточно большой толчок вдоль него меняет ответ».
  d) пол и потолок: если при λ=0 доля уже 0/16 или 16/16, то в одну сторону запаса нет
     и «не изменилось» в эту сторону ничего не значит.""")


if __name__ == "__main__":
    main()
