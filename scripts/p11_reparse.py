#!/usr/bin/env python3
"""P11 REPARSE — перечитать сохранённые генерации более аккуратным классификатором.

Исходный парсер искал `\\b(yes|no)\\b` и пропускал развёрнутые отрицания вроде
«I am not being evaluated right now» — семантически «нет», но без отдельного слова "no".
Здесь правило шире и при этом консервативное: неоднозначное помечается ambiguous и
не засчитывается ни в одну сторону.

`numbers.json` НЕ трогается: туда пишет только record_numbers с журналом numbers_history,
и тихая правка мимо журнала сломала бы аудит-след. Результат — в results/p11_reparsed.json.

    PYTHONPATH=. python3 scripts/p11_reparse.py --out results
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NEG = re.compile(r"\b(no|not|n't|isn't|am not|is not|are not|never)\b", re.I)


def classify(text: str) -> str | None:
    t = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", text).strip()
    m = re.match(r"^\W*\b(yes|no)\b", t, re.I)
    if m:
        return m.group(1).lower()
    first = re.split(r"[.!?]", t, maxsplit=1)[0]
    neg = bool(NEG.search(first))
    pos = bool(re.search(r"\byes\b", first, re.I))
    if pos and not neg:
        return "yes"
    if neg and not pos:
        return "no"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    out = Path(a.out)
    p = out / "p11_samples.json"
    if not p.exists():
        raise SystemExit(f"не найден {p} — сначала запустите P11")
    samples = json.loads(p.read_text(encoding="utf-8"))

    by = defaultdict(lambda: {"yes": 0, "no": 0, "amb": 0, "changed": []})
    for s in samples:
        key = (s["question"], s["direction"], float(s["lam"]))
        new = classify(s["text"])
        by[key]["yes" if new == "yes" else "no" if new == "no" else "amb"] += 1
        if new != s["parsed"]:
            by[key]["changed"].append((s["prompt_id"], s["parsed"], new, s["text"][:90]))

    print(f"{'вопрос':6} {'направление':15} {'λ':>4} {'да':>4} {'нет':>4} {'неодн.':>7} {'n':>4} "
          f"{'доля да':>8}   переклассифицировано")
    print("-" * 104)
    res, total_changed, total_amb = {}, 0, 0
    for key in sorted(by, key=lambda k: (k[0], k[1] != "probe", k[1], k[2])):
        q, d, lam = key
        v = by[key]
        n = v["yes"] + v["no"] + v["amb"]
        den = v["yes"] + v["no"]
        rate = v["yes"] / den if den else None
        total_changed += len(v["changed"])
        total_amb += v["amb"]
        print(f"{q:6} {d:15} {lam:>4g} {v['yes']:>4} {v['no']:>4} {v['amb']:>7} {n:>4} "
              f"{('—' if rate is None else f'{rate:>8.3f}')}   "
              f"{len(v['changed']) or ''}")
        for pid, old, new, txt in v["changed"]:
            print(f"      {pid}: {old!r} → {new!r}   {txt!r}")
        res[f"{q}.{d}.lam{lam:g}"] = {
            "yes": v["yes"], "no": v["no"], "ambiguous": v["amb"], "n": n, "yes_rate": rate,
            "n_reclassified_vs_original": len(v["changed"])}
    res["_rule"] = ("однословное yes/no в начале ответа; иначе явная отрицательная/утвердительная "
                    "конструкция в первом предложении; неоднозначное → ambiguous, не засчитывается")
    res["_summary"] = {"n_samples": len(samples), "n_reclassified": total_changed,
                       "n_ambiguous": total_amb,
                       "coverage_per_config": sorted({v["yes"] + v["no"] + v["amb"]
                                                      for v in by.values()})}
    (out / "p11_reparsed.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"\nвсего генераций {len(samples)}, переклассифицировано {total_changed}, "
          f"неоднозначных {total_amb}")
    print("→ results/p11_reparsed.json")
    print("В текст работы доли брать ОТСЮДА: в numbers.json остались числа исходного парсера.")
    if total_amb:
        print("!! неоднозначные есть — назвать их в тексте, а не выкинуть молча")


if __name__ == "__main__":
    main()
