#!/usr/bin/env python3
"""P11 BROWSE — прямой просмотр results/p11_samples.json, без пересчёта чего-либо.

Отличие от p11_view.py: там сырые генерации подмешаны к таблице доза-ответ и обрезаны до
--show строк на конфигурацию. Здесь — просто файл целиком, с фильтрами и в удобном виде,
когда нужно прочитать ВСЕ 16 ответов на конкретном λ, а не первые несколько.

Примеры:
    # всё подряд, сгруппировано по (вопрос, направление, λ)
    PYTHONPATH=. python3 scripts/p11_browse.py --out results

    # только self/probe при λ=8 — тот самый пограничный случай
    PYTHONPATH=. python3 scripts/p11_browse.py --out results --question self --direction probe --lam 8

    # только ответы, распознанные как "yes"
    PYTHONPATH=. python3 scripts/p11_browse.py --out results --parsed yes

    # текстовый поиск по содержимому ответа
    PYTHONPATH=. python3 scripts/p11_browse.py --out results --grep "not being evaluated"

    # как таблица (без текста, для быстрого обзора)
    PYTHONPATH=. python3 scripts/p11_browse.py --out results --format table

    # сохранить отфильтрованное в отдельный markdown-файл, чтобы читать не в логе
    PYTHONPATH=. python3 scripts/p11_browse.py --out results --md results/p11_browse.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def clean(t: str) -> str:
    t = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", t)
    return re.sub(r"\s+", " ", t).strip()


def repeat_ratio(t: str) -> float:
    w = [x.lower() for x in re.findall(r"[a-z']+", t, flags=re.I)]
    return 0.0 if len(w) < 4 else 1 - len(set(w)) / len(w)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--file", default=None, help="путь к p11_samples.json (по умолчанию --out/p11_samples.json)")
    ap.add_argument("--question", default=None, help="self | main | ...")
    ap.add_argument("--direction", default=None, help="probe | random_control")
    ap.add_argument("--lam", default=None, help="конкретное λ, например 8")
    ap.add_argument("--parsed", default=None, help="yes | no | none  (none = не распознано)")
    ap.add_argument("--grep", default=None, help="подстрока/regex по тексту ответа, без учёта регистра")
    ap.add_argument("--prompt-id", default=None)
    ap.add_argument("--format", choices=["full", "table"], default="full")
    ap.add_argument("--width", type=int, default=220, help="обрезка текста при --format full")
    ap.add_argument("--limit", type=int, default=0, help="0 = без ограничения")
    ap.add_argument("--md", default=None, help="если задано — вывод пишется в этот файл, а не в stdout")
    a = ap.parse_args(argv)

    path = Path(a.file) if a.file else Path(a.out) / "p11_samples.json"
    if not path.exists():
        raise SystemExit(f"нет файла {path} — сначала запустите P11 (scripts/p11_steer.py)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise SystemExit(f"{path} пуст")

    rows = []
    for s in data:
        s = dict(s)
        s["text_clean"] = clean(s.get("text", ""))
        s["repeat_ratio"] = round(repeat_ratio(s["text_clean"]), 2)
        rows.append(s)

    def keep(s):
        if a.question and s.get("question") != a.question:
            return False
        if a.direction and s.get("direction") != a.direction:
            return False
        if a.lam is not None and float(s.get("lam", -1)) != float(a.lam):
            return False
        if a.prompt_id and s.get("prompt_id") != a.prompt_id:
            return False
        if a.parsed is not None:
            want = None if a.parsed.lower() == "none" else a.parsed.lower()
            if s.get("parsed") != want:
                return False
        if a.grep and not re.search(a.grep, s["text_clean"], flags=re.I):
            return False
        return True

    rows = [s for s in rows if keep(s)]
    if not rows:
        raise SystemExit("фильтры не оставили ни одной записи — ослабьте условия")
    if a.limit:
        rows = rows[: a.limit]

    out_lines: list[str] = []

    def w(line=""):
        out_lines.append(line)

    w(f"# p11_samples.json — {len(rows)} из {len(data)} записей после фильтров")
    w(f"источник: {path}")
    if any([a.question, a.direction, a.lam, a.parsed, a.grep, a.prompt_id]):
        w("фильтры: " + ", ".join(
            f"{k}={v}" for k, v in
            (("question", a.question), ("direction", a.direction), ("lam", a.lam),
             ("parsed", a.parsed), ("grep", a.grep), ("prompt_id", a.prompt_id)) if v))
    w()

    if a.format == "table":
        w(f"{'question':8} {'direction':15} {'λ':>4} {'prompt_id':10} {'parsed':6} {'rep':>5}  text")
        w("-" * 120)
        for s in rows:
            t = s["text_clean"][: a.width - 60]
            w(f"{s.get('question',''):8} {s.get('direction',''):15} {float(s.get('lam',0)):>4g} "
              f"{s.get('prompt_id',''):10} {str(s.get('parsed')):6} {s['repeat_ratio']:>5.2f}  {t}")
    else:
        groups = defaultdict(list)
        for s in rows:
            groups[(s.get("question"), s.get("direction"), float(s.get("lam", 0)))].append(s)
        for (q, d, lam), items in sorted(groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "", kv[0][2])):
            rr = [i["repeat_ratio"] for i in items]
            mark = "  ⚠ похоже на вырождение (повторы слов)" if rr and max(rr) > 0.5 else ""
            w(f"## {q} · {d} · λ = {lam:g}   ({len(items)} ответов){mark}")
            w()
            for s in items:
                t = s["text_clean"]
                if len(t) > a.width:
                    t = t[: a.width] + "…"
                w(f"- **[{s.get('parsed') or '—'}]** `{s.get('prompt_id')}` (повтор {s['repeat_ratio']:.2f}): {t}")
            w()

    text = "\n".join(out_lines)
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")
        print(f"записано: {a.md} ({len(rows)} записей)")
    else:
        print(text)


if __name__ == "__main__":
    main()
