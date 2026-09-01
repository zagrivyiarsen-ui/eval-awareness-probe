#!/usr/bin/env python3
"""P9 — случайная выборка сырых промптов для текста работы.

Формат заявки требует прямо: «If bad data would sink your project, show me the data —
randomly selected, not cherry-picked», сразу после executive summary. Работа целиком стоит
на промптах, написанных автором, и на фильтре Dolly; ни одного показанного промпта в пакете
нет. Скрипт печатает выборку ПО СИДУ, стратифицированную так, чтобы читатель увидел:

  * обучающий контраст целиком (deployment и E1 одного и того же DP — видно ровно вставку),
  * один сюжет тематической тройки a1 / a2 / a_head (лексика против типа просьбы),
  * один сработавший и один несработавший внешний промпт Dolly,
  * один нейтральный.

Отбор внутри каждой страты — random.choice с фиксированным сидом, без повторных тяг.

    PYTHONPATH=. python3 scripts/p9_qualitative.py --out results --seed 20260903
→ results/qualitative_prompts.md (+ .json со списком выбранных ключей)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--max-chars", type=int, default=2600)
    ap.add_argument("--artifact", default="results/p4_frozen.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    rng = random.Random(a.seed)

    texts = {}
    for line in open(out / "records_texts.jsonl", encoding="utf-8"):
        d = json.loads(line)
        texts[d["prompt_key"]] = d["text"]
    rows = {r["prompt_key"]: r for r in csv.DictReader(open(out / "records_full.csv", encoding="utf-8"))}

    def pick(pred):
        pool = sorted(k for k, r in rows.items() if pred(r) and k in texts)
        return rng.choice(pool) if pool else None

    picked: list[tuple[str, str]] = []

    # 1. пара обучающего контраста: один и тот же DP, deployment и E1
    dep_keys = sorted(k for k, r in rows.items()
                      if r["cell"] == "deployment" and r["is_train_contrast"] == "1" and k in texts)
    if dep_keys:
        k_dep = rng.choice(dep_keys)
        env, pid, _ = k_dep.split("/")
        k_e1 = f"{env}/{pid}/E1"
        picked.append(("обучающий негатив (deployment)", k_dep))
        if k_e1 in texts:
            picked.append(("тот же промпт + вставка 13 токенов (E1) — ВЕСЬ обучающий контраст", k_e1))

    # 2. тематическая тройка одного сюжета
    art = Path(a.artifact)
    if art.exists():
        sm = json.loads(art.read_text(encoding="utf-8")).get("story_map", {})
        if sm:
            story = rng.choice(sorted(sm))
            ids = sm[story]
            names = ("a1 — оценочная тематика, богатая лексика",
                     "a2 — тот же сюжет, вся зарегистрированная лексика снята",
                     "a_head — тот же сюжет, eval-фон, просьба «напиши письмо»")
            for pid_, nm in zip(ids if isinstance(ids, (list, tuple)) else [], names):
                hit = [k for k in texts if k.endswith("/" + str(pid_)) or f"/{pid_}/" in k]
                if hit:
                    picked.append((nm, sorted(hit)[0]))

    # 3. внешние Dolly: сработавший и несработавший
    k = pick(lambda r: r["cell"] == "wild_other" and r["fired_dom_tau_insample"] == "1")
    if k:
        picked.append(("внешний Dolly, СРАБОТАЛ (ложное срабатывание монитора)", k))
    k = pick(lambda r: r["cell"] == "wild_other" and r["fired_dom_tau_insample"] == "0")
    if k:
        picked.append(("внешний Dolly, не сработал", k))

    # 4. нейтральный авторский
    k = pick(lambda r: r["cell"] == "n")
    if k:
        picked.append(("нейтральный авторский промпт", k))

    L = [f"# Случайно выбранные промпты (seed = {a.seed})", "",
         "Отбор: `random.choice` внутри каждой страты, сид зафиксирован в команде запуска, "
         "повторных тяг нет. Скор — diff-of-means, L21, позиция last; «сработал» — скор ≥ τ, "
         "где τ = 5 % квантиль скоров E1.", ""]
    for title, key in picked:
        r = rows.get(key, {})
        s = r.get("s_last_dom", "?")
        fired = "ДА" if r.get("fired_dom_tau_insample") == "1" else "нет"
        t = texts[key]
        cut = len(t) > a.max_chars
        L += [f"## {title}", "",
              f"`{key}` · ячейка `{r.get('cell','?')}` · скор **{s}** · сработал: **{fired}** · "
              f"слов {r.get('n_words','?')}", "", "```", t[:a.max_chars] +
              ("\n…[обрезано для показа]" if cut else ""), "```", ""]
    (out / "qualitative_prompts.md").write_text("\n".join(L), encoding="utf-8")
    (out / "qualitative_prompts.json").write_text(
        json.dumps({"seed": a.seed, "picked": [{"why": w, "prompt_key": k} for w, k in picked]},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ results/qualitative_prompts.md ({len(picked)} промптов, seed {a.seed})")
    for w, k in picked:
        print(f"   {k:44} {w}")


if __name__ == "__main__":
    main()
