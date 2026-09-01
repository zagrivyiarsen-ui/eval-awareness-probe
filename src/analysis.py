"""Анализ по configs/analysis.yaml: джойн плеч (192 строк), модель SD(Δ) и MDE, правило урезания (В.1–5),
границы для обрыва, code_sha256. Никаких S-результатов здесь не читается до заморозки пробе — это чистые функции."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import NormalDist, mean, pstdev
from typing import Any, Iterable, Sequence

from src.conditions import MAIN_IDS

Z_POWER = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)


# ---------------------------------------------------------------- code hash (Kaggle без git)
def code_sha256(root: Path | str) -> str:
    root = Path(root)
    files = sorted([*root.glob("src/*.py"), *root.glob("scripts/*.py"), root / "configs" / "conditions.yaml",
                    root / "configs" / "analysis.yaml"])
    h = hashlib.sha256()
    for f in files:
        if f.exists():
            h.update(str(f.relative_to(root)).encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- джойн плеч
def behaviour_by_prompt_key(gens: Iterable[dict[str, Any]], target_field: str = "target",
                            complete_case: bool = True) -> dict[str, dict[str, Any]]:
    """prompt_key → {rate, n, n_truncated, n_unparsable, lower, upper}. complete_case: ok-генерации с target∈{0,1}.
    lower/upper — границы: truncated/unparsable = no-target / = target (analysis.yaml truncation.secondary_bounds)."""
    acc: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "hit": 0, "excl": 0})
    for g in gens:
        a = acc[g["prompt_key"]]
        a["n"] += 1
        t = (g.get("score") or {}).get(target_field)
        usable = g.get("cot_category") == "ok" and t is not None
        if usable:
            a["hit"] += int(bool(t))
        else:
            a["excl"] += 1
    out = {}
    for k, a in acc.items():
        n_ok = a["n"] - a["excl"]
        out[k] = {"n": a["n"], "n_ok": n_ok, "n_excluded": a["excl"],
                  "rate": a["hit"] / n_ok if n_ok else None,
                  "lower": a["hit"] / a["n"], "upper": (a["hit"] + a["excl"]) / a["n"]}
    return out


def join_arms(beh: dict[str, dict[str, Any]], probe: dict[str, float], n_prompts: int = 32,
              conds: Sequence[str] = MAIN_IDS) -> list[dict[str, Any]]:
    """Одна строка на prompt_key (env/prompt_id/condition). Пробе-скор детерминирован — без sample_idx.
    Ассерт: строк = n_prompts × len(conds) (192), ключи не размножены."""
    keys = sorted(set(beh) | set(probe))
    keys = [k for k in keys if k.split("/")[-1] in conds]
    rows = [{"prompt_key": k, "env": k.split("/")[0], "prompt_id": k.split("/")[1], "condition_id": k.split("/")[2],
             "beh_rate": (beh.get(k) or {}).get("rate"), "beh_lower": (beh.get(k) or {}).get("lower"),
             "beh_upper": (beh.get(k) or {}).get("upper"), "probe_score": probe.get(k)} for k in keys]
    assert len(rows) == n_prompts * len(conds), f"join produced {len(rows)} rows, expected {n_prompts * len(conds)}"
    assert len({r["prompt_key"] for r in rows}) == len(rows)
    return rows


def paired_deltas(rows: list[dict[str, Any]], field: str, base: str = "deployment") -> dict[str, dict[str, float | None]]:
    """env/prompt_id → {condition: value − value(deployment)}."""
    by = defaultdict(dict)
    for r in rows:
        by[f"{r['env']}/{r['prompt_id']}"][r["condition_id"]] = r[field]
    out = {}
    for ep, d in by.items():
        b = d.get(base)
        out[ep] = {c: (None if (b is None or v is None) else v - b) for c, v in d.items() if c != base}
    return out


# ---------------------------------------------------------------- MDE / SD(Δ)
def sd_delta_model(p: float, k: int, sigma_between: float) -> float:
    """SD(Δ)² = 2p(1−p)/k + σ²_between (analysis.yaml cut_rule.sd_delta_model)."""
    return math.sqrt(2 * p * (1 - p) / k + sigma_between ** 2)


def mde(sd_delta: float, n_prompts: int) -> float:
    return Z_POWER * sd_delta / math.sqrt(n_prompts)


def estimate_between_sigma(deltas: Sequence[float], p: float, k: int, ci_upper_q: float = 0.75) -> dict[str, float]:
    """Из наблюдённых дельт при k: σ²_obs − 2p(1−p)/k = σ²_between (≥0). Верхняя граница по χ² (q=0.75) для выбора k."""
    n = len(deltas)
    var_obs = pstdev(deltas) ** 2 * n / max(1, n - 1)
    cell = 2 * p * (1 - p) / k
    sb2 = max(0.0, var_obs - cell)
    # грубая верхняя граница дисперсии: var_obs * (n−1)/χ²_{1−q}(n−1) через нормальную аппроксимацию χ²
    chi_lo = max(1e-6, (n - 1) * (1 - 2 / (9 * (n - 1)) - NormalDist().inv_cdf(ci_upper_q) * math.sqrt(2 / (9 * (n - 1)))) ** 3)
    var_up = var_obs * (n - 1) / chi_lo
    return {"n": n, "sd_obs": math.sqrt(var_obs), "sigma_between": math.sqrt(sb2),
            "sigma_between_upper": math.sqrt(max(0.0, var_up - cell)), "cell_var_at_k": cell}


# ---------------------------------------------------------------- бюджет и правило урезания (В.1–5)
def hours_for(n_gen: int, cap: int, tok_s: float, batch_size: int = 24, overhead_per_session: float = 1.0,
              session_hours: float = 11.0) -> float:
    raw = math.ceil(n_gen / batch_size) * batch_size * cap / tok_s / 3600
    return raw + overhead_per_session * math.ceil(raw / session_hours)


def gpu_hours_available(today: date, deadline: date, hours_per_week: float, hours_used_this_week: float,
                        safety_days: int = 2) -> float:
    """Квота Kaggle сбрасывается по субботам (UTC). Считаем до deadline − safety_days включительно."""
    end = date.fromordinal(deadline.toordinal() - safety_days)
    if end < today:
        return 0.0
    total = 0.0
    d = today
    week_left = max(0.0, hours_per_week - hours_used_this_week)
    while d <= end:
        if d != today and d.weekday() == 5:      # суббота — новая неделя
            week_left = hours_per_week
        # доля недели, оставшаяся до end, ограничивает только последнюю неделю
        nxt = date.fromordinal(d.toordinal() + 1)
        if nxt > end or nxt.weekday() == 5:
            total += week_left; week_left = 0.0
        d = nxt
    return total


def choose_cap(len_by_env_cond: dict[str, list[int]], candidates: Sequence[int], truncated_max: float) -> int:
    """cap = min из candidates, при котором доля длин > cap ≤ truncated_max во ВСЕХ ячейках env×cond пилота."""
    for cap in sorted(candidates):
        if all((sum(L > cap for L in ls) / len(ls) if ls else 1.0) <= truncated_max for ls in len_by_env_cond.values()):
            return cap
    return max(candidates)


def choose_k(n_prompts: int, cap: int, tok_s: float, hours_available: float, k_candidates: Sequence[int],
             n_pc: int = 0, batch_size: int = 24, overhead: float = 1.0, n_done_one_env: int = 0) -> dict[str, Any]:
    """max k с H ≤ available для n_prompts (обе среды); если k=min не помещается — одна среда, k=15 (env_rule).
    n_done_one_env (2026-08-28): пилотные deployment/E1-генерации выбранной среды, засчитанные в main (addendum) — вычитаются
    из n_gen одной среды; они не «бесплатны» в двух-средовой ветке (там среда ещё не выбрана)."""
    table = {}
    for k in sorted(k_candidates, reverse=True):
        if k < 10:
            break                                            # две среды только при k ≥ 10 (пул n=32 держит MDE≈0.10)
        n_gen = n_prompts * 6 * k + n_pc
        table[f"k{k}"] = hours_for(n_gen, cap, tok_s, batch_size, overhead)
        if table[f"k{k}"] <= hours_available:
            return {"k": k, "n_env": 2, "hours": table[f"k{k}"], "table": table, "rule": "two envs"}
    n1 = n_prompts // 2
    for k in (15, 10, 6):
        h1 = hours_for(max(0, n1 * 6 * k + n_pc - n_done_one_env), cap, tok_s, batch_size, overhead)
        table[f"ONE_ENV.k{k}"] = h1
        if h1 <= hours_available:
            table["n_done_one_env_subtracted"] = n_done_one_env
            return {"k": k, "n_env": 1, "hours": h1, "table": table, "rule": f"one env (env_rule), k={k}", "fits": True}
    table["n_done_one_env_subtracted"] = n_done_one_env
    return {"k": 6, "n_env": 1, "hours": table["ONE_ENV.k6"], "table": table, "rule": "NOTHING FITS — extension_required",
            "fits": False, "extension_required": True}


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ---------------------------------------------------------------- статистика без torch (поведенческое плечо, джойн)
import random as _random


def paired_perm_test_py(d: Sequence[float], n_perm: int = 5000, seed: int = 0) -> dict[str, float]:
    """Парная перестановка знаков на промптовых дельтах (план §4.6 — основной тест). Двусторонний p."""
    d = [x for x in d if x is not None]
    n = len(d)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0, "p": None}
    rng = _random.Random(seed)
    m = sum(d) / n
    obs = abs(m)
    cnt = 0
    for _ in range(n_perm):
        s = sum(x if rng.random() < 0.5 else -x for x in d) / n
        cnt += abs(s) >= obs
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else None
    return {"mean": m, "sd": sd, "n": n, "p": (cnt + 1) / (n_perm + 1), "se": (sd / n ** 0.5) if sd is not None else None}


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5; sy = sum((b - my) ** 2 for b in ys) ** 0.5
    return sum((a - mx) * (b - my) for a, b in pairs) / (sx * sy) if sx and sy else None


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    xs, ys = zip(*pairs)
    return pearson(ranks(list(xs)), ranks(list(ys)))


def recut_to_cap(g: dict[str, Any], cap: int) -> dict[str, Any]:
    """Пилотные генерации при cap 4096 → при меньшем основном cap: длина > cap ⇒ truncated (та же T/sampling → распределение то же).
    2026-08-28 (ревью R-Launch Д-1): длина тоже обрезается до cap — иначе cot_len_mean/cot_len_delta по условиям смещены
    (пере-обрезанные deployment/E1 несли бы длины до 4096 против S-условий, сгенерированных под cap)."""
    if g.get("n_generated_tokens", 0) > cap:
        g = dict(g)
        if g.get("cot_category") != "truncated":
            g["cot_category"] = "truncated"; g["recut_from_cap"] = True; g["score"] = {"class": "unparsable", "target": None}
        g["n_generated_tokens_precut"] = g["n_generated_tokens"]; g["n_generated_tokens"] = cap
    return g


def trim_to_equal_k(gens: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """2026-08-28 (ревью R-Launch, правило k-недобора, addendum): при незавершённом main анализ идёт на достигнутом
    РАВНОМ k′ = max s: sample_idx 0..s−1 присутствует во ВСЕХ ячейках prompt_key; хвост неполного слоя отбрасывается.
    k′ < 3 ⇒ поведенческое плечо снимается (решает вызывающий по k_equal)."""
    have: dict[str, set[int]] = defaultdict(set)
    for g in gens:
        have[g["prompt_key"]].add(int(g["sample_idx"]))
    if not have:
        return gens, {"k_equal": 0, "n_dropped": 0}
    k_eq = min(max((s for s in range(0, max(v) + 2) if set(range(s)) <= v), default=0) for v in have.values())
    kept = [g for g in gens if int(g["sample_idx"]) < k_eq]
    return kept, {"k_equal": k_eq, "n_dropped": len(gens) - len(kept)}


def load_gens(out: Path, env: str | None, cap: int | None) -> list[dict[str, Any]]:
    # 2026-08-28 (ревью R-Launch Д-2): env обязан совпадать с pilot.decision.env при n_env=1 — иначе молчаливый анализ
    # чужой среды по 48 пилотным генерациям (ассерты ниже адаптивны и не поймали бы). Обход: EVALAWARE_FORCE_ENV=1 (для отчёта — deviation).
    import os as _os
    npath = Path(out) / "numbers.json"
    if env is not None and npath.exists():
        _n = json.loads(npath.read_text(encoding="utf-8"))
        dec = _n.get("pilot.decision.env")
        if _n.get("pilot.decision.n_env") == 1 and dec and env != dec:
            if _os.environ.get("EVALAWARE_FORCE_ENV"):        # Р2 блокер 5: обход — только со следом
                from src.contracts import record_numbers
                record_numbers(npath, {"beh.deviation.force_env": f"analysed env={env} vs decision {dec} (EVALAWARE_FORCE_ENV)"})
            else:
                raise SystemExit(f"load_gens: env={env}, а pilot.decision.env={dec} (n_env=1) — поправь PARAMS.env или EVALAWARE_FORCE_ENV=1 (deviation, пишется в numbers)")
    gens = []
    for name in ("pilot_gens.jsonl", "main_gens.jsonl"):
        p = out / name
        if p.exists():
            gens += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    gens = [g for g in gens if g.get("group") == "main" and (env is None or g["env"] == env)]
    if cap:
        gens = [recut_to_cap(g, cap) for g in gens]
    seen = set(); uniq = []
    for g in gens:                       # дубликаты (пилот + main с тем же sample_idx) невозможны по построению, но проверим
        k = (g["prompt_key"], g["sample_idx"])
        assert k not in seen, f"duplicate generation {k}"
        seen.add(k); uniq.append(g)
    return uniq
