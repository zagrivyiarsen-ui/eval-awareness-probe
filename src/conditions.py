"""Загрузчик conditions.yaml — единственного источника правды."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

PLACEHOLDER = "TODO_PREREG"
SMOKE_UNPINNED = "SMOKE_UNPINNED"
MAIN_IDS = ("deployment", "E1", "S1", "S2", "E1_S1", "Dstr")
S_IDS = ("S1", "S2", "E1_S1", "Dstr")            # результаты по ним нельзя смотреть до заморозки пробе
FREEZE_GROUPS_HARD = ("main", "positive_control")
FREEZE_GROUPS_SOFT = ("e2",)
GATE_PARAMS = {
    "generation.temperature": ("generation", "temperature"),
    "generation.max_new_tokens": ("generation", "max_new_tokens"),
    "generation.k": ("generation", "k"),
    "generation.batch_size": ("generation", "batch_size"),
    "probe.candidate_layers": ("probe", "candidate_layers"),
    "tokenizer.revision": ("tokenizer", "revision"),
    "tokenizer.chat_template_sha256": ("tokenizer", "chat_template_sha256"),
    "model.revision": ("model", "revision"),
}
ROLES = ("none", "user", "system")
POSITIONS = ("none", "prefix", "suffix")
PROBE_POSITIONS = ("last", "last_user", "mean_user")
COT_RULE = "length_and_no_close_think"
REQUIRED_THRESHOLDS = ("check1_rel_l2_max", "check1_cos_min", "check1_assert_left", "check1_naive_rel_l2_min", "check3_max_abs_diff",
                       "check6_cos_min", "checkG_prefix_tokens_min", "toyprobe_heldout_auc_min",
                       "truncated_frac_max", "wallclock_hours_max", "mde_sd_delta_apriori")
SAMPLING_KEYS = ("top_p", "top_k", "presence_penalty", "repetition_penalty")


@dataclass(frozen=True)
class Condition:
    id: str
    group: str
    insert_role: str
    insert_position: str
    insert_text: str | None
    format_template: str
    forbidden_eval_terms: bool

    def placeholders(self) -> list[str]:
        out = []
        if self.insert_text is not None and PLACEHOLDER in self.insert_text:
            out.append(f"{self.id}.insert_text")
        if PLACEHOLDER in self.format_template:
            out.append(f"{self.id}.format_template")
        return out


@dataclass(frozen=True)
class Lexicon:
    stems: tuple[str, ...]
    allow: tuple[str, ...]          # allow ∪ prereg_deviations
    warn_stems: tuple[str, ...]
    deviations: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    conditions: tuple[Condition, ...]
    lexicon: Lexicon
    generation: dict[str, Any]
    probe: dict[str, Any]
    thresholds: dict[str, Any]
    tokenizer: dict[str, Any]
    model: dict[str, Any]
    placeholders: tuple[str, ...]
    config_hash: str

    @property
    def main(self) -> list[Condition]:
        return [c for c in self.conditions if c.group == "main"]

    def group(self, g: str) -> list[Condition]:
        return [c for c in self.conditions if c.group == g]

    def get(self, cid: str) -> Condition:
        for c in self.conditions:
            if c.id == cid:
                return c
        raise KeyError(cid)


def config_hash_of(raw: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _unset_gate_params(raw: dict[str, Any]) -> list[str]:
    return [k for k, (a, b) in GATE_PARAMS.items() if (raw.get(a) or {}).get(b) is None]


def text_placeholders(conds, groups: tuple[str, ...]) -> list[str]:
    return [p for c in conds if c.group in groups for p in c.placeholders()]


def _parse_condition(d: dict[str, Any]) -> Condition:
    keys = {"id", "group", "insert_role", "insert_position", "insert_text", "format_template", "forbidden_eval_terms"}
    assert set(d) == keys, f"condition {d.get('id')}: keys {set(d) ^ keys}"
    c = Condition(**d)
    assert c.insert_role in ROLES and c.insert_position in POSITIONS, c.id
    if c.insert_role == "none":
        assert c.insert_position == "none" and c.insert_text is None, f"{c.id}: no role but insert given"
    else:
        assert c.insert_position != "none" and c.insert_text is not None, f"{c.id}: role without position/text"
    assert isinstance(c.format_template, str) and c.format_template, c.id
    assert isinstance(c.forbidden_eval_terms, bool), c.id
    return c


def _parse_lexicon(raw: dict[str, Any]) -> Lexicon:
    assert raw.get("eval_lexicon_scope") == "added_text_only"
    stems = [s.lower() for s in raw["eval_lexicon_stems"]]
    assert stems and len(stems) == len(set(stems)), "stems empty/duplicated"
    for s in stems:
        assert re.fullmatch(r"[a-z]+", s) and len(s) >= 4, f"stem {s!r}: lowercase letters, >=4 chars"
    allow = [w.lower() for w in raw.get("eval_lexicon_allow") or []]
    devs = []
    for d in raw.get("prereg_deviations") or []:
        assert isinstance(d, dict) and set(d) == {"word", "reason"} and d["reason"].strip(), f"prereg_deviations entry {d!r} needs word+reason"
        devs.append(d["word"].lower())
    warn = [w.lower() for w in raw.get("eval_lexicon_warn_stems") or []]
    assert not set(warn) & set(stems), "warn stems overlap blocking stems"
    return Lexicon(tuple(stems), tuple(allow + devs), tuple(warn), tuple(devs))


def load_conditions(path: str | Path, allow_unset_gate_params: bool = False) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    conds = [_parse_condition(d) for d in raw["conditions"]]
    ids = [c.id for c in conds]
    assert len(ids) == len(set(ids)), f"duplicate condition ids: {ids}"
    main = [c for c in conds if c.group == "main"]
    assert len(main) == 6, f"exactly 6 main conditions required, got {len(main)}"
    assert set(c.id for c in main) == set(MAIN_IDS), f"main ids must be {MAIN_IDS}"
    by = {c.id: c for c in conds}
    assert by["deployment"].insert_role == "none"
    assert by["deployment"].format_template == "{context}\n\n{actions}", "deployment template must be identity"
    assert by["E1"].format_template == by["deployment"].format_template, "E1 must not change structure"
    assert by["E1"].insert_role == "user", "E1 insert role must be user (§4.7)"
    assert by["E1"].insert_position == raw["probe"]["e1_insert_position"]
    assert (by["E1_S1"].insert_text, by["E1_S1"].insert_role, by["E1_S1"].insert_position) == \
        (by["E1"].insert_text, by["E1"].insert_role, by["E1"].insert_position), "E1_S1 insert must equal E1"
    assert by["E1_S1"].format_template == by["S1"].format_template, "E1_S1 template must equal S1"
    for cid in ("S1", "S2", "Dstr"):
        assert by[cid].insert_role == "none" and by[cid].forbidden_eval_terms, cid
    assert by["positive_control"].insert_role == "system"

    lexicon = _parse_lexicon(raw)
    assert raw["probe"].get("position") in PROBE_POSITIONS
    assert set(raw["probe"].get("positions_diagnostic") or ()) <= set(PROBE_POSITIONS)
    assert raw["probe"].get("position_change_rule") == "none", "position is not chosen from Gate 0 data (review 1.3)"
    assert isinstance(raw["probe"].get("layer_rule"), str)
    assert raw["generation"].get("cot_rule") == COT_RULE
    assert isinstance(raw["generation"].get("thinking"), bool)
    assert isinstance(raw.get("tokenizer", {}).get("name"), str)
    assert isinstance(raw.get("model"), dict), "model: {revision:} section required"
    thr = raw["thresholds"]
    missing = [k for k in REQUIRED_THRESHOLDS if k not in thr]
    assert not missing, f"thresholds missing (prereg numbers): {missing}"

    hard = text_placeholders(conds, FREEZE_GROUPS_HARD)
    if hard:
        raise RuntimeError("PREREG TEXT NOT FROZEN (no bypass exists): " + ", ".join(hard))
    soft = text_placeholders(conds, FREEZE_GROUPS_SOFT)
    if soft:
        log.warning("E2 placeholders remain (blocked by assert_ready_for_generation(('e2',)), not here): %s", soft)
    unset = _unset_gate_params(raw)
    if unset and not allow_unset_gate_params:
        raise RuntimeError("GATE PARAMS UNSET: " + ", ".join(unset) + ". Fix in Gate 0/1 or pass allow_unset_gate_params=True.")
    if unset:
        banner = "\n".join(["#" * 78,
                            "#  WARNING: gate params unset (allow_unset_gate_params=True): " + ", ".join(unset),
                            "#  Legal for Gate 0 smoke tests and prefill. NOT for any generation.", "#" * 78])
        print(banner, file=sys.stderr, flush=True)
        log.warning(banner)
    return Config(conditions=tuple(conds), lexicon=lexicon, generation=dict(raw["generation"]), probe=dict(raw["probe"]),
                  thresholds=dict(thr), tokenizer=dict(raw["tokenizer"]), model=dict(raw["model"]),
                  placeholders=tuple(soft + unset), config_hash=config_hash_of(raw))


def _tf_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:  # noqa: BLE001
        return "?"


def _hf_default(key: str):
    """Численное значение дефолта transformers для sampling-ключа (то, что generate() применит молча при None)."""
    try:
        from transformers import GenerationConfig
        v = getattr(GenerationConfig(), key, None)
    except Exception:  # noqa: BLE001
        v = None
    fallback = {"top_p": 1.0, "top_k": 50, "presence_penalty": 0.0, "repetition_penalty": 1.0}
    return fallback[key] if v is None else (float(v) if isinstance(v, float) else v)


def _gc_get(generation_config, key):
    if generation_config is None:
        return None
    v = getattr(generation_config, key, None)
    return None if v is None else (float(v) if isinstance(v, float) else v)


def assert_ready_for_generation(cfg: Config, groups: tuple[str, ...], temperature_override: float | None = None,
                                generation_config=None, allow_smoke: bool = False) -> dict[str, Any]:
    """В начале ЛЮБОГО генерирующего кода. Возвращает ФАКТИЧЕСКИЕ sampling-параметры для meta:
    null в yaml → значение из generation_config модели (записывается, а не молча используется).
    allow_smoke — только dry-run на fake-конфиге в каталоге *fake*; снимает лишь проверку пиннинга."""
    bad = text_placeholders(cfg.conditions, groups)
    assert not bad, f"refusing to generate: placeholders in {bad}"
    temp = cfg.generation.get("temperature")
    if temp is None:
        assert temperature_override is not None, "refusing to generate: generation.temperature is null and no explicit temperature_override"
        temp = temperature_override
    elif temperature_override is not None:
        assert float(temperature_override) == float(temp), f"temperature_override={temperature_override} contradicts frozen {temp}"
    assert isinstance(temp, (int, float)) and temp >= 0, temp
    sha = cfg.tokenizer.get("chat_template_sha256")
    if not allow_smoke:
        assert sha and sha != SMOKE_UNPINNED, "refusing to generate: tokenizer.chat_template_sha256 not pinned"
        assert cfg.tokenizer.get("revision"), "refusing to generate: tokenizer.revision not pinned"
        assert cfg.model.get("revision"), "refusing to generate: model.revision not pinned"
    else:
        assert sha == SMOKE_UNPINNED, "allow_smoke is only for SMOKE_UNPINNED fake configs"
    assert cfg.generation.get("max_new_tokens"), "max_new_tokens unset"
    eff = {}
    for k in SAMPLING_KEYS:
        v = cfg.generation.get(k)
        src = "config"
        if v is None:
            v = _gc_get(generation_config, k)
            src = "generation_config" if v is not None else "hf_default"
        if v is None:                                  # Г-2 (ревью R-Gate0): hf_default → ЧИСЛО, не null
            v = _hf_default(k)
            src = f"hf_default(transformers={_tf_version()})"
        eff[k] = v
        eff[k + "_source"] = src
    return {"temperature": float(temp), "do_sample": float(temp) > 0, "max_new_tokens": int(cfg.generation["max_new_tokens"]),
            "thinking": bool(cfg.generation["thinking"]), "cot_rule": COT_RULE, "smoke": bool(allow_smoke), **eff}


def conditions_hash(cfg_path: str | Path) -> str:
    """Хеш ТОЛЬКО текстов условий + лексикона (заморожены); config_hash дополнительно покрывает gate-параметры и пороги."""
    raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    part = {k: raw[k] for k in ("conditions", "eval_lexicon_stems", "eval_lexicon_allow", "eval_lexicon_warn_stems",
                                "prereg_deviations", "eval_lexicon_scope") if k in raw}
    return hashlib.sha256(json.dumps(part, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
