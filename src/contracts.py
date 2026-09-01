"""Контракты: PromptRecord, meta генерации, правило обрыва, .pt, numbers.json (+история), env, MDE."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.conditions import COT_RULE, Condition, Config
from src.prompts import Message, build_prompt, render, template_sha256, user_content_span

ACTS_SCHEMA = "acts.v2"
META_SCHEMA = "gen.v2"


def dps_sha256(dps_or_path) -> str:
    if isinstance(dps_or_path, (str, Path)):
        return hashlib.sha256(Path(dps_or_path).read_bytes()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(dps_or_path, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PromptRecord:
    prompt_key: str
    env: str
    prompt_id: str
    condition_id: str
    group: str
    messages: list[Message]
    input_ids: list[int]
    n_prompt_tokens: int
    last_token_id: int
    last_token_str: str
    user_span: tuple[int, int]
    config_hash: str
    dps_sha256: str
    template_sha256: str
    tokenizer_name: str
    tokenizer_revision: str | None
    thinking: bool

    @property
    def env_prompt(self) -> str:
        return f"{self.env}/{self.prompt_id}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["user_span"] = list(self.user_span)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptRecord":
        d = dict(d)
        d["user_span"] = tuple(d["user_span"])
        return cls(**d)

    def position_index(self, position: str) -> int | tuple[int, int]:
        if position == "last":
            return self.n_prompt_tokens - 1
        if position == "last_user":
            return self.user_span[1] - 1
        if position == "mean_user":
            return self.user_span
        raise ValueError(position)


def make_record(dp: dict[str, Any], cond: Condition, tokenizer, cfg: Config, tokenizer_name: str | None = None,
                dps_sha: str = "?") -> PromptRecord:
    msgs = build_prompt(dp, cond, cfg.lexicon)
    ids = render(msgs, tokenizer, cfg)
    span = user_content_span(ids, tokenizer, msgs[-1]["content"])
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert span[1] < len(ids) and ids[span[1]] == im_end, "user span must be closed by <|im_end|> before generation prompt"
    return PromptRecord(prompt_key=f"{dp['env']}/{dp['prompt_id']}/{cond.id}", env=dp["env"], prompt_id=dp["prompt_id"],
                        condition_id=cond.id, group=cond.group, messages=msgs, input_ids=ids, n_prompt_tokens=len(ids),
                        last_token_id=ids[-1], last_token_str=tokenizer.decode([ids[-1]]), user_span=span,
                        config_hash=cfg.config_hash, dps_sha256=dps_sha, template_sha256=template_sha256(tokenizer),
                        tokenizer_name=tokenizer_name or cfg.tokenizer["name"], tokenizer_revision=cfg.tokenizer.get("revision"),
                        thinking=bool(cfg.generation["thinking"]))


def make_records(dps: Sequence[dict[str, Any]], conds: Sequence[Condition], tokenizer, cfg: Config,
                 tokenizer_name: str | None = None, dps_sha: str | None = None) -> list[PromptRecord]:
    sha = dps_sha or dps_sha256(list(dps))
    recs = [make_record(dp, c, tokenizer, cfg, tokenizer_name=tokenizer_name, dps_sha=sha) for dp in dps for c in conds]
    keys = [r.prompt_key for r in recs]
    assert len(keys) == len(set(keys)), "duplicate prompt_key"
    return recs


def save_records(recs: Sequence[PromptRecord], path: Path) -> None:
    atomic_write_text(path, "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in recs) + "\n")


def load_records(path: Path) -> list[PromptRecord]:
    return [PromptRecord.from_dict(json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------------ правило обрыва CoT
CoT_CATEGORIES = ("ok", "truncated", "no_close_eos", "empty_answer", "unknown_finish")
FINISH_REASONS = ("length", "stop", "unknown")


def classify_cot(output_text: str, finish_reason: str) -> dict[str, Any]:
    """<think> открыт шаблоном в промпте. truncated = length И нет </think>. reopened_think>0 — аномалия,
    исключается из CoT-метрик (cot_text — только до первого </think>)."""
    assert finish_reason in FINISH_REASONS, finish_reason
    closed = "</think>" in output_text
    reopened = output_text.count("<think>")
    if finish_reason == "unknown":
        cat, answer = "unknown_finish", ""
    elif not closed:
        cat, answer = ("truncated" if finish_reason == "length" else "no_close_eos"), ""
    else:
        answer = output_text.split("</think>", 1)[1].strip()
        cat = "empty_answer" if not answer else "ok"
    return {"cot_category": cat, "cot_closed": closed, "reopened_think": reopened, "answer_text": answer,
            "cot_text": output_text.split("</think>", 1)[0], "cot_rule": COT_RULE}


def finish_reason(generated_ids: Sequence[int], max_new_tokens: int, eos_ids: Sequence[int]) -> str:
    """stop ⇔ eos встречен; length ⇔ нет eos и len ≥ max; иначе unknown (стоп-сигнал, не прячется в 'stop')."""
    if any(t in eos_ids for t in generated_ids):
        return "stop"
    return "length" if len(generated_ids) >= max_new_tokens else "unknown"


def generation_meta(rec: PromptRecord, sample_idx: int, sampling: dict[str, Any], output_text: str,
                    generated_ids: Sequence[int], fr: str, model_name: str, model_revision: str | None) -> dict[str, Any]:
    cot = classify_cot(output_text, fr)
    return {"schema": META_SCHEMA, "prompt_key": rec.prompt_key, "env": rec.env, "prompt_id": rec.prompt_id,
            "condition_id": rec.condition_id, "group": rec.group, "sample_idx": sample_idx,
            "n_prompt_tokens": rec.n_prompt_tokens, "n_generated_tokens": len(generated_ids),   # включая eos
            "finish_reason": fr, **cot, "output_text": output_text, "sampling": sampling,
            "config_hash": rec.config_hash, "dps_sha256": rec.dps_sha256, "template_sha256": rec.template_sha256,
            "model_name": model_name, "model_revision": model_revision}


# ------------------------------------------------------------------ .pt
def acts_payload(recs: Sequence[PromptRecord], acts: dict[str, dict[int, Any]], model_name: str, model_revision: str | None,
                 compute_dtype: str, padding_side: str) -> dict[str, Any]:
    import torch
    n = len(recs)
    for pos, by_layer in acts.items():
        for L, t in by_layer.items():
            assert isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.shape[0] == n, (pos, L)
    return {"schema": ACTS_SCHEMA, "prompt_keys": [r.prompt_key for r in recs], "condition_ids": [r.condition_id for r in recs],
            "env_prompts": [r.env_prompt for r in recs], "n_prompt_tokens": [r.n_prompt_tokens for r in recs],
            "positions": sorted(acts), "layers": sorted({L for d in acts.values() for L in d}),
            "acts": {p: {int(L): t for L, t in d.items()} for p, d in acts.items()},
            "config_hash": recs[0].config_hash, "dps_sha256": recs[0].dps_sha256, "template_sha256": recs[0].template_sha256,
            "model_name": model_name, "model_revision": model_revision, "compute_dtype": compute_dtype, "padding_side": padding_side}


def validate_acts_payload(p: dict[str, Any]) -> None:
    assert p.get("schema") == ACTS_SCHEMA, p.get("schema")
    n = len(p["prompt_keys"])
    assert len(set(p["prompt_keys"])) == n
    for pos in p["positions"]:
        for L in p["layers"]:
            t = p["acts"][pos][L]
            assert t.dim() == 2 and t.shape[0] == n and str(t.dtype) == "torch.float32", (pos, L, t.shape, t.dtype)


# ------------------------------------------------------------------ numbers.json + история
def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _clean_scalar(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (list, tuple)):
        assert all(not isinstance(x, (dict, list, tuple)) for x in v), f"list must hold scalars: {v!r}"
        return [_clean_scalar(x) for x in v]
    assert v is None or isinstance(v, (str, int, float, bool)), f"unsupported value type {type(v)}: {v!r}"
    return v


def _flatten(k: str, v: Any, out: dict[str, Any]) -> None:
    if isinstance(v, dict):
        for kk, vv in v.items():
            _flatten(f"{k}.{kk}", vv, out)
    else:
        out[k] = _clean_scalar(v)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def record_numbers(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Dotted-ключи; dict-значения раскатываются; NaN → null. Каждый вызов дублируется в numbers_history.jsonl
    (timestamp, argv, git) — повторный запуск Gate 0 с другими параметрами оставляет след."""
    path = Path(path)
    cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    flat: dict[str, Any] = {}
    for k, v in updates.items():
        assert isinstance(k, str) and "." in k, f"key must be dotted: {k!r}"
        _flatten(k, v, flat)
    cur.update(flat)
    atomic_write_text(path, json.dumps(cur, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    with open(path.with_name("numbers_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv, "git": git_sha(), "updates": flat},
                           ensure_ascii=False) + "\n")
    return cur


def persist_results(out: Path) -> None:
    """Инкрементальный бэкап results/ после каждого батча (2026-08-28): zip-путь из EVALAWARE_PERSIST_ZIP (драйвер Kaggle
    задаёт /kaggle/working/evalaware_results_backup). Без переменной — no-op."""
    import os, shutil
    z = os.environ.get("EVALAWARE_PERSIST_ZIP")
    if not z:
        return
    out = Path(out).resolve()
    shutil.make_archive(z, "zip", root_dir=out.parent, base_dir=out.name)


def assert_continuity(numbers_path: Path, root: Path, dps_sha: str, stage: str) -> None:
    """2026-08-28 (ревью R-Pipeline А.8): при старте каждого скрипта после verify — DP и код те же, что в prereg;
    иначе стоп (DP) или запись отклонения (код). Пишет строку в numbers_history через record_numbers."""
    from src.analysis import code_sha256
    nums = json.loads(Path(numbers_path).read_text(encoding="utf-8"))
    assert nums.get("prereg.dps_sha256") == dps_sha, f"{stage}: dps_sha256 {dps_sha} != prereg {nums.get('prereg.dps_sha256')}"
    cur = code_sha256(root)
    if nums.get("prereg.code_sha256") != cur:
        record_numbers(numbers_path, {f"continuity.{stage}.code_sha256_deviation": f"{nums.get('prereg.code_sha256')}->{cur}"})
    record_numbers(numbers_path, {f"continuity.{stage}.code_sha256": cur})


def assert_matches_prereg(cfg: Config, numbers_path: Path, dps_sha: str) -> None:
    """Этап 2+: конфиг и DP не редактировались после verify_prereg.py."""
    nums = json.loads(Path(numbers_path).read_text(encoding="utf-8"))
    assert nums.get("prereg.config_hash") == cfg.config_hash, f"config_hash {cfg.config_hash} != prereg {nums.get('prereg.config_hash')}"
    assert nums.get("prereg.dps_sha256") == dps_sha, f"dps_sha256 {dps_sha} != prereg {nums.get('prereg.dps_sha256')}"


def env_info() -> dict[str, Any]:
    import importlib.util as iu
    import torch, transformers
    return {"env.python": sys.version.split()[0], "env.torch": torch.__version__, "env.transformers": transformers.__version__,
            "env.cuda": torch.version.cuda, "env.git_sha": git_sha(),
            "env.flash_attn_available": iu.find_spec("flash_attn") is not None,
            "env.fla_available": iu.find_spec("fla") is not None,
            "env.causal_conv1d_available": iu.find_spec("causal_conv1d") is not None}


# ------------------------------------------------------------------ априорный MDE (§4.3)
def mde_apriori(sd_delta: float, n_prompts: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Парный тест на n промптовых дельтах: MDE ≈ (z_{1-α/2}+z_{power})·SD(Δ)/√n. Нормальное приближение."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * sd_delta / math.sqrt(n_prompts)
