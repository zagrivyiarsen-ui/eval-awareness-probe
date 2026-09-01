"""Модуль d на smoke-модели: прогрев исключён, meta по контракту, eos-набор, unknown_finish, проверка G."""
import json, logging
from dataclasses import replace

from src.conditions import assert_ready_for_generation
from src.contracts import make_records
from src.prompts import render_text
from src.throughput import check_batch_generation_matches_single, eos_ids, measure_throughput
from tests.fake_config import DPS

log = logging.getLogger("test_throughput")


def test_eos_ids_union_with_generation_config(smoke):
    model, tok, _ = smoke
    e = eos_ids(tok, model)
    assert tok.convert_tokens_to_ids("<|im_end|>") in e
    gc = getattr(model.generation_config, "eos_token_id", None)
    for t in ([gc] if isinstance(gc, int) else (gc or [])):
        assert int(t) in e
    log.info("eos ids: %s (tokenizer eos=%s, generation_config eos=%s)", e, tok.eos_token_id, gc)


def test_throughput_structure_and_cot_rule(cfg, smoke):
    model, tok, name = smoke
    dps = json.loads(DPS.read_text())
    recs = make_records(dps[:3], [cfg.get("deployment"), cfg.get("E1")], tok, cfg, tokenizer_name=name)
    cfg24 = replace(cfg, generation=dict(cfg.generation, max_new_tokens=24))
    sampling = assert_ready_for_generation(cfg24, ("main",), temperature_override=0.7, generation_config=model.generation_config, allow_smoke=True)
    assert sampling["top_p"] is not None and sampling["top_k"] is not None and sampling["repetition_penalty"] is not None  # Г-2
    th = measure_throughput(model, tok, recs, sampling, max_new_tokens=24, batch_size=2, n_warmup_batches=1,
                            total_generations_planned=4900, model_name=name)
    metas = th.pop("metas")
    preopens = render_text(recs[0].messages, tok, cfg).rstrip().endswith("<think>")
    log.info("template pre-opens <think>: %s (Qwen3.5 — да; Qwen3-smoke — нет)", preopens)
    log.info("throughput: %s", th)
    assert th["n_measured"] == 4 and th["tok_per_s"] > 0 and th["wallclock_est_hours_planned"] > 0
    assert th["wallclock_valid_for_batch_size"] == 2 and th["unknown_finish_count"] == 0
    assert set(th["truncated_by_condition"]) == {"deployment", "E1"}
    assert sorted(m["sample_idx"] for m in metas) == [0, 0, 0, 0] or len({(m["prompt_key"], m["sample_idx"]) for m in metas}) == 4
    for m in metas:
        assert m["schema"] == "gen.v2" and m["prompt_key"].endswith(m["condition_id"]) and m["dps_sha256"]
        assert m["sampling"]["temperature"] == 0.7 and m["cot_rule"] == "length_and_no_close_think"
        assert m["cot_category"] in ("ok", "truncated", "no_close_eos", "empty_answer", "unknown_finish")
        assert m["finish_reason"] in ("length", "stop")
        if preopens:
            assert "<think>" not in m["output_text"][:8]
        log.info("%s fr=%s cat=%s out=%r", m["condition_id"], m["finish_reason"], m["cot_category"], m["output_text"][:80])


def test_generate_refuses_unresolved_sampling(cfg, smoke):
    """Г-2: None в sampling не должен молча превращаться в дефолт transformers."""
    import pytest
    from src.throughput import generate_ids
    model, tok, name = smoke
    dps = json.loads(DPS.read_text())
    recs = make_records(dps[:1], [cfg.get("deployment")], tok, cfg, tokenizer_name=name)
    with pytest.raises(AssertionError, match="resolved to a number"):
        generate_ids(model, tok, recs, {"temperature": 0.7, "do_sample": True, "top_p": None, "top_k": 50, "repetition_penalty": 1.0}, 4)


def test_checkG_batch_greedy_matches_single(cfg, smoke):
    model, tok, name = smoke
    dps = json.loads(DPS.read_text())
    recs = make_records(dps[:3], [cfg.get("deployment")], tok, cfg, tokenizer_name=name)
    g = check_batch_generation_matches_single(model, tok, recs, n_tokens=16)
    log.info("checkG: %s", g)
    assert g["prefix_match_min"] >= 4, "left-pad батч расходится с поштучным уже в первых токенах"
