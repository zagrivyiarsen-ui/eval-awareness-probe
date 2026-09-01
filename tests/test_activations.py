"""Модуль c на smoke-модели. Проверки 1, 3, 6, структура .pt, toy-probe, запрет S в диагностике."""
import json, logging

import pytest
import torch

from src.activations import (check_hook_matches_hidden_states, check_last_token_uniform, check_naive_last_index_breaks_on_right_pad,
                             check_norm_vs_length, check_padding_invariance, check_repeatability_and_memory,
                             default_candidate_layers, extract, get_layers, hooks_are_clean)
from src.contracts import acts_payload, make_records, validate_acts_payload
from src.probe_toy import position_diagnostic
from tests.fake_config import DPS

log = logging.getLogger("test_activations")
POS = ["last", "last_user", "mean_user"]


@pytest.fixture(scope="module")
def recs(cfg, smoke):
    model, tok, name = smoke
    dps = json.loads(DPS.read_text())
    return make_records(dps[:4], cfg.main + cfg.group("e2"), tok, cfg, tokenizer_name=name)


@pytest.fixture(scope="module")
def layers(smoke):
    n = len(get_layers(smoke[0]))
    L = default_candidate_layers(n)
    log.info("smoke layers=%d candidates=%s", n, L)
    assert len(L) <= 3 and all(l < n - 1 for l in L)
    return L


def test_check1_padding_invariance_and_naive_index_breaks(cfg, smoke, recs, layers):
    model, tok, _ = smoke
    sub = [r for r in recs if r.prompt_id == recs[0].prompt_id]
    assert len({r.n_prompt_tokens for r in sub}) >= 3
    thr = cfg.thresholds
    res = check_padding_invariance(model, tok, sub, layers, POS, rel_l2_max=thr["check1_rel_l2_max"], cos_min=thr["check1_cos_min"],
                                   assert_sides=("right", "left"))
    log.info("check1: %s", res)
    naive = check_naive_last_index_breaks_on_right_pad(model, tok, sub, layers[0])
    log.info("naive [-1] on right pad: %s (must be large)", naive)
    assert naive["rel_l2_max"] >= thr["check1_naive_rel_l2_min"] and naive["cos_min"] < 0.9, "тест не кусается"


def test_check2_last_token_uniform(recs):
    u = check_last_token_uniform(recs)
    log.info("check2 (smoke template, NOT the plan template): %s", u)
    assert u["uniform"]


def test_check3_hook_equals_hidden_states(cfg, smoke, recs, layers):
    model, tok, _ = smoke
    res = check_hook_matches_hidden_states(model, tok, recs[0], layers, max_abs_diff=cfg.thresholds["check3_max_abs_diff"])
    log.info("check3: %s", res)
    assert res["n_hidden_states"] == len(get_layers(model)) + 1
    with pytest.raises(AssertionError, match="последний блок"):
        check_hook_matches_hidden_states(model, tok, recs[0], [len(get_layers(model)) - 1])


def test_check6_repeatable_hooks_clean(cfg, smoke, recs, layers):
    model, tok, _ = smoke
    res = check_repeatability_and_memory(model, tok, recs[:6], layers, POS, n_repeats=2)
    log.info("check6: %s", res)
    assert res["hooks_clean"] and (res["bitwise_repeatable"] or res["cos_min"] >= cfg.thresholds["check6_cos_min"])
    if "cuda_mem_growth_bytes" in res:
        assert res["cuda_mem_growth_bytes"] < 16 * 1024 * 1024


def test_extract_payload_schema_and_check5(smoke, recs, layers, tmp_path):
    model, tok, name = smoke
    acts = extract(model, tok, recs, layers, POS, batch_size=5)
    assert hooks_are_clean(model, layers)
    payload = acts_payload(recs, acts, name, None, str(next(model.parameters()).dtype), "right")
    validate_acts_payload(payload)
    torch.save(payload, tmp_path / "acts.pt")
    back = torch.load(tmp_path / "acts.pt", weights_only=False)
    validate_acts_payload(back)
    assert back["prompt_keys"] == [r.prompt_key for r in recs] and back["schema"] == "acts.v2" and back["padding_side"] == "right"
    for p in POS:
        log.info("check5 %s: %s", p, check_norm_vs_length(acts[p][layers[-1]], recs))


def test_toy_probe_diagnostic_positions_and_s_leak_guard(smoke, recs, layers):
    model, tok, _ = smoke
    keep = [r for r in recs if r.condition_id in ("deployment", "E1", "e2_a", "e2_b", "e2_c")]
    acts = extract(model, tok, keep, layers, POS, batch_size=8)
    L = layers[-1]
    diag = position_diagnostic({p: acts[p][L] for p in POS}, [r.condition_id for r in keep], [r.env_prompt for r in keep])
    log.info("toy-probe diagnostic L=%d: %s", L, diag)
    for p in POS:
        assert "auc_e2_vs_deploy_along_e1_dir" in diag[p] and len(diag[p]["heldout_aucs"]) >= 2
    # стринг-детектор: на last_user AUC E1/deploy ≥ на last (хвост вставки) — это и есть аргумент против выбора по AUC
    log.info("AUC e1/deploy last=%.3f last_user=%.3f", diag["last"]["heldout_auc_e1_vs_deploy"], diag["last_user"]["heldout_auc_e1_vs_deploy"])
    with pytest.raises(AssertionError, match="S-conditions"):
        position_diagnostic({p: extract(model, tok, recs, [L], [p])[p][L] for p in ["last"]},
                            [r.condition_id for r in recs], [r.env_prompt for r in recs])
    with pytest.raises(AssertionError, match="env/prompt_id"):
        position_diagnostic({"last": acts["last"][L]}, [r.condition_id for r in keep], [r.prompt_id for r in keep])
