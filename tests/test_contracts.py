import json

import pytest

from src.contracts import PromptRecord, classify_cot, finish_reason, mde_apriori, record_numbers, assert_matches_prereg


def test_classify_cot_rule():
    ok = classify_cot("I think about it.</think>\n\nfix_tests", "stop")
    assert ok["cot_category"] == "ok" and ok["answer_text"] == "fix_tests" and ok["cot_closed"]
    assert classify_cot("I keep thinking and thinking", "length")["cot_category"] == "truncated"
    assert classify_cot("no close but eos", "stop")["cot_category"] == "no_close_eos"
    assert classify_cot("thoughts</think>\n  \n", "stop")["cot_category"] == "empty_answer"
    re_ = classify_cot("a</think>b<think>c</think>d", "stop")
    assert re_["reopened_think"] == 1 and re_["cot_category"] == "ok" and re_["cot_text"] == "a"
    assert classify_cot("x</think>y", "unknown")["cot_category"] == "unknown_finish"
    with pytest.raises(AssertionError):
        classify_cot("x", "eos")


def test_finish_reason_unknown_is_not_hidden():
    assert finish_reason([1, 2, 3], 3, [99]) == "length"
    assert finish_reason([1, 2, 99], 3, [99]) == "stop"
    assert finish_reason([1, 2], 3, [99]) == "unknown"


def test_numbers_json_flatten_history_nan(tmp_path):
    p = tmp_path / "numbers.json"
    record_numbers(p, {"gate0.tok_per_s": 10.0, "gate0.trunc": 0.1, "gate0.d": {"a": 1, "b": {"c": float("nan")}}})
    record_numbers(p, {"gate0.tok_per_s": 12.5})
    d = json.loads(p.read_text())
    assert d == {"gate0.tok_per_s": 12.5, "gate0.trunc": 0.1, "gate0.d.a": 1, "gate0.d.b.c": None}
    hist = [json.loads(l) for l in (tmp_path / "numbers_history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 and hist[0]["updates"]["gate0.tok_per_s"] == 10.0 and "ts" in hist[1]
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(AssertionError, match="dotted"):
        record_numbers(p, {"flat": 1})
    with pytest.raises(AssertionError, match="scalars"):
        record_numbers(p, {"x.y": [{"a": 1}]})


def test_assert_matches_prereg(tmp_path):
    from types import SimpleNamespace
    p = tmp_path / "numbers.json"
    record_numbers(p, {"prereg.config_hash": "abc", "prereg.dps_sha256": "d1"})
    assert_matches_prereg(SimpleNamespace(config_hash="abc"), p, "d1")
    with pytest.raises(AssertionError, match="config_hash"):
        assert_matches_prereg(SimpleNamespace(config_hash="zzz"), p, "d1")
    with pytest.raises(AssertionError, match="dps_sha256"):
        assert_matches_prereg(SimpleNamespace(config_hash="abc"), p, "d2")


def test_mde_apriori_matches_plan_table():
    assert abs(mde_apriori(0.2, 16) - 0.14) < 0.01      # k=10-модель плана
    assert abs(mde_apriori(0.11, 16) - 0.077) < 0.005   # k=25 → ≈0.07 (§4.4)


def test_prompt_record_roundtrip():
    r = PromptRecord(prompt_key="e/p/c", env="e", prompt_id="p", condition_id="c", group="main",
                     messages=[{"role": "user", "content": "x"}], input_ids=[1, 2, 3, 4], n_prompt_tokens=4,
                     last_token_id=4, last_token_str="\n", user_span=(2, 3), config_hash="h", dps_sha256="d",
                     template_sha256="t", tokenizer_name="tok", tokenizer_revision=None, thinking=True)
    assert PromptRecord.from_dict(json.loads(json.dumps(r.to_dict()))) == r
    assert r.env_prompt == "e/p"
    assert r.position_index("last") == 3 and r.position_index("last_user") == 2 and r.position_index("mean_user") == (2, 3)
