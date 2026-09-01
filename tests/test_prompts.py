import json, logging
from dataclasses import replace

import pytest

from src.contracts import make_record, make_records
from src.prompts import (assert_single_insertion, build_prompt, chat_turns, check_forbidden_terms, instruction_sentences,
                         lexicon_hits, load_tokenizer, render, render_text, template_has_date, user_content_span)
from tests.fake_config import DPS, FAKE, INSTR

log = logging.getLogger("test_prompts")
DP = json.loads(DPS.read_text())[0]


def test_lexicon_explicit_stems():
    stems = ["evalu", "eval", "test", "exam", "assess", "grad", "scor", "judg", "redteam", "rating"]
    allow = ["example", "examples"]
    hits = lexicon_hits("We Evaluated the EXAM; graded, testing, assessments. Evaluation: done. Examination too. A red-team judged the score.", stems, allow)
    assert {h[0] for h in hits} == {"evaluated", "exam", "graded", "testing", "assessments", "evaluation", "examination", "redteam", "judged", "score"}
    assert lexicon_hits("Choose one option and explain.", stems, allow) == []
    assert lexicon_hits("For example, the examples were clear.", stems, allow) == []
    assert lexicon_hits("The testimony was gradual.", stems, allow) == [("testimony", "test"), ("gradual", "grad")]


def test_instruction_sentences_equal_across_s(cfg):
    sets = {cid: instruction_sentences(cfg.get(cid)) for cid in ("S1", "S2", "Dstr")}
    assert sets["S1"] == sets["S2"] == sets["Dstr"] == {INSTR}
    assert instruction_sentences(cfg.get("deployment")) == set()


def test_s_conditions_added_text_has_no_eval_terms(cfg):
    for cid in ("S1", "S2", "Dstr", "deployment"):
        assert check_forbidden_terms(cfg.get(cid), cfg.lexicon) == [], cid
        build_prompt(DP, cfg.get(cid), cfg.lexicon)


def test_warn_stems_log_but_do_not_block(cfg, caplog):
    bad = replace(cfg.get("Dstr"), format_template="Checklist:\n{context}\n\nReview the options:\n{actions}")
    with caplog.at_level(logging.WARNING):
        assert check_forbidden_terms(bad, cfg.lexicon) == []
    assert "WARN-lexicon" in caplog.text and "checklist" in caplog.text


def test_forbidden_term_in_template_is_rejected(cfg):
    with pytest.raises(AssertionError, match="forbidden eval terms"):
        build_prompt(DP, replace(cfg.get("S1"), format_template="## Evaluation\n{context}\n{actions}"), cfg.lexicon)
    with pytest.raises(AssertionError, match="quiz"):
        build_prompt(DP, replace(cfg.get("S2"), format_template="Quiz item 1\n{context}\n{actions}"), cfg.lexicon)


def test_six_main_give_six_distinct_prompts_with_verbatim_content(cfg):
    texts = {}
    for c in cfg.main:
        msgs = build_prompt(DP, c, cfg.lexicon)
        assert [m["role"] for m in msgs] == ["user"], c.id
        u = msgs[-1]["content"]
        assert DP["context"] in u and "\n".join(DP["actions"]) in u, c.id
        texts[c.id] = u
    assert len(set(texts.values())) == 6
    assert texts["deployment"] in texts["E1"] and texts["E1_S1"].startswith(texts["S1"])


def test_system_inserts(cfg):
    msgs = build_prompt(DP, cfg.get("positive_control"), cfg.lexicon)
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"] == FAKE["pc"]
    dp2 = dict(DP, system="You are a coding agent.")
    msgs = build_prompt(dp2, cfg.get("positive_control"), cfg.lexicon)
    assert msgs[0]["content"].startswith(FAKE["pc"]) and msgs[0]["content"].endswith("You are a coding agent.")
    assert build_prompt(dp2, cfg.get("e2_b"), cfg.lexicon)[0]["content"].endswith(FAKE["e2_b"])


def test_tokenizer_pin_mismatch_raises(cfg):
    pinned = type(cfg)(**{**cfg.__dict__, "tokenizer": dict(cfg.tokenizer, chat_template_sha256="0" * 64)})
    try:
        with pytest.raises(AssertionError, match="sha256 mismatch"):
            load_tokenizer(pinned)
    except Exception as e:  # noqa: BLE001
        if "mismatch" not in str(e):
            pytest.skip(f"tokenizer unavailable: {str(e)[:100]}")
        raise


def test_e1_vs_deployment_and_e1s1_vs_s1_are_exactly_one_insertion(cfg, tok):
    for base_id, new_id in (("deployment", "E1"), ("S1", "E1_S1")):
        base = render(build_prompt(DP, cfg.get(base_id), cfg.lexicon), tok, cfg)
        new = render(build_prompt(DP, cfg.get(new_id), cfg.lexicon), tok, cfg)
        j1, j2 = assert_single_insertion(base, new, tok)
        log.info("%s→%s insertion (%d tokens): %r", base_id, new_id, j2 - j1, tok.decode(new[j1:j2]))
        assert FAKE["E1"] in tok.decode(new[j1:j2])


def test_render_deterministic_and_thinking_flag_reaches_template(cfg, tok):
    msgs = build_prompt(DP, cfg.get("S2"), cfg.lexicon)
    a, b = render(msgs, tok, cfg), render(msgs, tok, cfg)
    assert a == b
    assert render(msgs, tok, replace(cfg, generation=dict(cfg.generation, thinking=False))) != a
    dates = template_has_date(tok)
    log.info("date-like tokens in chat_template: %r", dates)
    assert not dates


def test_user_content_span_robust_to_system_and_generation_prompt(cfg, tok):
    """Ревью 0: span не зависит от add_generation_prompt и от наличия system; берётся последний ЗАКРЫТЫЙ user-ход."""
    for dp in (DP, dict(DP, system="You are a coding agent.")):
        msgs = build_prompt(dp, cfg.get("E1"), cfg.lexicon)
        spans = set()
        for agp in (True, False):
            ids = render(msgs, tok, cfg, add_generation_prompt=agp)
            s, e = user_content_span(ids, tok, msgs[-1]["content"])
            assert tok.decode(ids[s:e]) == msgs[-1]["content"]
            spans.add((s, e))
        assert len(spans) == 1, spans
        roles = [t[3] for t in chat_turns(ids, tok)]
        assert roles == (["system", "user"] if "system" in dp else ["user"])
    with pytest.raises(AssertionError, match="no closed user turn"):
        user_content_span(render(msgs, tok, cfg)[:5], tok, "x")


def test_records_last_token_uniform_and_user_span(cfg, tok):
    conds = cfg.main + cfg.group("e2") + cfg.group("positive_control")
    recs = make_records([DP, dict(DP, prompt_id="fake_004b", system="You are a coding agent.")], conds, tok, cfg)
    lasts = {(r.last_token_id, r.last_token_str) for r in recs}
    log.info("last token across %d versions: %r ; tail=%r", len(recs), lasts, tok.decode(recs[0].input_ids[-6:]))
    assert len(lasts) == 1
    for r in recs:
        s, e = r.user_span
        assert tok.decode(r.input_ids[s:e]) == r.messages[-1]["content"]
        assert r.input_ids[e] == tok.convert_tokens_to_ids("<|im_end|>") and e < r.n_prompt_tokens - 1
        assert r.position_index("last") == r.n_prompt_tokens - 1 and r.position_index("last_user") == e - 1
    log.info("distinct last_user tokens over main: %r", {tok.decode([r.input_ids[r.user_span[1] - 1]]) for r in recs if r.group == "main"})
    rec = make_record(DP, cfg.get("deployment"), tok, cfg)
    assert rec.template_sha256 and rec.config_hash == cfg.config_hash and rec.tokenizer_revision == "main"
    assert recs[0].dps_sha256 == recs[-1].dps_sha256 and len(recs[0].dps_sha256) == 16
    log.info("rendered deployment prompt:\n%s", render_text(rec.messages, tok, cfg))
