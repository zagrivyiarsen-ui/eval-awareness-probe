import pytest, yaml
from src.conditions import MAIN_IDS, PLACEHOLDER, SMOKE_UNPINNED, assert_ready_for_generation, load_conditions
from tests.fake_config import CFG, load_fake, write_fake


def _dump(tmp_path, raw, name):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def _pinned(cfg, sha="deadbeef"):
    return type(cfg)(**{**cfg.__dict__, "tokenizer": dict(cfg.tokenizer, chat_template_sha256=sha)})


def test_shipped_yaml_state():
    """До шага 2: PREREG TEXT NOT FROZEN. После: gate-параметры null → без флага GATE PARAMS UNSET, с флагом грузится."""
    raw_text = CFG.read_text(encoding="utf-8")
    if PLACEHOLDER in raw_text:
        with pytest.raises(RuntimeError, match="PREREG TEXT NOT FROZEN"):
            load_conditions(CFG, allow_unset_gate_params=True)
    else:
        with pytest.raises(RuntimeError, match="GATE PARAMS UNSET"):
            load_conditions(CFG)
        cfg = load_conditions(CFG, allow_unset_gate_params=True)
        assert [c.id for c in cfg.main] == list(MAIN_IDS)


def test_positive_control_placeholder_is_hard_block(tmp_path):
    raw = yaml.safe_load(write_fake(tmp_path).read_text())
    next(c for c in raw["conditions"] if c["id"] == "positive_control")["insert_text"] = PLACEHOLDER
    with pytest.raises(RuntimeError, match="positive_control.insert_text"):
        load_conditions(_dump(tmp_path, raw, "pc.yaml"), allow_unset_gate_params=True)


def test_e2_placeholder_is_soft_but_generation_guard_catches_it(tmp_path, caplog):
    raw = yaml.safe_load(write_fake(tmp_path).read_text())
    next(c for c in raw["conditions"] if c["id"] == "e2_a")["insert_text"] = PLACEHOLDER
    raw["tokenizer"]["chat_template_sha256"] = "deadbeef"
    cfg = load_conditions(_dump(tmp_path, raw, "e2.yaml"))
    assert "E2 placeholders remain" in caplog.text
    assert_ready_for_generation(cfg, ("main", "positive_control"))
    with pytest.raises(AssertionError, match="e2_a"):
        assert_ready_for_generation(cfg, ("e2",))


def test_gate_params_null_needs_flag(tmp_path, capsys):
    with pytest.raises(RuntimeError, match="GATE PARAMS UNSET"):
        load_conditions(write_fake(tmp_path, fill_gate_params=False))
    cfg = load_fake(tmp_path, fill_gate_params=False)
    assert "gate params unset" in capsys.readouterr().err
    assert set(cfg.placeholders) == {"generation.temperature", "generation.batch_size", "probe.candidate_layers"}
    assert [c.id for c in cfg.main] == ["deployment", "E1", "S1", "S2", "E1_S1", "Dstr"]


def test_null_temperature_never_silently_generates(tmp_path):
    cfg = _pinned(load_fake(tmp_path, fill_gate_params=False))
    with pytest.raises(AssertionError, match="temperature is null"):
        assert_ready_for_generation(cfg, ("main",))
    s = assert_ready_for_generation(cfg, ("main",), temperature_override=0.6)
    assert s["temperature"] == 0.6 and s["do_sample"] is True and s["cot_rule"] == "length_and_no_close_think"
    assert s["top_p"] == 1.0 and s["top_p_source"].startswith("hf_default")   # Г-2: дефолт резолвится в число
    cfg2 = _pinned(load_fake(tmp_path))
    with pytest.raises(AssertionError, match="contradicts frozen"):
        assert_ready_for_generation(cfg2, ("main",), temperature_override=0.9)


def test_effective_sampling_from_generation_config(tmp_path):
    """Ревью 1.4: null в yaml → фактическое значение из generation_config записывается, а не теряется."""
    from types import SimpleNamespace
    cfg = _pinned(load_fake(tmp_path))
    gc = SimpleNamespace(top_p=0.95, top_k=20, presence_penalty=1.5, repetition_penalty=None)
    s = assert_ready_for_generation(cfg, ("main",), generation_config=gc)
    assert s["top_p"] == 0.95 and s["top_p_source"] == "generation_config" and s["top_k"] == 20
    assert s["repetition_penalty"] == 1.0 and s["repetition_penalty_source"].startswith("hf_default")


def test_smoke_unpinned_blocks_generation_unless_allow_smoke(tmp_path):
    cfg = load_fake(tmp_path)
    assert cfg.tokenizer["chat_template_sha256"] == SMOKE_UNPINNED
    with pytest.raises(AssertionError, match="not pinned"):
        assert_ready_for_generation(cfg, ("main",))
    assert assert_ready_for_generation(cfg, ("main",), allow_smoke=True)["smoke"] is True
    with pytest.raises(AssertionError, match="only for SMOKE_UNPINNED"):
        assert_ready_for_generation(_pinned(cfg), ("main",), allow_smoke=True)


def test_model_revision_required(tmp_path):
    cfg = _pinned(load_fake(tmp_path))
    cfg = type(cfg)(**{**cfg.__dict__, "model": {"revision": None}})
    with pytest.raises(AssertionError, match="model.revision"):
        assert_ready_for_generation(cfg, ("main",))


def test_reorder_changes_output_order(tmp_path):
    assert [c.id for c in load_fake(tmp_path, reorder=("S1", "S2")).main] == ["deployment", "E1", "S2", "S1", "E1_S1", "Dstr"]


def test_seven_main_conditions_rejected(tmp_path):
    raw = yaml.safe_load(write_fake(tmp_path).read_text())
    extra = dict(raw["conditions"][0]); extra["id"] = "S3"
    raw["conditions"].append(extra)
    with pytest.raises(AssertionError, match="exactly 6 main"):
        load_conditions(_dump(tmp_path, raw, "seven.yaml"))


def test_s1_anchor_shared_with_e1s1():
    raw_text = CFG.read_text(encoding="utf-8")
    assert "&s1_fmt" in raw_text and "*s1_fmt" in raw_text and "&e1_text" in raw_text and "*e1_text" in raw_text


def test_e1s1_template_drift_rejected(tmp_path):
    raw = yaml.safe_load(write_fake(tmp_path).read_text())
    next(c for c in raw["conditions"] if c["id"] == "E1_S1")["format_template"] += "\nextra"
    with pytest.raises(AssertionError, match="E1_S1 template must equal S1"):
        load_conditions(_dump(tmp_path, raw, "drift.yaml"))


def test_position_change_rule_must_be_none(tmp_path):
    raw = yaml.safe_load(write_fake(tmp_path).read_text())
    raw["probe"]["position_change_rule"] = "best_e2_auc"
    with pytest.raises(AssertionError, match="position is not chosen"):
        load_conditions(_dump(tmp_path, raw, "pos.yaml"))


def test_lexicon_stems_validation_and_deviations(tmp_path):
    raw = yaml.safe_load(write_fake(tmp_path).read_text())
    raw["eval_lexicon_stems"].append("ev")
    with pytest.raises(AssertionError, match=">=4"):
        load_conditions(_dump(tmp_path, raw, "short.yaml"))
    raw = yaml.safe_load(write_fake(tmp_path).read_text())
    raw["eval_lexicon_warn_stems"].append("test")
    with pytest.raises(AssertionError, match="overlap"):
        load_conditions(_dump(tmp_path, raw, "overlap.yaml"))
    raw = yaml.safe_load(write_fake(tmp_path).read_text())
    raw["prereg_deviations"] = [{"word": "observations"}]
    with pytest.raises(AssertionError, match="word\\+reason"):
        load_conditions(_dump(tmp_path, raw, "dev.yaml"))
    raw["prereg_deviations"] = [{"word": "Observations", "reason": "Dstr header"}]
    cfg2 = load_conditions(_dump(tmp_path, raw, "dev2.yaml"))
    cfg = load_fake(tmp_path)
    assert "observations" in cfg2.lexicon.allow and cfg2.lexicon.deviations == ("observations",)
    assert cfg2.config_hash != cfg.config_hash          # deviation попадает в хеш
    assert "check" in cfg.lexicon.warn_stems and "example" in cfg.lexicon.allow and len(cfg.config_hash) == 16
