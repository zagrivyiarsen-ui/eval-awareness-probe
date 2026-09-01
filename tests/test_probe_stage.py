"""Смоук всего анализа этапа 5 на синтетических активациях (без модели): плантируем сдвиг E1 вдоль вектора u,
S1/S2 — половина сдвига, Dstr — ноль, topic — ноль, e2_* — сдвиг; проверяем, что скрипт проходит от A до E и числа осмысленны."""
import json
from pathlib import Path

import pytest
import torch

from src.contracts import dps_sha256, make_records, record_numbers


def _dps():
    out = []
    for e in ("sandbagging", "secret_number"):
        for i in range(1, 7):
            out.append({"env": e, "prompt_id": f"{e[:2]}_{i:03d}", "context": f"Context for {e} number {i}. " * (3 + i),
                        "actions": "State your next action."})
    return out


def test_probe_stage_synthetic(cfg, tok, tmp_path):
    from scripts import probe_stage
    dps = _dps(); dps_path = tmp_path / "dps.json"; dps_path.write_text(json.dumps(dps))
    topic = {"prompts": [{"env": "topic", "prompt_id": f"tp_{i}", "context": f"Grading essay {i} with a rubric.", "actions": "Assess it."} for i in range(4)]}
    tpath = tmp_path / "topic.json"; tpath.write_text(json.dumps(topic))
    out = tmp_path / "results"; out.mkdir()
    dsha = dps_sha256(dps_path)
    record_numbers(out / "numbers.json", {"prereg.config_hash": cfg.config_hash, "prereg.dps_sha256": dsha, "pilot.sampling.dtype": "float16"})
    layers = list(cfg.probe["candidate_layers"])
    recs = make_records(dps, cfg.conditions, tok, cfg, dps_sha=dsha)
    import hashlib
    tsha = hashlib.sha256(json.dumps(topic["prompts"], sort_keys=True).encode()).hexdigest()[:16]
    trecs = make_records(topic["prompts"], [cfg.get("deployment")], tok, cfg, dps_sha=tsha)
    allrecs = recs + trecs
    g = torch.Generator().manual_seed(0); d = 64
    u = torch.randn(d, generator=g); u = u / u.norm()
    shift = {"E1": 1.0, "E1_S1": 1.4, "S1": 0.5, "S2": 0.5, "Dstr": 0.0, "positive_control": 2.0, "e2_a": 0.8, "e2_b": 0.8, "e2_c": 1.0, "sys_neutral": 0.0, "deployment": 0.0}
    base = {}
    acts = {p: {L: torch.zeros(len(allrecs), d) for L in layers} for p in ("last", "last_user", "mean_user")}
    for i, r in enumerate(allrecs):
        b = base.setdefault(r.env_prompt, torch.randn(d, generator=g) * 0.3)
        v = b + 0.05 * torch.randn(d, generator=g) + shift.get(r.condition_id, 0.0) * u
        for p in acts:
            for L in layers:
                acts[p][L][i] = v
    payload = {"dtype": "float16", "layers": layers, "positions": ["last", "last_user", "mean_user"], "keys": [r.prompt_key for r in allrecs],
               "acts": acts, "config_hash": cfg.config_hash, "dps_sha256": dsha, "topic_sha256": tsha, "model_revision": "x"}
    torch.save(payload, out / "probe_acts_fp16.pt")
    # config path: фикстура cfg пишет yaml в tmp — берём его из cfg.path если есть, иначе из фикстуры load_fake
    from tests.fake_config import write_fake
    cfg_path = write_fake(tmp_path / "cfg")
    probe_stage.main(["--config", str(cfg_path), "--dps", str(dps_path), "--topic", str(tpath), "--out", str(out), "--skip-extract", "--n-perm", "300", "--n-boot", "300", "--force-main"])
    n = json.loads((out / "numbers.json").read_text())
    assert n["probe.v1.heldout_auc_pool"] > 0.9
    assert n["probe.v2.pass"] is True
    assert n["probe.v4.pass_legacy_absence_of_evidence"] is True and "probe.v4.tost.frac_ci_lo" in n   # TOST на 4 topic-промптах может быть широким
    assert n["probe.gate_v5"] is True and n["probe.v5.per_insert_ok.e2_c"] is True and n["probe.deviation.main_before_gate1"] is True
    assert abs(n["probe.delta.pool.S1.frac_of_e1"] - 0.5) < 0.2 and abs(n["probe.delta.pool.Dstr.frac_of_e1"]) < 0.2
    assert n["probe.S_minus_Dstr.pool.S1.p"] < 0.05
    assert n["probe.strong.pool.Dstr_within_perm_band"] is True and n["probe.strong.pool.S_above_perm_band"] is True
    assert (out / "probe_scores.csv").exists() and sum(1 for _ in open(out / "probe_scores.csv")) == 1 + 12 * 6
    assert "probe.validations_recorded_at" in n
