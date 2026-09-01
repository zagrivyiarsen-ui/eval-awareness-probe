"""ОДНА правка config_hash после пилота: вписывает gate-параметры в conditions.yaml и ассертит, что sampling/dtype/batch совпадают
с тем, чем пилот генерировал (pilot.sampling.* в numbers.json), а k/cap/T — с решением gate1_checks (pilot.decision.*).
    PYTHONPATH=. python scripts/fill_gate_params.py --out results
Затем: PYTHONPATH=. python scripts/verify_prereg.py --dps data/decision_points.json  (без --allow-unset-gate-params)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conditions import load_conditions


def set_scalar(text: str, section: str, key: str, value) -> str:
    val = json.dumps(value) if not isinstance(value, list) else "[" + ", ".join(map(str, value)) + "]"
    # 2026-08-28: было (?ms) с `.*` — DOTALL давал катастрофический бэктрекинг (generation.temperature не находился за минуты)
    pat = re.compile(rf"(?m)^({section}:\n(?:[ \t]+[^\n]*\n)*?)([ \t]+{key}:)[^\n]*")
    m = pat.search(text)
    assert m, f"{section}.{key} not found"
    return text[:m.start(2)] + f"{m.group(2)} {val}   # Gate 1 (fill_gate_params.py)" + text[m.end():]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--k", type=int, default=None); ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--refill", action="store_true", help="R-Launch Д-6: повторная правка yaml (config_change_log уже непуст) — только явно")
    a = ap.parse_args(argv)
    nums = json.loads((Path(a.out) / "numbers.json").read_text(encoding="utf-8"))
    assert not nums.get("prereg.config_change_log") or a.refill, \
        "fill уже выполнялся (prereg.config_change_log непуст) — правка «одна»; повторный запуск только с --refill (второй diff уйдёт в change log)"
    ps = {k.split("pilot.sampling.")[1]: v for k, v in nums.items() if k.startswith("pilot.sampling.")}
    k = a.k or nums.get("pilot.decision.k"); cap = a.cap or nums.get("pilot.decision.cap")
    T = a.temperature if a.temperature is not None else ps["temperature"]
    assert k and cap, "k/cap: gate1_checks.py не запускался"
    td = str(nums.get("pilot.temperature_decision", ""))
    assert td.startswith("keep") or a.temperature is not None, f"температура не решена: {td}"
    layers = nums.get("gate0.candidate_layers") or [19, 21, 23]
    p = Path(a.config); s = p.read_text(encoding="utf-8")
    for sec, key, val in (("generation", "temperature", T), ("generation", "max_new_tokens", cap), ("generation", "k", k),
                          ("generation", "batch_size", ps["batch_size"]), ("generation", "top_p", ps["top_p"]), ("generation", "top_k", ps["top_k"]),
                          ("generation", "repetition_penalty", ps["repetition_penalty"]),
                          ("generation", "presence_penalty", ps["presence_penalty"]),      # R-Launch Д-6: один источник и для presence
                          ("probe", "candidate_layers", layers)):
        s = set_scalar(s, sec, key, val)
    if not re.search(r"(?m)^model:\n(?:[ \t]+.*\n)*?[ \t]+dtype:", s):
        s = s.replace("model:\n", f"model:\n  dtype: {ps['dtype']}   # Gate 0 (gate0_fp16_checks.py)\n", 1)
    # 2026-08-28: та же ОДНА правка config_hash приводит probe.layer_rule в соответствие analysis.yaml (cross-env, addendum §9)
    s = set_scalar(s, "probe", "layer_rule", "cross_env_heldout_auc_deploy_vs_e1_tie_middle")
    p.write_text(s, encoding="utf-8")
    # 2026-08-28 (ревью R-Pipeline, блокер 2): заполненный yaml живёт только в репозитории, который Kaggle-ячейки распаковывают
    # из блоба заново → копия в results/ (попадает в бэкап); ячейки после Gate 1 восстанавливают её в configs/ до make_dps.
    (Path(a.out) / "conditions_filled.yaml").write_text(s, encoding="utf-8")
    cfg = load_conditions(a.config)          # без allow_unset — всё должно быть заполнено
    assert cfg.generation["k"] == k and cfg.generation["max_new_tokens"] == cap and float(cfg.generation["temperature"]) == float(T)
    print(f"вписано: k={k} cap={cap} T={T} bs={ps['batch_size']} top_p={ps['top_p']} top_k={ps['top_k']} dtype={ps['dtype']} layers={layers}")
    print(f"копия: {Path(a.out) / 'conditions_filled.yaml'} (восстанавливается ячейками B5+ в configs/)")
    print(f"новый config_hash={cfg.config_hash} (был {nums.get('prereg.config_hash')}) → запустить verify_prereg.py; diff попадёт в prereg.config_change_log")


if __name__ == "__main__":
    main()
