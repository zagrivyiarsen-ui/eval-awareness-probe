"""Верификация ЗАМОРОЖЕННОГО conditions.yaml (после заполнения TODO_PREREG, включая E2 и positive_control):
    PYTHONPATH=. python scripts/verify_prereg.py --dps data/decision_points.json --allow-unset-gate-params
Пишет prereg.* (config_hash, dps_sha256, sha шаблона, последний токен, инструкции S) в results/numbers.json,
рендеры в results/prereg_renders.txt, pip freeze в results/env_freeze.txt."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import code_sha256
from src.conditions import MAIN_IDS, conditions_hash, load_conditions, text_placeholders
from src.contracts import dps_sha256, env_info, make_records, record_numbers, save_records
from src.scoring import analysis_hash
from src.prompts import (assert_single_insertion, check_forbidden_terms, instruction_sentences, load_tokenizer, render_text,
                         template_has_date, template_sha256)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verify_prereg")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml")
    ap.add_argument("--dps", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--allow-unset-gate-params", action="store_true")
    ap.add_argument("--allow-dps-change", default=None, metavar="REASON",
                    help="2026-08-28: смена dps_sha256 после первой записи — только явно, с причиной в change log")
    a = ap.parse_args(argv)
    cfg = load_conditions(a.config, allow_unset_gate_params=a.allow_unset_gate_params)
    assert not text_placeholders(cfg.conditions, ("main", "positive_control", "e2")), "placeholders remain"
    tok = load_tokenizer(cfg)
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8"))
    dsha = dps_sha256(a.dps)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for c in cfg.conditions:
        assert not check_forbidden_terms(c, cfg.lexicon), c.id
    instr = {cid: sorted(instruction_sentences(cfg.get(cid))) for cid in ("S1", "S2", "Dstr")}
    assert instr["S1"] == instr["S2"] == instr["Dstr"], \
        f"инструкции-предложения различаются между S1/S2/Dstr (ревью 1.6): {instr}"
    if instr["S1"]:
        log.warning("S-шаблоны содержат инструкцию %r, которой нет в deployment — записать в Method как известную асимметрию", instr["S1"])

    recs = make_records(dps, cfg.conditions, tok, cfg, dps_sha=dsha)
    by = {r.prompt_key: r for r in recs}
    n_ins = []
    for dp in dps:
        k = f"{dp['env']}/{dp['prompt_id']}/"
        for base, new in (("deployment", "E1"), ("S1", "E1_S1")):
            j1, j2 = assert_single_insertion(by[k + base].input_ids, by[k + new].input_ids, tok)
            n_ins.append(j2 - j1)
            assert cfg.get("E1").insert_text in tok.decode(by[k + new].input_ids[j1:j2])
    # --- 2026-08-27 (ревью R-Gate0 Б): `actions` идентичны во всех 6 main-версиях НА УРОВНЕ ТОКЕНОВ; e2_c — одна вставка
    def _sub(seq, sub):
        n = len(sub)
        return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))
    keys_all = json.loads(Path(a.dps).with_name(Path(a.dps).stem + "_keys.json").read_text(encoding="utf-8"))["keys"]
    kmap = {f"{k['env']}/{k['prompt_id']}": k for k in keys_all}
    n_actions_tok = set()
    for dp in dps:
        k = f"{dp['env']}/{dp['prompt_id']}/"
        act_ids = tok(dp["actions"], add_special_tokens=False)["input_ids"]
        variants = {act_ids[0]} | {tok("\n" + dp["actions"], add_special_tokens=False)["input_ids"][1]}
        for cid in MAIN_IDS:
            ids = by[k + cid].input_ids
            ok = _sub(ids, act_ids) or any(_sub(ids, [v] + act_ids[1:]) for v in variants)
            assert ok, f"{k}{cid}: `actions` не совпадает токен-в-токен (BPE на границе заголовка)"
        n_actions_tok.add(len(act_ids))
        j1, j2 = assert_single_insertion(by[k + "deployment"].input_ids, by[k + "e2_c"].input_ids)
        assert cfg.get("e2_c").insert_text in tok.decode(by[k + "e2_c"].input_ids[j1:j2])
        # утечка ключей в рендер (Б, тупые): answer_key / секрет как отдельное число
        kk = kmap[k[:-1]]
        for cid in MAIN_IDS:
            txt = tok.decode(by[k + cid].input_ids)
            if "answer_key" in kk:
                assert kk["answer_key"] not in txt
            if "secret" in kk:
                import re as _re
                assert not _re.search(rf"(?<!\d){kk['secret']}(?!\d)", txt), f"{k}{cid}: секрет {kk['secret']} виден в рендере"
    lasts = {(r.last_token_id, r.last_token_str) for r in recs}
    assert len(lasts) == 1, f"last token differs across versions: {lasts}"
    (lid, lstr), = lasts
    preopens = render_text(recs[0].messages, tok, cfg).rstrip().endswith("<think>")

    lines = [f"config_hash={cfg.config_hash}", f"dps_sha256={dsha}", f"tokenizer={cfg.tokenizer['name']} revision={cfg.tokenizer.get('revision')}",
             f"chat_template_sha256={template_sha256(tok)}", f"template_date_tokens={template_has_date(tok)}",
             f"last_token_id={lid} last_token_str={lstr!r} template_preopens_think={preopens}",
             f"lexicon_deviations={list(cfg.lexicon.deviations)}", f"instruction_sentences_S={instr['S1']}", ""]
    for c in cfg.conditions:
        txt = render_text(by[f"{dps[0]['env']}/{dps[0]['prompt_id']}/{c.id}"].messages, tok, cfg)
        lines += [f"===== {c.id} (group={c.group}) sha256={hashlib.sha256(txt.encode()).hexdigest()[:16]}", txt, ""]
    (out / "prereg_renders.txt").write_text("\n".join(lines), encoding="utf-8")
    save_records(recs, out / "prompt_records.jsonl")
    try:
        (out / "env_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    except Exception as e:  # noqa: BLE001
        log.warning("pip freeze failed: %s", e)
    prev = json.loads((out / "numbers.json").read_text(encoding="utf-8")) if (out / "numbers.json").exists() else {}
    change_log = list(prev.get("prereg.config_change_log") or [])
    # 2026-08-28 (ревью R-Pipeline А.7): раньше dps_sha256/code_sha256 молча перезаписывались
    if prev.get("prereg.dps_sha256") and prev["prereg.dps_sha256"] != dsha:
        if not a.allow_dps_change:
            raise SystemExit(f"dps_sha256 изменился: {prev['prereg.dps_sha256']} -> {dsha}; make_dps.py другой? --allow-dps-change REASON")
        change_log.append(f"dps {prev['prereg.dps_sha256']}->{dsha}: {a.allow_dps_change}")
    if prev.get("prereg.analysis_hash") and prev["prereg.analysis_hash"] != analysis_hash():
        change_log.append(f"analysis {prev['prereg.analysis_hash']}->{analysis_hash()} argv={' '.join(sys.argv[1:])}")
    cur_code = code_sha256(Path(__file__).resolve().parents[1])
    if prev.get("prereg.code_sha256") and prev["prereg.code_sha256"] != cur_code:
        change_log.append(f"code {prev['prereg.code_sha256']}->{cur_code} argv={' '.join(sys.argv[1:])}")
    if prev.get("prereg.config_hash") and prev["prereg.config_hash"] != cfg.config_hash:
        change_log.append(f"{prev['prereg.config_hash']}->{cfg.config_hash} argv={' '.join(sys.argv[1:])}")
    nums = record_numbers(out / "numbers.json", {
        "prereg.analysis_hash": analysis_hash(), "prereg.conditions_hash": conditions_hash(a.config), "prereg.code_sha256": cur_code,
        "prereg.config_change_log": change_log, "prereg.n_actions_tokens_by_dp": sorted(n_actions_tok),
        "prereg.config_hash": cfg.config_hash, "prereg.dps_sha256": dsha, "prereg.chat_template_sha256": template_sha256(tok),
        "prereg.last_token_id": lid, "prereg.last_token_str": lstr, "prereg.template_preopens_think": preopens,
        "prereg.e1_insert_n_tokens": sorted(set(n_ins)), "prereg.n_prompt_versions": len(recs),
        "prereg.template_date_tokens": template_has_date(tok), "prereg.lexicon_deviations": list(cfg.lexicon.deviations),
        "prereg.instruction_sentences_S": instr["S1"], **env_info()})
    print(json.dumps({k: v for k, v in nums.items() if k.startswith(("prereg.", "env."))}, indent=2, ensure_ascii=False))
    if cfg.tokenizer.get("chat_template_sha256") is None:
        print(f"\n>>> впиши в conditions.yaml: tokenizer.chat_template_sha256: {template_sha256(tok)}")
    print(f">>> рендеры: {out / 'prereg_renders.txt'}; записи: {out / 'prompt_records.jsonl'}")


if __name__ == "__main__":
    main()
