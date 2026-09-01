"""Таблица структурной плотности (план этап 3, ревью R-Pipeline Б «Dstr — контроль»): заголовки / нумерация / маркеры списков
на 100 токенов промпта для всех 6 main-версий каждого DP (+ косинусная близость mean_user-активаций S1/S2/Dstr к deployment,
если есть results/probe_acts_fp16.pt — secondary). 0 GPU. Пишет prereg.density.* и results/structure_density.md.
    PYTHONPATH=. python scripts/structure_density.py --dps data/decision_points.json --out results"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.conditions import MAIN_IDS, load_conditions
from src.contracts import dps_sha256, make_records, record_numbers
from src.prompts import load_tokenizer, render_text

MARKERS = {"headers": r"(?m)^\s*(#{1,6}\s|[A-Z][A-Za-z ]{1,30}:\s*$)", "numbering": r"(?m)^\s*\d+[.)]\s", "bullets": r"(?m)^\s*[-*•]\s",
           "blank_lines": r"\n\s*\n", "colon_lines": r"(?m)^[^\n:]{1,40}:\s*$"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conditions.yaml"); ap.add_argument("--dps", required=True); ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    out = Path(a.out); cfg = load_conditions(a.config, allow_unset_gate_params=True); tok = load_tokenizer(cfg)
    dps = json.loads(Path(a.dps).read_text(encoding="utf-8"))
    recs = make_records(dps, [cfg.get(c) for c in MAIN_IDS], tok, cfg, dps_sha=dps_sha256(a.dps))
    acc = defaultdict(lambda: defaultdict(list))
    for r in recs:
        txt = render_text(r.messages, tok, cfg)
        user = txt.split("<|im_start|>user", 1)[-1].split("<|im_end|>", 1)[0] if "<|im_start|>user" in txt else txt
        n100 = max(1, r.n_prompt_tokens) / 100
        for k, pat in MARKERS.items():
            acc[r.condition_id][k].append(len(re.findall(pat, user)) / n100)
        acc[r.condition_id]["n_prompt_tokens"].append(r.n_prompt_tokens)
    D = {c: {k: mean(v) for k, v in acc[c].items()} for c in MAIN_IDS}
    sim = {}
    acts = out / "probe_acts_fp16.pt"
    if acts.exists():
        try:
            import torch
            pl = torch.load(acts, weights_only=False); keys = pl["keys"]; L = pl["layers"][len(pl["layers"]) // 2]
            X = pl["acts"]["mean_user"][L]; idx = {k: i for i, k in enumerate(keys)}
            for c in ("S1", "S2", "Dstr", "E1", "E1_S1"):
                cs = []
                for dp in dps:
                    k0 = f"{dp['env']}/{dp['prompt_id']}/deployment"; k1 = f"{dp['env']}/{dp['prompt_id']}/{c}"
                    if k0 in idx and k1 in idx:
                        cs.append(torch.nn.functional.cosine_similarity(X[idx[k0]][None], X[idx[k1]][None]).item())
                sim[c] = {"cos_mean_user_vs_deployment_mean": mean(cs), "min": min(cs), "layer": L, "n": len(cs)}
        except Exception as e:  # noqa: BLE001
            sim["error"] = str(e)[:200]
    record_numbers(out / "numbers.json", {"prereg.density": D, "prereg.density_similarity": sim or {"note": "probe_acts not found"}})
    cols = list(MARKERS) + ["n_prompt_tokens"]
    L = ["# Structural density per 100 prompt tokens (user turn), mean over DPs", "", "| condition | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for c in MAIN_IDS:
        L.append(f"| {c} | " + " | ".join(f"{D[c][k]:.2f}" for k in cols) + " |")
    if sim:
        L += ["", "Cosine similarity (mean_user activations, middle candidate layer) vs deployment: " + json.dumps(sim, default=str)]
    L += ["", "Plan stage 3: S1, S2, Dstr comparable in density, else Dstr is not a structural control; similarity > 0.85 expected (insertions excluded)."]
    (out / "structure_density.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
