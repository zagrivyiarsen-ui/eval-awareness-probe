"""Игрушечный пробе Gate 0: diff-of-means, held-out по env/prompt_id, AUC, random-полоса.
Диагностика позиций, НЕ выбор. S-условия сюда попадать не могут (ревью 1.2)."""
from __future__ import annotations

from typing import Any, Sequence

import torch

from src.conditions import S_IDS


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return ((d > 0).float().mean() + 0.5 * (d == 0).float().mean()).item()


def diff_of_means(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (d.norm() + 1e-8)


def heldout_by_prompt(X: torch.Tensor, y: torch.Tensor, env_prompts: Sequence[str], n_folds: int = 4) -> dict[str, Any]:
    """Сплит по env/prompt_id (ревью 2.4: одинаковые prompt_id в двух средах не должны течь между фолдами)."""
    assert all("/" in p for p in env_prompts), "keys must be env/prompt_id"
    uniq = sorted(set(env_prompts))
    folds = [uniq[i::n_folds] for i in range(n_folds)]
    aucs = []
    for f in folds:
        te = torch.tensor([p in f for p in env_prompts])
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        aucs.append(auc(X[te] @ diff_of_means(X[~te], y[~te]), y[te]))
    return {"heldout_auc_mean": float(torch.tensor(aucs).nanmean()), "heldout_aucs": aucs, "direction": diff_of_means(X, y)}


def random_band(X_contrast, y, X_target, X_ref, n_dirs: int = 10, seed: int = 0) -> dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    d = X_contrast.shape[1]
    shifts = []
    for _ in range(n_dirs):
        v = torch.randn(d, generator=g); v = v / v.norm()
        shifts.append(((X_target @ v).mean() - (X_ref @ v).mean()).abs().item())
    w = diff_of_means(X_contrast, y)
    real = ((X_target @ w).mean() - (X_ref @ w).mean()).item()
    return {"random_band_max": max(shifts), "random_band_mean": sum(shifts) / len(shifts), "probe_shift": real,
            "above_band": abs(real) > max(shifts)}


def position_diagnostic(acts_by_pos: dict[str, torch.Tensor], cond_ids: Sequence[str], env_prompts: Sequence[str],
                        pos_cond="E1", neg_cond="deployment", gen_conds: Sequence[str] = ("e2_a", "e2_b", "e2_c")) -> dict[str, Any]:
    leaked = sorted(set(cond_ids) & set(S_IDS))
    assert not leaked, f"S-conditions passed to position_diagnostic before probe freeze: {leaked}"
    cond = list(cond_ids); pid = list(env_prompts)
    out: dict[str, Any] = {}
    for pos, X in acts_by_pos.items():
        m = torch.tensor([c in (pos_cond, neg_cond) for c in cond])
        y = torch.tensor([1 if c == pos_cond else 0 for c in cond])[m]
        r = heldout_by_prompt(X[m], y, [p for p, k in zip(pid, m) if k])
        w = r["direction"]
        res: dict[str, Any] = {"heldout_auc_e1_vs_deploy": r["heldout_auc_mean"], "heldout_aucs": r["heldout_aucs"]}
        mg = torch.tensor([c in gen_conds for c in cond])
        if mg.any():
            md = torch.tensor([c == neg_cond for c in cond])
            s = torch.cat([X[mg] @ w, X[md] @ w])
            yy = torch.cat([torch.ones(int(mg.sum())), torch.zeros(int(md.sum()))])
            res["auc_e2_vs_deploy_along_e1_dir"] = auc(s, yy)
            # 2026-08-27: по каждому E2 отдельно — e2_c (позиция E1) против e2_a/e2_b различает насыщение и локальность (addendum §2)
            for gc in gen_conds:
                mgc = torch.tensor([c == gc for c in cond])
                if mgc.any():
                    s1 = torch.cat([X[mgc] @ w, X[md] @ w]); y1 = torch.cat([torch.ones(int(mgc.sum())), torch.zeros(int(md.sum()))])
                    res[f"auc_{gc}_vs_deploy_along_e1_dir"] = auc(s1, y1)
        out[pos] = res
    return out


# ---------------------------------------------------------------- 2026-08-27 (ревью R-Gate0 Г-3, А.2, addendum §9–10)
def perm_band(X_contrast: torch.Tensor, y: torch.Tensor, pair_ids: Sequence[str], X_target: torch.Tensor,
              X_ref: torch.Tensor, n_perm: int = 200, seed: int = 0) -> dict[str, Any]:
    """Перестановочная полоса ТОЙ ЖЕ процедуры: метки deployment/E1 переставляются ВНУТРИ пар промптов
    (pair_ids = env/prompt_id), w_perm = diff-of-means, проекция сдвига target−ref. p = доля |proj_perm| ≥ |proj|.
    В отличие от random_band (случайная ось в d≈4096 ловит ~3 % нормы сдвига), это ворота v3."""
    g = torch.Generator().manual_seed(seed)
    w = diff_of_means(X_contrast, y)
    real = ((X_target @ w).mean() - (X_ref @ w).mean()).item()
    pairs = {}
    for i, p in enumerate(pair_ids):
        pairs.setdefault(p, []).append(i)
    projs = []
    for _ in range(n_perm):
        yp = y.clone()
        for idx in pairs.values():
            if len(idx) == 2 and torch.rand(1, generator=g).item() < 0.5:
                yp[idx[0]], yp[idx[1]] = y[idx[1]], y[idx[0]]
        wp = diff_of_means(X_contrast, yp)
        projs.append(((X_target @ wp).mean() - (X_ref @ wp).mean()).item())
    absr = abs(real)
    p_val = (sum(abs(x) >= absr for x in projs) + 1) / (n_perm + 1)
    return {"probe_shift": real, "perm_abs_max": max(abs(x) for x in projs), "perm_abs_q95": sorted(abs(x) for x in projs)[int(0.95 * n_perm) - 1],
            "perm_p": p_val, "n_perm": n_perm}


def shift_frac_and_sign(w: torch.Tensor, X_deploy: torch.Tensor, X_e1: torch.Tensor, X_target: torch.Tensor) -> dict[str, float | bool]:
    """v5-статистика: (mean proj target − deploy)/(mean proj E1 − deploy); знак должен совпадать со знаком E1 (v3_sign_must_match_e1)."""
    d0 = (X_deploy @ w).mean().item(); d1 = (X_e1 @ w).mean().item(); dt = (X_target @ w).mean().item()
    e1 = d1 - d0
    return {"e1_shift": e1, "target_shift": dt - d0, "frac_of_e1": (dt - d0) / e1 if e1 else float("nan"),
            "sign_matches_e1": (dt - d0) * e1 > 0}


def cross_env_auc(X: torch.Tensor, y: torch.Tensor, envs: Sequence[str]) -> dict[str, float]:
    """Правило слоя (analysis.yaml layer_rule): train на одной среде (deployment vs E1), test на другой; среднее по двум направлениям.
    Не использует E2 и S; не насыщается на 1.0 так же легко, как внутрисредовой held-out."""
    envs_t = list(envs); uniq = sorted(set(envs_t))
    assert len(uniq) == 2, uniq
    out = {}
    for a, b in ((uniq[0], uniq[1]), (uniq[1], uniq[0])):
        tr = torch.tensor([e == a for e in envs_t]); te = torch.tensor([e == b for e in envs_t])
        out[f"train_{a}_test_{b}"] = auc(X[te] @ diff_of_means(X[tr], y[tr]), y[te])
    out["mean"] = sum(v for k, v in out.items() if k.startswith("train_")) / 2
    return out


def choose_layer_cross_env(acts_by_layer: dict[int, torch.Tensor], y: torch.Tensor, envs: Sequence[str]) -> dict[str, Any]:
    scores = {L: cross_env_auc(X, y, envs)["mean"] for L, X in acts_by_layer.items()}
    best = max(scores.values())
    ties = sorted(L for L, s in scores.items() if abs(s - best) < 1e-9)
    layer = ties[len(ties) // 2]
    return {"layer": layer, "scores": scores, "tie": len(ties) > 1}
