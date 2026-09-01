# Deviations from the pre-registration

Every departure from the frozen plan, with the reason and where the evidence sits. Ordered by
how much each one could affect a reader's conclusions.

## Design

**1. The behavioural arm was cancelled (pivot 4.2).** Two independent stop signals fired on the
same day: the positive control was dead after escalation (0/57 pooled, upper 95 % CI 0.063) and
the recorded compute-budget rule returned `decision.fits = false` (14.19 h required against 14.0
available). The two behavioural confirmatory tests were declared void. Everything reported after
that point is **exploratory**; the only surviving confirmatory family is the `S − Dstr` pair with
Holm correction.

**2. Four of five pre-registered validations failed.** v1 (separability) passed; v2
(monotonicity), v3 (permutation gate), v4 (topical control) and v5 (generalisation to other
insertion wordings) did not. The work continued along the branch described in advance for that
outcome — "the direction did not validate" — rather than being reframed around a success.

**3. The layer-selection rule degenerated.** Cross-environment AUC was 1.000 / 1.000 / 0.984 for
the three candidate layers, so the tie-break by median selected L21. The criterion was blind to
contamination, which varies by 40 % across those same layers.

**4. Multiplicity was declared and not corrected.** Eight forks, dozens of exploratory p-values.
Only the `S − Dstr` pair is treated as confirmatory, with Holm.

## Execution

**5. fp16 accepted despite a failed check.** `greedy_prefix_single_match` was
[32,32,32,32,32,32,27,1] against a threshold of 16. All other checks passed: KL median 6.6e-06,
top-1 agreement 1.0, `checkG_bs24` 64/64, non-finite 0, max|h| 347. Reason: bf16 on a T4 ran at
~14 tok/s and did not fit the budget before the deadline; dtype is constant across all conditions
and therefore invariant to the contrast. The decision was taken *before* any real generation,
under the recorded rule `gate0.dtype_decision.rule`.

**6. `ask_model` was run at T = 0** while the behavioural-arm rule specified T = 1.0. The rule
belongs to the cancelled arm; the mode is "answer without reasoning" with an empty think prefill.

**7. The rescue analysis of truncated generations was proposed by external code review after the
pilot** — post-hoc, and marked as such.

**8. The answer classifier used in the steering analysis was written after inspecting the
outputs.** It reclassified 16 of 512 generations, all of them long-form negations that a
`\b(yes|no)\b` regex had missed, with zero ambiguous cases. Post-hoc, and marked as such.

## Data and environment

**9. `input_ids` is absent for all 137 additional records.** The token-level analysis therefore
covers 384 of 529; `token_only_predictor` and η² are not claimed. The `last_user` conclusion
rests on the e2 conditions instead.

**10. The leading out-of-sample procedure was added after the fact.** Leave-one-decision-point-out covering all fifteen cells was computed in a repair stage (`scripts/p3_repair.py`) after an earlier snapshot flagged the gap. It reads the frozen activations and writes only under `probe.exploratory.p3.*`.


**11. The `wild_ai` cell is empty (n = 0).** The pre-registered regex selected no Dolly prompts.
The neutral pool is `n` + `wild_other`, and the AI-topicality question is answered instead by
comparing `b` against `c1`/`c2`.

**12. The external negative set was not enlarged.** Extending Dolly from 37 to several hundred
was attempted. The code that composed Dolly rows into prompts was not preserved in the snapshot
and had to be reconstructed from the 37 existing records: 35 of 37 are recovered exactly or up to
the position of one internal line break, two are truncations of longer contexts under a rule that
could not be recovered. Rather than draw a larger sample under a composition that provably
differs from the original, the FPR is reported on the original 37 with its Wilson interval.

**13. `env.transformers` is overwritten** by whichever session last touched the snapshot.
Generation and activation extraction were performed on transformers 5.16.1.

**14. `prereg.density_similarity` was computed without activations**, so the cosines of S/Dstr to
deployment may come from an earlier run. The markup densities themselves are valid.

**15. NLL and token lengths exist for 209 of 529 records.**

**16. Two scales coexist in the snapshot** — normalised (‖w‖ = 1) and unnormalised. `frac_of_e1`
is invariant to the choice and comparable everywhere; absolute shifts are not. See section 0 of
`results/FINAL_DUMP_EN.md`.

**17. Numbers differ in the third decimal between recomputations** of the same quantities from
the same activations. A single snapshot is quoted throughout.

## Provenance of the material

**18. 58 prompts were written by one author in one day after the result on the original eight was
known.** Mitigation, not elimination: the pre-registered `legacy` negatives give FPR 0.250 —
*higher* than the post-hoc authored ones (0.182) and the external Dolly set (0.162).

**19. The external set was pre-registered**: source (Databricks Dolly-15k), sha, seed and the
55–85 word filter were recorded before selection.

**20. The category "interpret a person's measured numbers" was defined after looking at the
scores.** Five of the six firing Dolly prompts fall in it. The blind-recoding package and its
frozen rule are prepared (`results/dolly_for_blind_coding.md`); the labels are not in, so the
category is not claimed.
