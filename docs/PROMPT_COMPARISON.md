# Development comparison: model weights and operational instructions

This experiment asks how much of the workflow gain can be obtained through better
instructions alone, and whether the selected adapter benefits from those same
instructions. It does not train or change the selected SFT v3 weights.

| Arm | Weights | Prompt |
|---|---|---|
| base_original | Original Tiny Aya Global | Historical instructions |
| base_operational | Original Tiny Aya Global | Historical + operational_v1 |
| v3_original | Selected SFT v3 | Historical instructions |
| v3_operational | Selected SFT v3 | Historical + operational_v1 |

All four arms use all 60 known development cases, one base instance, the same
tokenizer, greedy decoding, context 4096, output 128 and four turns. An isolated
sandbox is created per episode. The adapter is attached once, after both base
arms complete. No model selection or changes use the published TEST200 here.

The bilingual operational addendum clarifies integer minor units, task-specific
tool choice, supervisory proposals versus payments, untrusted ticket text and
the difference between permission and authorization. It contains no case IDs,
hidden oracle labels, family labels or per-case solutions. The candidate was
written from the public tool contracts and known development diagnostics after
the original test was published. It is not a prompt-optimization search, a proven
best prompt, or independent confirmatory evidence.

Both prompts use the corrected **explicit_resolved_eos_v1** policy, passing
resolved terminators directly to generation. This policy differs from the
historical evaluator's inherited defaults; new scores must not replace historical
TEST200 figures or attribute any change solely to prompting. Comparisons within
this new experiment share the same explicit policy.

```sh
python scripts/run_prompt_comparison.py --adapter-path PATH_TO_ADAPTER --preflight-only
python scripts/run_prompt_comparison.py --adapter-path PATH_TO_ADAPTER --device cuda --output-dir outputs/prompt_dev60
```

The command does not provision hardware. The output directory must not exist.
Dataset and adapter hashes are verified before model loading. Initial prompts
are checked against the actual tokenizer, then the actor checks each later turn
without silent truncation. Prompt addenda, hashes, initial message hashes, runtime
and progress are recorded. Completed arms survive a later failure; a partial run
does not produce a completed comparison.

Reports retain native evaluator metrics and the supplementary intent audit as
separate definitions. They include paired discordant outcomes, language/family/
cohort breakdowns, and canonical per-case action metrics including ticket exposure,
gateway denials, parsing and the original proposal counter. Every episode remains
available for inspection. Report counts and denominators, not only aggregate reward.

Interpretation remains limited by known synthetic templates, one seed, unequal
ticket exposure and the nonempty-draft predicate. This experiment does not itself
establish injection robustness or generalization. A subsequent independently
authored evaluation is required before making stronger claims.

## Completed development result

The four-arm run completed on 10 November 2026. See
[results and prompt-sensitivity analysis](PROMPT_COMPARISON_RESULTS.md).
The operational addendum is rejected for the selected adapter; the original
prompt remains selected. A passing syntax check does not establish correct actions.
