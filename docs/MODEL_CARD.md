# Model card — selected AegisTune SFT v3

## Identity

- Base: `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`.
- Adapter SHA-256: `269f6cc06c31506e651fc146c06ec675b6e21f012d119dd5a93e76a3c724a81a`.
- PEFT LoRA: rank 8, alpha 16, dropout 0, `q_proj`/`v_proj`, 1,916,928 parameters.
- Languages: English and Spanish. Domain: synthetic support and billing tools.
- Training: Transformers/PEFT SFT. Genuine AgentTune GRPO experiments are separate; the headline improvement is not attributed to GRPO.

## Training and selection

Offline privileged teachers supply 400 train missions and 1,200 action rows. Oracle labels are not runtime inputs. Host/prompt tokens are masked; assistant targets receive causal loss. V1 trained 300 steps at 5e-5. V3 continued v1 for 100 steps at 1e-5, with tool-name weight 8 and corrected staging reasons; it did not continue rejected v2.

Selection used 60 dev cases, followed by a frozen 200-case reserved evaluation. Later private experiments did not change these published weights. [Reproduction](REPRODUCTION.md) and [follow-up provenance](FOLLOWUP_EVALUATIONS.md) explain historical generation-argument limitations.

## Historical reserved result

| Measure | Base | v3 |
|---|---:|---:|
| Mission checks passed | 77/200 | 182/200 |
| Benign mission checks passed | 4/100 | 82/100 |
| Unauthorized-effect episodes | 17/200 | 0/200 |
| Valid action format | 71.29% | 100% |

All 17 base effects occurred in benign tasks; adversarial effects were zero for both arms. All 18 v3 residual mission failures were Spanish. A supplementary audit found two reversible intent deviations missed by the legacy proposal counter. Base and adapter had unequal ticket exposure.

## Follow-up limitations

On a non-blind, equal-initial-exposure 48-case challenge, base passed 25 mission checks with zero effects; v3 passed 40 with one unauthorized simulated refund: USD 98.63 instead of USD 28.63. All eight reviewed v3 inquiry drafts were uninformative despite passing the nonempty-text predicate.

P15 confirmed that 48 inquiry drafts omitted balances present in their generation inputs, across v3/P14 and two wrappers. P16 showed conditional preference for observed balances under forced teacher prefixes in known train contexts, without demonstrating an autonomous repair. These findings constrain interpretation of the unchanged historical benchmark.

A later 200-step opening repair produced 4/12 grounded dev inquiry answers but regressed dev mission completion from 56/60 to 51/60, introduced malformed outputs and failed sensitivity criteria. It was rejected; v3 remains selected.

## Intended use

Research into tool proposals, host permissions, actual effects, user intent and evaluator blind spots. Every episode has fresh synthetic state. The model is not the permission boundary and connects to no real payment or messaging infrastructure.

Not suitable for autonomous production payments. One seed, related synthetic templates, explicit policy cues, incomplete semantic scoring and language imbalance constrain generalization. Correct JSON, a permitted operation or a nonempty draft can still be wrong for the customer.

## License and access

Upstream weights and the adapter remain subject to CC-BY-NC 4.0 with the upstream Acceptable Use Addendum/AUP and attribution requirements. The base retains its own access conditions; a public ungated adapter does not remove them. Project code is Apache-2.0 and does not relicense third-party weights. See repository notices and the adapter LICENSE.md for preserved terms.

[Selected adapter](https://huggingface.co/RicciYazigi/aegistune-tiny-aya-lora) · [Original results](RESULTS_TEST200.md) · [Follow-up evaluations](FOLLOWUP_EVALUATIONS.md)
