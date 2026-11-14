# Submission abstract

Submission body: 346 whitespace-delimited words, below 500.

AegisTune studies a billing-agent failure mode: a technically permitted tool call can still perform the wrong business action. We adapt Cohere Labs Tiny Aya Global with supervised LoRA training and evaluate English and Spanish workflows through a deterministic gateway, recorded attempts, execution receipts and independent sandbox outcomes.

On a frozen synthetic test (200 paired cases, 400 episodes), SFT v3 passed 182 historical mission checks (91%), versus 77 (38.5%) for the base model. Unauthorized-effect episodes were 0 versus 17; all 17 base effects occurred in benign tasks, while adversarial effects were zero for both models. All 18 adapter mission failures were Spanish. A supplementary audit identified two reversible intent deviations missed by the original counter.

A further experiment selected among three base prompts on 60 development cases (180 episodes), then compared the selected few-shot prompt, original base prompt and frozen v3 on 72 internally authored bilingual cases (216 episodes). All arms received identical initial ticket payloads. Few-shot examples were training-only: three per language. Historical mission completion was 43/72 for the selected prompt, 47/72 for the original and 50/72 for v3. Unauthorized-effect episodes were 2, 1 and 0; parse-error episodes were 5, 11 and 0. V3 nevertheless had more intent deviations: 12 versus 6 and 9. Strict factual/procedural passes were 0, 0 and 4, with four v3 drafts pending human semantic review. That rubric also requires prior tool reads; it is not isolated answer accuracy.

Earlier Challenge48 diagnostics exposed a simulated v3 refund of USD 98.63 instead of USD 28.63, and generic drafts passed a nonempty-text predicate while omitting balances. Repair candidates were not promoted. P17 recovered 4/12 development balance answers (4/6 English, 0/6 Spanish) but reduced historical mission completion to 51/60. These findings motivate distinct measures for permissions, effects, intent and useful content.

AgentTune supports a separate bounded GRPO pilot and AuditKIT supports trajectory auditing; measured training gains come from Transformers/PEFT SFT. Public artifacts expose results and failures. Limitations include one training seed, shared synthetic families, correlated bilingual pairs, historical exposure differences and incomplete semantic scoring. This is an inspectable research result, not a production payment-safety guarantee.

Primary: Track 3.7. Secondary: Track 3.9 (evaluator blind spots, not a demonstrated general defence).
