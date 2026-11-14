# P23: equal initial ticket delivery and a development-selected prompt baseline

P23 evaluates the frozen SFT v3 adapter against the base model with original instructions and a prompt selected exclusively on development data. No new training or adapter promotion occurred. This supplements TEST200; its denominators and protocol remain separate.

## Protocol and selection

Three base prompts (original, operational_v1, schema_fewshot_v1) ran on DEV60, 180 episodes. The few-shot prompts use six training-only demonstrations in total: three English examples in the English prompt and three Spanish examples in the Spanish prompt. The preregistered selection ranks factual/procedural passes, then fewer historical unauthorized-effect episodes, intent deviations, parse-error episodes and prompt length. All three scored zero strict passes. Few-shot won on effects and intent, not by adding historical mission completion as a selection criterion.

| DEV60 /60 | Original | Operational | Schema few-shot |
|---|---:|---:|---:|
| Historical mission checks |10|6|30|
| Historical unauthorized effects |21|5|5|
| Intent-deviation episodes |40|33|11|
| Parse-error episodes |27|26|12|
| Strict factual/procedural passes |0|0|0|

The selected prompt was frozen before Exam72. Exam72 contains 72 internally authored cases (36 bilingual pairs), six familiar synthetic families. Authoring was informed by prior findings; this is not an external blind evaluation or evidence on unseen families. All three arms received identical initial ticket payloads, four turns, context 4096, output limit 128, greedy decoding and explicit EOS. Each episode used a fresh sandbox. Subsequent tool choices and ledger exposure may differ.

## Exam72 results

| Measure /72 episodes | Base original | Base schema few-shot | Frozen SFT v3 |
|---|---:|---:|---:|
| Historical mission checks |47 (65.3%)|43 (59.7%)|50 (69.4%)|
| Historical unauthorized-effect episodes |1|2|0|
| Parse-error episodes |11|5|0|
| Intent-deviation episodes |9|6|12|
| Strict factual/procedural passes, pending review excluded |0|0|4|
| Drafts pending human semantic adjudication |0|0|4|

V3 completed three more historical missions than the original base (+4.17 percentage points) and seven more than the dev-selected few-shot baseline (+9.72 points). The selected prompt did not outperform the original on this exam. This is a result for these candidates, not a general failure of prompt engineering. V3 had fewer observed effects and parse errors but MORE intent-deviation episodes. Zero effects is not perfect intent alignment or production safety. Challenge48 previously exposed an incorrect v3 simulated refund; this result does not erase it.

Strict scoring requires prior get_ticket and lookup_account_ledger tool calls in addition to correct actions/content, even with the ticket delivered initially. It is not standalone textual factual accuracy. A zero strict score does not imply every answer is factually false. The rubric's no_unauthorized_effect field includes additional action restrictions and is not the complement of the historical effect counter. Historical completion still uses the weaker task predicate and does not establish balance-answer quality.

The four pending v3 drafts are Spanish adversarial refusals, not balance answers. They remain pending; no human-adjudicated 7/72 result is claimed. The automatic summary is preserved unchanged. Review proposals by an AI are not human adjudication.

## Interpretation and boundaries

This experiment addresses initial-ticket exposure and a worked prompt baseline more directly than TEST200. It does not establish the historical +52.5-point gain under the new protocol, a universal advantage over prompting, a causal mechanism of numeric avoidance, or a new reward-hacking defence. Shared synthetic families, one training seed and correlated bilingual cases remain limitations. Evaluator blind spots are demonstrated by observed outputs, not by inferring the model's motives.

The historical TEST200 score remains 182/200 versus 77/200 under its original protocol. Keep it beside this smaller effect; do not pool datasets into a new overall rate. All principal gains attributed to training come from Transformers/PEFT SFT; AgentTune's bounded GRPO pilot is separate. AuditKIT supports trajectory auditing.

## Execution provenance

DEV60 v5 failed without usable scientific result downloads; its generated Space entrypoint lacked an HTTP health listener. The missing downloads do not establish that inference never started. DEV60 v6 produced all 180 episodes but exited during a final generation-ID uniqueness check; deterministic generation IDs were confused with distinct episode IDs. The downloaded trajectories were preserved and selection finalized offline with v7. Its original failed receipt remains failed. Exam72 v7 completed successfully with 216 episodes, ten downloaded artifacts and provider-confirmed PAUSED state. No additional inference was required for the DEV60 recovery.

Public evidence uses the p23/ prefix in the evaluation-evidence dataset, and p23_exam72/ in the benchmark dataset, once the accompanying package is deployed. Until deployment, these are package destinations, not a claim that remote files already exist. The evidence package contains sanitized derivatives of receipts and metadata; originals and original remote artifact hashes are retained locally. Redaction changes hashes and must not be mistaken for byte-identical remote artifacts.

The selected adapter SHA-256 is 269f6cc06c31506e651fc146c06ec675b6e21f012d119dd5a93e76a3c724a81a; base revision af89d219b53ed9b13b8a4645f9c8028973510324. No replacement weights or multimedia are included.
