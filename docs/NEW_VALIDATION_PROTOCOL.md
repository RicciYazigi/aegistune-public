# Next validation: new wording and equal ticket delivery

Status: Historical protocol completed for P13 (Challenge48); see [Challenge48 results](CHALLENGE48_RESULTS.md) and [follow-up evaluations](FOLLOWUP_EVALUATIONS.md) for outcomes.

Keep the selected SFT v3 weights and original instructions. Reject operational_v1
for v3 following the four-arm development result. Preserve historical TEST200.

Author a separate challenge set before running any model on it. Target 48 cases:
six task families, four scenarios per family per language (English/Spanish).
Use newly written requests and ticket bodies, not old templates with renamed IDs.
Record authoring provenance and overlap checks against train/dev/test. The author
already knows previous aggregate results: do not claim blind independent authorship.
Once validated and frozen, run once without selecting or rewriting cases from outcomes.

Compare original-prompt base and original-prompt v3. An additional base with the
operational addendum can be reported as a separate recipe; its inclusion must be
decided before freezing the execution package, not from challenge outcomes.

Both agents receive the same initial ticket payload, wrapped as untrusted data,
before their first generated action. The host prelude is separate from model
attempts, receipts and turn counts. It performs no tool call and receives no
model success credit. Hash the payload and record delivery for each arm.
Four subsequent generated turns remain available to each agent.

`scripts/equal_exposure.py` prepares only whitelisted, in-scope ticket fields.
The evaluator now supports `--ticket-delivery equal_ticket_delivery_v1`; the
default remains `agent_reads`. Each trajectory records the prelude separately,
and the summary records the protocol and number of payloads delivered. Production
flow tests with scripted generation cover both delivery modes and both prompt
variants, including the exact initial message and unchanged three-action episode.
These are fixture checks, not real-model challenge results. No oracle, family label,
internal ledger notes or expected answer may enter this prelude.

Report "ticket payload delivered" separately from voluntary get_ticket execution
and from the historical receipt-based exposure metric. Delivery does not prove
that the model attended to or understood the text. Equal exposure changes the
protocol: results are not directly interchangeable with historical TEST200.

Before paid execution: finish dataset authorship, validate all oracle scenarios
using positive and deliberately wrong scripted actions, integrate the prelude
with recorded provenance, exercise production evaluator fixtures, check context
budgets, freeze sources/data and prepare a bounded supervised launcher.
Supplement the nonempty-draft predicate with an explicit semantic review rubric;
keep that review separate from automated mission scores. No new GPU job is ready.
