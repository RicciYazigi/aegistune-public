# Positioning and hackathon fit

## The thesis

**A billing agent should be judged by whether it performs the authorized work,
not by whether its JSON parses or its payment fits a technical limit.**

AegisTune is an outcome-audited business-workflow prototype. Its strongest
measured result is legitimate usefulness:82/100benign tasks completed versus4/100
for the base, inside the same deterministic host envelope.

## What distinguishes this contribution

1. **Independent consequence evidence.** Attempts, gateway decisions and actual receipts remain separate. A model's claim of success is not the score.
2. **Useful work is part of safety.** Refunds and supervisory proposals must actually happen when authorized; blanket drafting loses utility.
3. **Inspectable before/after behavior.** The recorded replay includes all200paired cases, failed cases, exact model output and source hashes.
4. **An audit of the audit.** The supplementary intent check exposes two remaining errors hidden by the original zero-proposal counter.
5. **A preserved negative result.** V2 learned staging but regressed globally. The selected v3 returned to v1 and gained utility without hiding that failure.
6. **A measured language gap.** English100/100and Spanish82/100are reported separately rather than described as uniform multilingual safety.

These distinguish the project's design and evidence. They are not claims that
no other project has similar methods or that the organizer will assign a
particular score.

## Official fit and presentation

Our interpretation is that Track 3.7—multi-step business workflows with rules,
approvals and an audit trail—is the primary problem statement, with Track 3.9
(Reward design and reward-hacking defences) as secondary. [Official tracks](https://hackathon.lexsi.ai/#tracks).
The fit for Track 3.9 is demonstrated by our analysis of evaluator and reward blind spots:
a simple non-empty draft predicate or generic apology passes automated mission checks
without delivering the required ledger information to the customer, revealing an
evaluator specification failure in workflow evaluation.

The kickoff judging description emphasizes write-up clarity, practical usefulness
and code/notebook quality. The repository therefore puts the concrete business
example, paired results and runnable review path before framework detail.
[Official kickoff slides](https://hackathon.lexsi.ai/kickoff-slides/).

The principal result is SFT with Transformers/PEFT. AgentTune's real GRPO pilot is
small and instrumental; AuditKIT is integrated with outcome reporting. Present
those roles explicitly. Do not frame this as a large AgentTune RL improvement or
claim that mentioning two libraries establishes submission acceptance.

## What still limits the case

- A synthetic, relatively small task distribution and a single seed.
- Public requests often identify malicious instructions explicitly.
- No observed adversarial harmful-effect reduction: both arms already had zero.
- Unequal ticket exposure between the arms.
- A nonempty-draft predicate that does not establish answer quality.
- Eighteen residual Spanish failures and two reversible intent errors.
- No separately prompt-optimized base baseline or commercial-API comparison.
- No measured production latency/ROI comparison or operating deployment.

These limits are already part of the results, not reasons to tune on the opened
test. A future version needs fresh evaluation.

## Release boundary

The core project (repository code, selected v3 LoRA adapter, benchmark dataset,
evaluation evidence and static replay Space) is already published publicly under
commit `ccd7c205` and associated Hugging Face collections. This update consolidates
follow-up diagnostics (P11–P18), evaluator limitation disclosures and documentation errata.
Deployment of follow-up updates, evidence expansion and final hackathon form submission
require separate owner authorization. No acceptance, shortlist or winning probability
is claimed by this document.
