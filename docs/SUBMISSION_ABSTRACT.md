# Submission abstract

**Submission form text (265 words):**

AegisTune studies a billing-agent failure mode: a technically permitted tool call can still perform the wrong business action. We adapt Cohere Labs Tiny Aya Global with supervised LoRA training and evaluate English and Spanish workflows using independent sandbox outcomes, recorded attempts and execution receipts.

On a frozen 200-case synthetic test, SFT v3 completed 182 missions (91%), versus 77 (38.5%) for the base model. Unauthorized-effect episodes fell from 17 to zero; all 17 base effects occurred in benign tasks, while adversarial effects were zero for both models. All 18 adapter mission failures were Spanish. A supplementary audit found two reversible intent deviations missed by the legacy proposal counter.

Further evaluation exposes limits of those headline results. On 48 additional, non-blind diagnostic cases with equal initial ticket delivery, v3 completed 40 missions versus 25 for the base, but executed one simulated refund of USD 98.63 instead of USD 28.63. Content review found all eight v3 inquiry drafts uninformative despite passing the nonempty-draft predicate. A 300-step grounded-teacher repair tied v3 at 56/60 development missions and failed to restore balance reporting; it was not promoted. A later opening-focused repair produced 4/12 grounded dev inquiry answers but regressed mission completion to 51/60 and was also rejected.

AgentTune supports genuine GRPO experiments and AuditKIT supports trajectory auditing; the measured improvement comes from SFT. Public adapters, datasets and an interactive recorded replay expose the evidence. Limitations include one seed, shared synthetic families, unequal exposure in the historical test and incomplete semantic scoring. The contribution is an inspectable account of workflow improvement and evaluator blind spots, not a claim of production payment safety.

---

**Form metadata:**
- Primary Track: Track 3.7 (Multi-step business workflows with rules, approvals and an audit trail)
- Secondary Track: Track 3.9 (Reward design and reward-hacking defences)
- Submission body word count: 265 words [Limit: 500 words]
