AegisTune studies a practical failure mode in billing agents: a technically permitted tool call can still perform the wrong business action. We adapt Cohere Labs Tiny Aya Global with supervised LoRA training and evaluate authorized work through independent sandbox outcomes rather than model claims or JSON validity alone.

The synthetic environment covers English and Spanish support inquiries, authorized refunds, supervisory proposals and instruction attacks. Host-owned permissions, recorded attempts, execution receipts and an independent mission oracle remain separate. Training uses 400 cases, development uses 60, and the selected adapter was frozen before a paired 200-case reserved evaluation.

SFT v3 completed 182/200 missions (91%), compared with 77/200 (38.5%) for the base. Benign completion improved from 4/100 to 82/100. Unauthorized-effect episodes fell from 17/200 to zero; all 17 base effects occurred in benign tasks, while adversarial effects were already zero in both arms. A supplementary audit identified two remaining reversible intent deviations despite a zero legacy proposal counter. All 18 remaining mission failures were Spanish.

AgentTune supports genuine GRPO rollout experiments, while AuditKIT supports trajectory and consequence auditing. The main measured improvement is SFT; subsequent GRPO attempts did not reach a completed comparative evaluation. Public code and an interactive recorded replay expose both successes and failures.

Limitations include one seed, shared synthetic template families, unequal ticket exposure and a draft predicate that checks nonempty text rather than semantic quality. AegisTune demonstrates auditable workflow improvement within this benchmark, not production payment safety or universal injection resistance.
