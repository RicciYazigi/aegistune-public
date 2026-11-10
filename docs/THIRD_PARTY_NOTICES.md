# Third-Party Notices & Licenses

AegisTune incorporates and interacts with several third-party libraries, foundation models, and evaluation frameworks. This document summarizes their provenance, licenses, and applicable terms.

---

## 1. Cohere Tiny Aya Global
- **Repository**: [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global)
- **Revision Used**: `af89d219b53ed9b13b8a4645f9c8028973510324`
- **Origin**: Cohere Labs
- **License**: Creative Commons Attribution-NonCommercial 4.0 International ([CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/))
- **Terms & Attribution**: Tiny Aya Global is used strictly for non-commercial research, benchmark evaluation, and hackathon demonstration purposes. Any downstream deployment or commercial use requires appropriate licensing from Cohere.

---

## 2. AgentTune & AuditKIT
- **Origin**: Hackathon Lexsi 2026 (Track 3: Agentic Fine-tuning)
- **Components**:
  - `agenttune-1.1.3-py3-none-any.whl` (GRPO trainer integration)
  - `AuditKIT` consequence metric contracts (`uapr_call`, `unauthorized_effect_pct`, `containment_pct`)
- **License**: AgentTune 1.1.3 installed metadata: Lexsi Labs Source Available License (LSAL) v1.1. AuditKIT 1.1.3 installed metadata: LSAL Version 1.2 (AuditKit). These are not Apache licenses; consult the complete terms before redistribution.
- **Usage**: AgentTune was used in the bounded genuine GRPO pilot. The principal SFT candidate uses Transformers/PEFT. AuditKIT provides native trajectory/consequence integration during evaluation; mission outcomes also use the independent sandbox oracle.

---

## 3. Hugging Face TRL (Transformer Reinforcement Learning) & PEFT
- **Components**: `trl`, `peft`
- **Origin**: Hugging Face Inc.
- **License**: Apache License, Version 2.0
- **Usage**: PEFT provides official LoRA parameter-efficient adaptation (`LoraConfig`, `get_peft_model`, `PeftModel`). TRL provides `GRPOTrainer` and policy loss calculations.

---

## 4. Fuse planning reference

No Fuse monitor was used in the reported base/SFT comparisons. Historical planning references do not establish an executed P6 experiment or a dependency of the selected candidate. No measured improvement is attributed to Fuse.

---

## 5. Core Machine Learning Libraries
- **PyTorch**: BSD 3-Clause License (Copyright (c) 2016-present Meta Platforms, Inc. and affiliates).
- **Hugging Face Transformers**: Apache License, Version 2.0 (Copyright (c) Hugging Face Inc.).
- **Datasets**: not required by the selected SFT/evaluation recipe; training uses a native torch Dataset. Historical GRPO dependencies are recorded separately.


## Verification boundary

The current upstream Tiny Aya model card was consulted on 9 November 2026 and lists CC-BY-NC-4.0 with acceptable-use conditions: https://huggingface.co/CohereLabs/tiny-aya-global . Access to the revision-specific raw license was not established through that public fetch. Do not treat a model-card label as a complete review of every addendum.

The repository packaging metadata declares Apache-2.0 for its own code. It does not relicense third-party weights, wheels or reference material. The selected-code LICENSE/NOTICE bundle is included. Final review of exact distribution permissions remains a publication step; nothing in this local draft authorizes publication.
