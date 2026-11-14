# Release status — 13 November 2026

**Selected adapter: SFT v3.** Its frozen historical result is 182/200 mission checks versus 77/200 for base. These checks do not measure full semantic answer quality. [Follow-up evaluations](FOLLOWUP_EVALUATIONS.md) report the incorrect Challenge48 refund, generic inquiry drafts and unsuccessful repairs alongside the original result.

Public weights remain unchanged. The completed P17 opening-repair candidate was rejected after content and operational regressions. P14 and other experimental continuations are not selected releases. No later GRPO improvement, training loss or conditional probability is presented as a superior autonomous agent.

- [Public code](https://github.com/RicciYazigi/aegistune-public)
- [Selected adapter](https://huggingface.co/RicciYazigi/aegistune-tiny-aya-lora)
- [Model collection](https://huggingface.co/collections/RicciYazigi/aegistune-models-6af330852fdbcb4194a7ba2c)
- [Data and evidence](https://huggingface.co/collections/RicciYazigi/aegistune-data-and-evidence-6af3308596f1197ded458a3e)
- [TEST200 recorded replay](https://huggingface.co/spaces/RicciYazigi/aegistune-evidence)
- [P11 Spanish recorded inference](https://ricciyazigi-aegistune-evidence.static.hf.space/single-case-es.html)

The Space and saved pages are recorded evidence, not a live inference service. All business actions are simulated. Stronger independent validation and a live service remain separate work. Submission has not been sent; a supplementary recorded video walkthrough is included in this release selection.

Scientific evidence, public packaging and production readiness are distinct. This is an inspectable research prototype, not authorization for real payments.


## P23 follow-up (13 November 2026)

The frozen adapter was also compared with an original base prompt and a few-shot prompt selected on development data. All arms received the same initial ticket. On the 72-case internally authored bilingual exam:

| Episodes /72 | Base original | Dev-selected few-shot | SFT v3 |
|---|---:|---:|---:|
| Historical mission checks |47|43|50|
| Unauthorized-effect episodes |1|2|0|
| Parse-error episodes |11|5|0|
| Intent-deviation episodes |9|6|12|

The utility advantage is smaller than in TEST200; v3 improves effects and format but has more intent deviations. Strict factual/procedural scoring is 0/72, 0/72 and 4/72, with four v3 drafts pending human review. These are separate metrics and protocols, not a replacement 91% claim. [Protocol, results and limitations](P23_RESULTS.md).

