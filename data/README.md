# Synthetic billing benchmark

The benchmark contains660project-authored synthetic scenarios. It includes no
customer records or real banking credentials, and its tools perform no external
payments or messages. Documentation is English; original task/ticket text is
English or Spanish and must not be translated in place after evaluation.

| Split | Cases | Benign | Adversarial | English | Spanish |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Train |400|260|140|200|200|
| Development |60|39|21|30|30|
| Reserved test |200|100|100|100|100|

Six families cover billing inquiries, authorized refunds, supervisory proposals,
instruction injection, wrong destinations and excessive amounts. A reversible
proposal may be required even below a technical instant-payment limit.

Each scenario contains public task/policy/state data and separate host/oracle
metadata. Runtime prompt construction must not serialize hidden oracle answers
or family labels. The offline teacher can use privileged **train** labels;
evaluation uses the mission oracle only to score actions and trusted outcomes.

Entity prefixes and normalized surface text are separated. Shared root families,
sequential synthetic entities and explicit task cues remain limitations; this is
not an unseen-family or natural-customer deployment benchmark.

## Integrity

Keep the original `p4/*_cases.jsonl` and split manifests unchanged. The frozen test
SHA-256 is `15d70b505580a319c0a70cea30de39f8e2acf356b2b98d88431ae9f2ce2d8500`.
If generating a new dataset, use a new destination and identity. These test cases
have now been evaluated and must not be reused to claim unseen validation after
retuning.

The release includes original split data/manifests and recorded evaluation
artifacts. A license declaration for project-authored material does not relicense
third-party weights or libraries; consult [NOTICE](../NOTICE).
