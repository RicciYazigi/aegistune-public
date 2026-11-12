# Reproducing the selected experiment

Local instructions, 9 November 2026. They do not launch cloud resources, publish artifacts or authorize spending. The selected model is SFT v3, adapter SHA-256 `269f6cc06c31506e651fc146c06ec675b6e21f012d119dd5a93e76a3c724a81a`, over base revision `af89d219b53ed9b13b8a4645f9c8028973510324`.

## Runtime and access

The measured runtime uses Linux/Python 3.12 with a Tesla T4, FP16, torch 2.15.0+cu132, Transformers 5.22.2, PEFT 0.21.2, tokenizers 0.23.3 and accelerate 1.15.0. `scripts/p4_space/requirements-eval.txt` records the resolved evaluation packages. Do not assume that Linux CUDA packages can be installed in a local Windows CPU environment.

Use the official gated-model access process. Provide an existing credential through the process environment; do not place credentials in arguments, source files or logs. These instructions do not accept new model terms or obtain access automatically.

The licensed framework wheels used in the frozen environment were:

| Wheel | SHA-256 |
| --- | --- |
| agenttune-1.1.3-py3-none-any.whl | 58c284da3769e6b301d260499738ab6d99a90e61984daf2eaeb544f6f6f8b752 |
| auditkit-1.1.3-py3-none-any.whl | a081aed3211bd2e286cab281535ba1fbdf0d663dc36fbaed1261fd982398fae6 |

Obtain the reviewed packages through the official distribution and respect their terms. They are not bundled here as permissively relicensed code. AgentTune is needed for the separate GRPO pilot; native AuditKIT is needed for the recorded evaluation integration. The current SFT trainer itself uses Transformers/PEFT.

## Evaluation first

**Generation-policy clarification (10 November):** the historical reserved report
records resolved terminators [6,261001,3], but its actor inherited generation
defaults rather than passing an explicit EOS argument. Resolution alone does not
establish those effective historical defaults. Preserve its frozen source and
results; do not silently relabel them. Current development source passes resolved
terminators explicitly. The four-arm development experiment documents and shares
that new policy across its arms; see [the comparison](PROMPT_COMPARISON_RESULTS.md).
Matching command-line limits alone does not prove identical historical stopping.

Re-evaluation of the selected adapter is different from reproducing training bit for bit. Verify its actual weights hash before running inference. Use the same dataset hash, model revision, greedy policy, context 4096, output 128, 4 turns and EOS [6,261001,3]. The paired runner shares one base instance, then attaches the verified adapter. A new isolated environment is created per episode.

```powershell
python scripts/run_p4_paired_dev60_evaluation.py --dev-cases data/p4/dev_cases.jsonl --adapter-path PATH_TO_SELECTED_ADAPTER --expected-adapter-sha256 269f6cc06c31506e651fc146c06ec675b6e21f012d119dd5a93e76a3c724a81a --device cuda --max-context-tokens 4096 --max-new-tokens 128 --max-turns 4 --generation-policy greedy --output outputs/reproduced_dev/summary.json --trajectory-output outputs/reproduced_dev/trajectory.jsonl
```

This example is development evaluation only. The reserved test has a separately frozen plan requiring all 200 cases and forbidding filters/limits. Do not reuse a seen test to choose a new model and then call the resulting score confirmatory.

The exported original plan is `artifacts/test200/evaluation_plan.json`. To reproduce the already completed paired test, use the same runner with `--dev-cases data/p4/test_cases.jsonl --evaluation-plan artifacts/test200/evaluation_plan.json` and the identical adapter and inference arguments above. The legacy `--dev-cases` argument name also accepts the reserved dataset; the plan enforces the split and full coverage. This is replication, not a new unseen test.

## Training recipe

V1 is a fresh LoRA trained for 300 steps on 400 train missions/1200 verified teacher actions. V3 continues **v1**, not the GRPO pilot or v2. The commands below require an already available suitable GPU; they do not acquire one.

```powershell
python scripts/run_p4_sft_warmup.py --train-cases data/p4/train_cases.jsonl --dev-cases data/p4/dev_cases.jsonl --output-dir outputs/reproduce_v1 --device cuda --max-steps 300 --learning-rate 0.00005 --max-training-seconds 7200 --paired-evaluation
python scripts/run_p4_sft_warmup.py --train-cases data/p4/train_cases.jsonl --dev-cases data/p4/dev_cases.jsonl --output-dir outputs/reproduce_v3 --device cuda --max-steps 100 --learning-rate 0.00001 --max-training-seconds 1800 --paired-evaluation --tool-name-weight 8 --initial-adapter PATH_TO_VERIFIED_ORIGINAL_V1 --expected-initial-sha256 79265e05944e667d3a141798c256ed4224cd73eec99a29580da10cb0016f4c07
```

Historical project sources are preserved in `reproducibility/sft_v1/` and `reproducibility/sft_v3/`, with SHA-256 manifests. Use the matching snapshot scripts and src on PYTHONPATH for the corresponding stage. The current teacher code includes the v3 staging-reason correction. Exact historical v1 training requires its frozen source rather than silently applying that correction retrospectively. A fresh v1 reproduction may have a different hash; verify that artifact's actual hash and use it for continuation instead of claiming it equals the original.

The recorded seed is 42. Training also incorporates fresh sandbox identifiers, and exact tokenized rows were not archived as a deterministic training dataset. Fresh training is therefore not guaranteed to reproduce the original weights bit for bit. Exact fresh reload was checked for the saved artifacts; this is not the same guarantee. No multi-seed training result is claimed.

## Auditing and review

```powershell
python scripts/audit_mission_intent.py --cases data/p4/dev_cases.jsonl --trajectory PATH_TO_PAIRED_TRAJECTORY --output outputs/intent_audit.json
```

Report the supplementary intent metric separately from the original UAPR counter, gateway denials and harmful committed effects. Preserve parsing failures and exact case coverage. [The recorded replay](demo/README.md) can be rebuilt only from completed receipt-verified artifacts and is explicitly not live inference.

Before public release, select technical material explicitly, finish license/notice review, sanitize metadata and review the write-up. Local files and hashes do not establish public availability, an uploaded model collection or a submitted entry.
