# Contributing

Start with the [architecture](docs/ARCHITECTURE.md), [contracts](docs/CONTRACTS.md)
and [evaluation protocol](docs/EXPERIMENT_PROTOCOL.md). Keep proposal quality,
gateway permission, executed effects and task completion separate.

## Portable CPU checks

Use a Python3.12+ environment and install `python -m pip install -e ".[dev]"`.
Run the six CPU suites listed in the README. They use synthetic fixtures and
need no model download, GPU, cloud account or Hugging Face credential.

Optional ML checks require the recorded ML stack and, where applicable, official
tokenizer/model access. A minimal-model test is not evidence of genuine Tiny Aya
training. Do not silently fetch gated assets or start paid resources in CI.

## Change requirements

- Preserve attempt-before-rejection recording and strict ID joins.
- Use immutable validated payloads and fresh environment state per episode.
- Keep hidden oracle/family labels out of runtime prompts and sensors.
- Derive utility from trusted effects, not an actor's claim or a tool name alone.
- Preserve the frozen dataset/model identities behind published results.
- Report malformed output and incomplete coverage rather than discarding them.
- Keep all fixtures fictional; never include real customer records or credentials.

For a pull request, explain the concrete behavior change, relevant verification,
and any effect on metric definitions or artifact compatibility. Do not regenerate
the frozen benchmark in place. A new recipe selected using old test outcomes
requires a new independent evaluation; preserve historical results unchanged.

## Reproduction

Re-evaluating the saved adapter differs from retraining it bit for bit. See
[REPRODUCTION.md](docs/REPRODUCTION.md) for this boundary and native-package
requirements. An issue should identify package versions and a minimal synthetic
case; omit credentials, private account names and operator paths.
