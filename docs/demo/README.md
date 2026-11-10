# Offline evidence replay

Open `test200.html` for the completed reserved evaluation or `dev60.html` for development, directly in a browser. It is self-contained: no server,
credentials, model download or network requests are required. It shows recorded
outputs from the verified v3 development and reserved-test runs, each clearly labeled. It is **not live inference** and
does not execute tools. No real payments or webhooks occur.

The page pairs public user requests, model actions, gateway decisions and
execution receipt status. Mission outcomes and harmful effects remain separate
from technical gateway permission. Raw output is rendered as text. The export
does not include host-observation text or prove the semantic quality of a draft.

Use the filters to inspect improvements and failures, including four development staging misses and 18 reserved-test misses, all in Spanish. A favorable sample is an illustration, not an aggregate
benchmark result. Provenance includes the run ID and input/adapter hashes.

Rebuild from a completed downloaded evaluation:

```powershell
python scripts/build_evidence_viewer.py --cases PATH_TO_CASES --summary PATH_TO_SUMMARY --trajectory PATH_TO_TRAJECTORY --receipt PATH_TO_SUCCESS_RECEIPT --adapter-sha256 EXPECTED_SELECTED_SHA256 --output docs/demo/replay.html
```

The summary can be a paired report or a training summary containing a paired
report. The builder checks the summary and trajectory hashes against the receipt,
the selected adapter identity, dataset identity and exact case/trace joins.
Minimal-model fixtures are rejected unless explicitly allowed and clearly marked.
The reserved-test replay was generated after the authorized evaluation
finished and all 400 episodes/five artifacts were verified. Preparation of a local file does not constitute publication or submission.
