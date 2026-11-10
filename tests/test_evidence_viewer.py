"""Provenance and untrusted-output checks for the offline evidence viewer."""
import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_evidence_viewer import build, digest


@pytest.fixture
def bundle(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"case_id": "fixture", "language": "en", "template_family": "fixture",
        "user_task": "</script><script>untrusted()</script>",
        "mission_oracle": {"utility_predicate_spec": {}, "expected_effects": {"webhook_count": 0}}}) + "\n")
    trajectory = tmp_path / "trajectory.jsonl"
    episode = {"case_id": "fixture", "task_completed": True, "unauthorized_effect": False,
        "termination_reason": "fixture", "receipts": [],
        "attempts": [{"attempt_id": "a", "step_index": 0, "parsed_tool_name": "get_ticket",
                      "parsed_arguments": {}, "raw_model_output": "recorded output", "parse_error": None}],
        "decisions": [{"attempt_id": "a", "allowed": True, "reason_code": "OK"}]}
    trajectory.write_text("".join(json.dumps(dict(episode, arm=arm)) + "\n" for arm in ("base", "adapter")))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"evaluation_name": "fixture", "adapter_verification": {"safetensors_sha256": "selected"},
        "dataset": {"sha256": digest(cases)}, "base_summary": {"is_minimal_offline_test": True},
        "paired_metrics_comparison": {"missions_completed": {"base": "1/1", "adapter": "1/1"}}}))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"overall_status": "SUCCESS", "run_id": "fixture",
        "flow_result": {"run": {"results_verified": True, "remote_status": {
            "file_hashes_sha256": {p.name: digest(p) for p in (summary, trajectory)}}}}}))
    return cases, summary, trajectory, receipt, "selected", tmp_path / "viewer.html"


def test_fixture_guard_and_script_escaping(bundle):
    with pytest.raises(ValueError, match="CPU fixture"):
        build(*bundle)
    evidence = build(*bundle, allow_fixture=True)
    assert evidence["fixture"]
    html = bundle[-1].read_text(encoding="utf-8")
    assert "</script><script>untrusted()" not in html
    encoded = re.search(r'<script type="application/json" id="evidence">(.*?)</script>', html, re.S).group(1)
    assert json.loads(encoded)["cases"][0]["task"] == "</script><script>untrusted()</script>"


def test_changed_artifact_or_adapter_is_rejected(bundle):
    changed = list(bundle);changed[4] = "wrong"
    with pytest.raises(ValueError, match="adapter identity"):
        build(*changed, allow_fixture=True)
    bundle[2].write_text("tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        build(*bundle, allow_fixture=True)
