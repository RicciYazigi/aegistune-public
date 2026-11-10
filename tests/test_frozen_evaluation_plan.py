"""Reserved-plan tests use synthetic fixtures, never the actual reserved cases."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from frozen_evaluation_plan import load_evaluation_plan, validate_run_identity


@pytest.fixture
def frozen(tmp_path):
    cases = tmp_path / "fixture.jsonl"
    cases.write_text("".join(json.dumps({"case_id": str(i), "split": "test"}) + "\n" for i in range(200)))
    plan = tmp_path / "plan.json"
    meta = {"dataset_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
            "expected_case_count": 200, "expected_split": "test", "reserved_test": True}
    plan.write_text(json.dumps(meta))
    return cases, plan, meta


def test_reserved_requires_full_200_cases(frozen):
    cases, plan, _ = frozen
    rows, ids, meta = load_evaluation_plan(cases, plan, None)
    assert len(rows) == len(ids) == 200 and meta["reserved_test"]
    for options in ({"limit": 1}, {"case_ids": "0"}, {"case_ids": []}):
        with pytest.raises(ValueError, match="forbids"):
            load_evaluation_plan(cases, plan, None, **options)


def test_tampering_and_bad_manifest_fail_before_inference(frozen):
    cases, plan, meta = frozen
    meta["dataset_sha256"] = "0" * 64;plan.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="SHA-256"):
        load_evaluation_plan(cases, plan, None)
    with pytest.raises(FileNotFoundError):
        load_evaluation_plan(cases, plan.with_name("missing.json"), None)


@pytest.mark.parametrize("change,message", [({"expected_case_count": 201}, "count"),
                                           ({"expected_split": "dev"}, "split"),
                                           ({"cases": [{"case_id": "0"}]}, "IDs mismatch")])
def test_frozen_coverage_constraints(frozen, change, message):
    cases, plan, meta = frozen
    meta.update(change);plan.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=message):
        load_evaluation_plan(cases, plan, None)


def test_frozen_adapter_and_inference_identity():
    meta = {"reserved_test": True, "adapter_sha256": "selected", "inference": {"max_turns": 4}}
    validate_run_identity(meta, "selected", {"max_turns": 4})
    with pytest.raises(ValueError, match="adapter identity"):
        validate_run_identity(meta, "other", {"max_turns": 4})
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_run_identity(meta, "selected", {"max_turns": 2})
    with pytest.raises(ValueError, match="incomplete"):
        validate_run_identity(meta, "selected", {"max_turns": 4, "max_new_tokens": 128})
