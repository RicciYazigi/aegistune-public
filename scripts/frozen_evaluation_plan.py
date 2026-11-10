"""Validate frozen dataset identity and coverage before any model inference."""
import hashlib
import json


def validate_run_identity(metadata, adapter_sha256, inference):
    if metadata.get("adapter_sha256") not in (None, adapter_sha256):
        raise ValueError("Frozen adapter identity mismatch")
    declared = metadata.get("inference", {})
    for key, value in inference.items():
        if key in declared and declared[key] != value:
            raise ValueError(f"Frozen inference configuration mismatch: {key}")
    if metadata.get("reserved_test") and not set(inference) <= set(declared):
        raise ValueError("Reserved inference configuration incomplete")


def load_evaluation_plan(path, manifest_path, default_sha256, case_ids=None, limit=None):
    metadata = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else {}
    data = path.read_bytes()
    expected_sha = metadata.get("dataset_sha256", default_sha256)
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Dataset SHA-256 mismatch before inference")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    ids = [row.get("case_id", row.get("case_spec", {}).get("case_id")) for row in rows]
    if not ids or any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Dataset requires unique nonempty case IDs")
    if metadata.get("expected_case_count") is not None and len(ids) != metadata["expected_case_count"]:
        raise ValueError("Frozen case count mismatch")
    split = metadata.get("expected_split")
    if split and any(row.get("split", row.get("case_spec", {}).get("split")) != split for row in rows):
        raise ValueError("Frozen split mismatch")
    if metadata.get("reserved_test"):
        if split != "test" or metadata.get("expected_case_count") != 200:
            raise ValueError("Reserved test requires explicit test split and 200 cases")
        if case_ids is not None or limit is not None:
            raise ValueError("Reserved test forbids case filters and limits")
    declared = metadata.get("cases")
    if declared is not None:
        declared_ids = [row["case_id"] for row in declared]
        if len(set(declared_ids)) != len(declared_ids) or set(declared_ids) != set(ids):
            raise ValueError("Manifest case IDs mismatch")
        ids = declared_ids
    if case_ids:
        requested = [v.strip() for v in case_ids.split(",") if v.strip()] if isinstance(case_ids, str) else list(case_ids)
        if len(set(requested)) != len(requested) or not set(requested) <= set(ids):
            raise ValueError("Requested case IDs mismatch")
        ids = [cid for cid in ids if cid in set(requested)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("Case limit must be positive")
        ids = ids[:limit]
    return rows, ids, metadata
