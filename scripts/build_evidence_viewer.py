"""Build a standalone offline review of hash-verified, recorded model actions.

No inference, network calls, credentials or real financial operations. The
reserved test may be rendered only from its completed evaluation receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from audit_mission_intent import audit, read_jsonl, unique_map


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(cases_path, summary_path, trajectory_path, receipt_path, adapter_sha256, output, allow_fixture=False):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    flow = receipt["flow_result"]
    if receipt["overall_status"] != "SUCCESS" or not flow["run"]["results_verified"]:
        raise ValueError("Viewer requires a completed verified evaluation")
    hashes = flow["run"]["remote_status"]["file_hashes_sha256"]
    for path in (summary_path, trajectory_path):
        if digest(path) != hashes.get(path.name):
            raise ValueError("Receipt/artifact SHA-256 mismatch")
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    report = source.get("paired_evaluation") or source
    if report["adapter_verification"]["safetensors_sha256"] != adapter_sha256:
        raise ValueError("Selected adapter identity mismatch")
    if digest(cases_path) != report["dataset"]["sha256"]:
        raise ValueError("Dataset identity mismatch")
    fixture = bool(report["base_summary"].get("is_minimal_offline_test"))
    if fixture and not allow_fixture:
        raise ValueError("CPU fixture is not genuine model evidence")
    cases = read_jsonl(cases_path)
    traces = read_jsonl(trajectory_path)
    supplement = audit(cases, traces)  # Strict case and attempt/decision joins.
    by_arm = {arm: unique_map([e for e in traces if e["arm"] == arm], "case_id") for arm in ("base", "adapter")}
    if set(supplement["arms"]) != {"base", "adapter"}:
        raise ValueError("Viewer requires exactly two paired arms")
    projected = []
    for case in cases:
        arms = {}
        for arm in ("base", "adapter"):
            episode = by_arm[arm][case["case_id"]]
            decisions = unique_map(episode["decisions"], "attempt_id")
            receipts = unique_map(episode["receipts"], "attempt_id")
            actions = []
            for attempt in episode["attempts"]:
                decision = decisions[attempt["attempt_id"]]
                actions.append({"step": attempt["step_index"] + 1, "tool": attempt["parsed_tool_name"],
                    "arguments": attempt["parsed_arguments"], "raw": attempt["raw_model_output"],
                    "parse_error": attempt["parse_error"], "allowed": decision["allowed"],
                    "reason": decision["reason_code"], "receipt": receipts.get(attempt["attempt_id"], {}).get("status")})
            arms[arm] = {"completed": episode["task_completed"], "harm": episode["unauthorized_effect"],
                         "termination": episode["termination_reason"], "actions": actions}
        projected.append({"id": case["case_id"], "language": case["language"], "family": case["template_family"],
                          "task": case["user_task"], "arms": arms})
    for arm in ("base", "adapter"):
        expected = f"{sum(row['arms'][arm]['completed'] for row in projected)}/{len(cases)}"
        if report["paired_metrics_comparison"]["missions_completed"][arm] != expected:
            raise ValueError("Summary/trajectory outcome mismatch")
    evidence = {"evaluation": report["evaluation_name"], "reserved_test": report["dataset"].get("reserved_test", False),
        "fixture": fixture, "run_id": receipt["run_id"], "adapter_sha256": adapter_sha256,
        "dataset_sha256": digest(cases_path), "summary_sha256": digest(summary_path),
        "trajectory_sha256": digest(trajectory_path), "metrics": report["paired_metrics_comparison"],
        "intent_audit": {arm: {k: v for k, v in v.items() if k != "details"} for arm, v in supplement["arms"].items()},
        "cases": projected}
    encoded = json.dumps(evidence, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    encoded = re.sub(r"hf_[A-Za-z0-9]{10,}", "<REDACTED_HF_TOKEN>", encoded)
    template = Path(__file__).resolve().parents[1] / "docs/demo/viewer_template.html"
    html = template.read_text(encoding="utf-8").replace("__EVIDENCE_JSON__", encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "summary", "trajectory", "receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    result = build(args.cases, args.summary, args.trajectory, args.receipt, args.adapter_sha256,
                   args.output, args.allow_fixture)
    print(json.dumps({"cases": len(result["cases"]), "run_id": result["run_id"],
                      "reserved_test": result["reserved_test"], "fixture": result["fixture"]}))


if __name__ == "__main__":
    main()
