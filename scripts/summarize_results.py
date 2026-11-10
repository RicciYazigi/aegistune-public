"""Validate and summarize recorded results without inference or network access."""
import argparse
import json
from pathlib import Path


def fraction(value):
    numerator, denominator = map(int, value.split("/"))
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise ValueError("Invalid result denominator")
    return numerator, denominator


def summarize(report):
    metrics = report["paired_metrics"]
    cohorts = report["cohort_breakdown"]
    for arm in ("base", "adapter"):
        completed, total = fraction(metrics["missions_completed"][arm])
        effects, effect_total = fraction(metrics["unauthorized_effects_episodes"][arm])
        intent = report["intent_audit"][arm]
        if total != effect_total or total != intent["episodes"]:
            raise ValueError("Episode coverage mismatch")
        if sum(g["total"] for g in cohorts.values()) != total:
            raise ValueError("Cohort coverage mismatch")
        if sum(g[arm + "_completed"] for g in cohorts.values()) != completed:
            raise ValueError("Cohort completion mismatch")
        if sum(g[arm + "_unauthorized_effects"] for g in cohorts.values()) != effects:
            raise ValueError("Cohort effect mismatch")
        if intent["completed_missions"] != completed or intent["episodes_with_unauthorized_effect"] != effects:
            raise ValueError("Audit/outcome mismatch")
    rows = [
        ("Missions completed", metrics["missions_completed"]["base"], metrics["missions_completed"]["adapter"]),
        ("Unauthorized-effect episodes", metrics["unauthorized_effects_episodes"]["base"], metrics["unauthorized_effects_episodes"]["adapter"]),
        ("Original proposal counter", metrics["uapr_evaluable_proposals"]["base_fraction"], metrics["uapr_evaluable_proposals"]["adapter_fraction"]),
        ("Supplementary intent deviations", *[f"{report['intent_audit'][a]['intent_deviation_proposals']}/{report['intent_audit'][a]['evaluable']}" for a in ("base", "adapter")]),
    ]
    return "\n".join(["| Measure | Base | Selected adapter |", "| :--- | ---: | ---: |"] +
                     [f"| {name} | {base} | {adapter} |" for name, base, adapter in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(summarize(report))
    print("\nRecorded run:", report["run_id"])
    print("No inference performed. Proposal definitions and outcome metrics are distinct.")


if __name__ == "__main__":
    main()
