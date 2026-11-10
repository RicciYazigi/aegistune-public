"""Compare P4 Baseline vs Adapted Evaluation Runs.

Loads two evaluation summary artifacts (base vs adapted) produced by the standard
P4 baseline evaluation harness (run_p4_baseline_evaluation.py) and produces:
1. Formatted GitHub Markdown comparison table.
2. Structured JSON delta metrics with explicit denominators.
3. Transparent run classification labeling (genuine weights vs test controls).
4. Strict comparability validation (rejecting protocol mismatches and training artifacts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_run_data(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load evaluation report. Returns (summary_dict, full_report_dict)."""
    if not path.is_file():
        raise FileNotFoundError(f"Run summary artifact not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Explicitly reject training summary artifacts
    if (
        "training_steps_executed" in raw
        or "training_episodes" in raw
        or "pipeline_classification" in raw
        or "steps_completed" in raw
    ):
        raise ValueError(
            f"Artifact '{path.name}' is a training summary ('training_summary.json'), not an evaluation report. "
            "Comparisons require evaluation reports produced by run_p4_baseline_evaluation.py (or dev_evaluation_report.json)."
        )

    if "summary" in raw and isinstance(raw["summary"], dict):
        return raw["summary"], raw
    elif "overall_metrics" in raw:
        return raw, raw
    else:
        raise ValueError(
            f"Artifact '{path.name}' does not contain a valid evaluation summary schema ('summary' or 'overall_metrics')."
        )


def extract_key_metrics(summary: dict[str, Any], full_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract and validate mandatory evaluation metrics without substituting favorable defaults."""
    # Mandatory sections
    for sec in ("overall_metrics", "benign_cohort", "adversarial_cohort"):
        if sec not in summary or not isinstance(summary[sec], dict):
            raise ValueError(f"Invalid evaluation report: missing mandatory section '{sec}'")

    overall = summary["overall_metrics"]
    benign = summary["benign_cohort"]
    adversarial = summary["adversarial_cohort"]
    classification = summary.get("actor_classification", {})

    # Mandatory overall metrics
    mandatory_overall = (
        "overall_utility_mean",
        "unauthorized_effects_pct",
        "format_valid_pct",
        "total_tool_attempts",
        "parse_error_count",
        "valid_format_tool_attempts",
        "evaluable_proposals",
        "unauthorized_proposals",
    )
    for m in mandatory_overall:
        if m not in overall:
            raise ValueError(f"Invalid evaluation report: missing mandatory metric '{m}' in overall_metrics")

    # Mandatory benign metrics
    for m in ("total", "utility_mean", "task_completed_count", "task_completed_pct"):
        if m not in benign:
            raise ValueError(f"Invalid evaluation report: missing mandatory metric '{m}' in benign_cohort")

    # Mandatory adversarial metrics
    for m in ("total", "without_unauthorized_effect_count", "contained_count", "utility_mean"):
        if m not in adversarial:
            raise ValueError(f"Invalid evaluation report: missing mandatory metric '{m}' in adversarial_cohort")

    # Top-level mandatory fields
    if "total_cases_evaluated" not in summary:
        raise ValueError("Invalid evaluation report: missing 'total_cases_evaluated'")
    if "dataset_sha256" not in summary:
        raise ValueError("Invalid evaluation report: missing 'dataset_sha256'")
    if "evaluated_case_ids" not in summary:
        raise ValueError("Invalid evaluation report: missing 'evaluated_case_ids'")

    total_cases = summary["total_cases_evaluated"]
    utility = float(overall["overall_utility_mean"])
    unauthorized_pct = float(overall["unauthorized_effects_pct"])
    format_valid_pct = float(overall["format_valid_pct"])
    parse_errors = int(overall["parse_error_count"])
    total_attempts = int(overall["total_tool_attempts"])
    evaluable_proposals = int(overall["evaluable_proposals"])

    # uapr_call: None if no evaluable proposals exist
    uapr_call = overall.get("uapr_call")
    if uapr_call is not None:
        uapr_call = float(uapr_call)

    # Benign metrics
    benign_count = int(benign["total"])
    benign_utility = float(benign["utility_mean"])
    benign_completed = int(benign["task_completed_count"])
    benign_completed_pct = float(benign["task_completed_pct"])

    # Adversarial metrics
    adv_count = int(adversarial["total"])
    adv_utility = float(adversarial["utility_mean"])
    adv_contained = int(adversarial["contained_count"])
    containment_pct = adversarial.get("containment_pct")
    if containment_pct is not None:
        containment_pct = float(containment_pct)

    # Total damage calculation (derived from overall_metrics or case_results if present, or null if cannot be calculated)
    total_damage_usd: float | None = None
    if "unauthorized_refund_damage_usd" in overall and overall["unauthorized_refund_damage_usd"] is not None:
        total_damage_usd = round(float(overall["unauthorized_refund_damage_usd"]), 2)
    elif full_report and "case_results" in full_report:
        case_damages = [
            c.get("unauthorized_refund_damage_minor")
            for c in full_report.get("case_results", [])
        ]
        if case_damages and all(d is not None for d in case_damages):
            total_damage_usd = round(sum(case_damages) / 100.0, 2)
    elif "unauthorized_effects_count" in overall and overall["unauthorized_effects_count"] == 0:
        total_damage_usd = 0.0

    # Actor/Model classification
    is_genuine = bool(classification.get("is_genuine_tiny_aya_weights", False))
    is_random = bool(classification.get("is_random_minimal_test_stub", False))
    actor_mode = str(summary.get("actor_mode", "unknown"))

    if is_genuine:
        label = "Genuine Tiny Aya Global Weights"
    elif is_random:
        label = "Synthetic Random Minimal Stub (CPU Control)"
    elif actor_mode == "public_rule_agent":
        label = "Deterministic Public Rule Control"
    elif actor_mode == "rule_oracle":
        label = "Privileged Rule Oracle Ceiling"
    else:
        label = f"Control / {actor_mode}"

    return {
        "label": label,
        "is_genuine": is_genuine,
        "is_random": is_random,
        "actor_mode": actor_mode,
        "model_id": summary.get("model_id"),
        "model_revision": summary.get("model_revision"),
        "adapter_path": summary.get("adapter_path"),
        "dataset_file": summary.get("dataset_file"),
        "dataset_sha256": summary["dataset_sha256"],
        "evaluated_case_ids": list(summary["evaluated_case_ids"]),
        "max_context_tokens": summary.get("max_context_tokens"),
        "max_new_tokens": summary.get("max_new_tokens"),
        "max_turns": summary.get("max_turns"),
        "parser_version": summary.get("parser_version"),
        "generation_policy": summary.get("generation_policy"),
        "is_minimal_offline_test": summary.get("is_minimal_offline_test"),
        "total_cases": total_cases,
        "utility": utility,
        "unauthorized_pct": unauthorized_pct,
        "total_damage_usd": total_damage_usd,
        "uapr_call": uapr_call,
        "format_valid_pct": format_valid_pct,
        "parse_errors": parse_errors,
        "total_attempts": total_attempts,
        "evaluable_proposals": evaluable_proposals,
        "benign_count": benign_count,
        "benign_utility": benign_utility,
        "benign_completed": benign_completed,
        "benign_completed_pct": benign_completed_pct,
        "adv_count": adv_count,
        "adv_contained": adv_contained,
        "adv_utility": adv_utility,
        "containment_pct": containment_pct,
    }


def validate_comparability(base_m: dict[str, Any], adapted_m: dict[str, Any]) -> None:
    """Validate that two runs evaluated identical benchmark distributions under identical protocols."""
    # 1. Dataset SHA-256 match
    if base_m["dataset_sha256"] != adapted_m["dataset_sha256"]:
        raise ValueError(
            f"Dataset SHA-256 mismatch: Base evaluated '{base_m['dataset_sha256'][:16]}...', "
            f"Adapted evaluated '{adapted_m['dataset_sha256'][:16]}...'. Comparisons must use identical test sets."
        )

    # 2. Case IDs match
    if base_m["evaluated_case_ids"] != adapted_m["evaluated_case_ids"]:
        raise ValueError(
            f"Evaluated case IDs mismatch: Base ({len(base_m['evaluated_case_ids'])}) vs "
            f"Adapted ({len(adapted_m['evaluated_case_ids'])})."
        )

    # 3. Total count match
    if base_m["total_cases"] != adapted_m["total_cases"]:
        raise ValueError(
            f"Total cases evaluated mismatch: Base={base_m['total_cases']}, Adapted={adapted_m['total_cases']}."
        )

    # 4. Context budget and token generation limit match
    if (
        base_m["max_context_tokens"] is not None
        and adapted_m["max_context_tokens"] is not None
        and base_m["max_context_tokens"] != adapted_m["max_context_tokens"]
    ):
        raise ValueError(
            f"Evaluation token budget mismatch: Base max_context_tokens={base_m['max_context_tokens']}, "
            f"Adapted max_context_tokens={adapted_m['max_context_tokens']}."
        )
    if (
        base_m["max_new_tokens"] is not None
        and adapted_m["max_new_tokens"] is not None
        and base_m["max_new_tokens"] != adapted_m["max_new_tokens"]
    ):
        raise ValueError(
            f"Evaluation generation limit mismatch: Base max_new_tokens={base_m['max_new_tokens']}, "
            f"Adapted max_new_tokens={adapted_m['max_new_tokens']}."
        )

    # 5. Base model architecture match for model evaluations
    if base_m["is_random"] != adapted_m["is_random"]:
        raise ValueError(
            f"Evaluation actor type mismatch: Base has is_random={base_m['is_random']}, "
            f"Adapted has is_random={adapted_m['is_random']}. Cannot compare random test stub with genuine weights."
        )

    if (
        base_m["model_id"]
        and adapted_m["model_id"]
        and (base_m["model_id"] != adapted_m["model_id"] or base_m["model_revision"] != adapted_m["model_revision"])
    ):
        raise ValueError(
            f"Base model mismatch: Base has {base_m['model_id']}@{base_m['model_revision']}, "
            f"Adapted has {adapted_m['model_id']}@{adapted_m['model_revision']}."
        )

    # 6. Evaluation protocol parameters match (mandatory fields, strict equality)
    for field in ("max_turns", "parser_version", "generation_policy"):
        base_val = base_m.get(field)
        adapted_val = adapted_m.get(field)
        if base_val is None or adapted_val is None:
            raise ValueError(
                f"Protocol validation error: missing mandatory field '{field}' in evaluation report. "
                f"Base={base_val!r}, Adapted={adapted_val!r}. "
                "Comparisons require explicit max_turns, parser_version, and generation_policy."
            )
        if base_val != adapted_val:
            raise ValueError(
                f"Evaluation protocol mismatch for '{field}': Base={base_val!r} vs Adapted={adapted_val!r}."
            )


def compute_comparison(base_m: dict[str, Any], adapted_m: dict[str, Any]) -> dict[str, Any]:
    """Compute mathematical deltas between base and adapted runs."""
    validate_comparability(base_m, adapted_m)

    def safe_delta(val_a: float | None, val_b: float | None) -> float | None:
        if val_a is None or val_b is None:
            return None
        return round(val_a - val_b, 4)

    delta_utility = safe_delta(adapted_m["utility"], base_m["utility"])
    delta_unauth = safe_delta(adapted_m["unauthorized_pct"], base_m["unauthorized_pct"])
    delta_damage = safe_delta(adapted_m["total_damage_usd"], base_m["total_damage_usd"])
    delta_benign_util = safe_delta(adapted_m["benign_utility"], base_m["benign_utility"])
    delta_format = safe_delta(adapted_m["format_valid_pct"], base_m["format_valid_pct"])

    delta_uapr_call = safe_delta(adapted_m["uapr_call"], base_m["uapr_call"])
    delta_containment = safe_delta(adapted_m["containment_pct"], base_m["containment_pct"])

    return {
        "base": base_m,
        "adapted": adapted_m,
        "deltas": {
            "delta_utility": delta_utility,
            "delta_unauthorized_pct": delta_unauth,
            "delta_damage_usd": delta_damage,
            "delta_benign_utility": delta_benign_util,
            "delta_format_valid_pct": delta_format,
            "delta_uapr_call": delta_uapr_call,
            "delta_containment_pct": delta_containment,
        },
    }


def generate_markdown_table(comp: dict[str, Any]) -> str:
    b = comp["base"]
    a = comp["adapted"]
    d = comp["deltas"]

    def fmt_val(val: Any, is_pct: bool = False, is_usd: bool = False, note_on_none: str = "N/A") -> str:
        if val is None:
            return note_on_none
        if is_usd:
            return f"${val:.2f}"
        if is_pct:
            return f"{val:.1f}%"
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)

    def fmt_delta(delta: float | None, is_pct: bool = False, is_usd: bool = False, reverse_good: bool = False) -> str:
        if delta is None:
            return "—"
        sign = "+" if delta > 0 else ""
        if is_usd:
            s = f"{sign}${delta:.2f}"
        elif is_pct:
            s = f"{sign}{delta:.1f}%"
        else:
            s = f"{sign}{delta:.3f}"

        if delta == 0:
            return s
        is_good = (delta > 0 and not reverse_good) or (delta < 0 and reverse_good)
        indicator = "✅" if is_good else "⚠️"
        return f"{s} {indicator}"

    uapr_note = "N/A (0 evaluable proposals)" if b["evaluable_proposals"] == 0 else "N/A"
    containment_note = "N/A (0 evaluable proposals)" if b["evaluable_proposals"] == 0 else "N/A"

    lines = [
        "### Comparación Experimental: Modelo Base vs. Adaptado",
        "",
        f"- **Modelo Base:** {b['label']} (`{b['model_id'] or b['actor_mode']}`)",
        f"- **Modelo Adaptado:** {a['label']} (`{a['adapter_path'] or 'LoRA adapter'}`)",
        f"- **Dataset Evaluado:** `{b['dataset_file']}` (SHA-256: `{b['dataset_sha256'][:16]}...`)",
        f"- **Casos Evaluados:** {b['total_cases']} casos (Benignos: {b['benign_count']}, Adversariales: {b['adv_count']})",
        "",
        "| Métrica | Base | Adaptado | Delta (Adaptado - Base) |",
        "|---|---|---|---|",
        f"| **Utilidad de Tarea (Overall)** | {fmt_val(b['utility'])} | {fmt_val(a['utility'])} | {fmt_delta(d['delta_utility'])} |",
        f"| **Utilidad Cohorte Benigna** | {fmt_val(b['benign_utility'])} | {fmt_val(a['benign_utility'])} | {fmt_delta(d['delta_benign_utility'])} |",
        f"| **Contención Cohorte Adversarial** | {fmt_val(b['containment_pct'], is_pct=True, note_on_none=containment_note)} | {fmt_val(a['containment_pct'], is_pct=True, note_on_none=containment_note)} | {fmt_delta(d['delta_containment_pct'], is_pct=True)} |",
        f"| **Efectos No Autorizados (%)** | {fmt_val(b['unauthorized_pct'], is_pct=True)} | {fmt_val(a['unauthorized_pct'], is_pct=True)} | {fmt_delta(d['delta_unauthorized_pct'], is_pct=True, reverse_good=True)} |",
        f"| **Daño Residual Monetario ($ USD)** | {fmt_val(b['total_damage_usd'], is_usd=True)} | {fmt_val(a['total_damage_usd'], is_usd=True)} | {fmt_delta(d['delta_damage_usd'], is_usd=True, reverse_good=True)} |",
        f"| **Tasa de Propuestas No Autorizadas (UAPR Call)** | {fmt_val(b['uapr_call'], note_on_none=uapr_note)} | {fmt_val(a['uapr_call'], note_on_none=uapr_note)} | {fmt_delta(d['delta_uapr_call'], reverse_good=True)} |",
        f"| **Validez de Formato (%)** | {fmt_val(b['format_valid_pct'], is_pct=True)} | {fmt_val(a['format_valid_pct'], is_pct=True)} | {fmt_delta(d['delta_format_valid_pct'])} |",
        f"| **Intentos de Herramienta Totales** | {b['total_attempts']} | {a['total_attempts']} | — |",
        f"| **Errores de Parseo (JSON/Fences)** | {b['parse_errors']} | {a['parse_errors']} | — |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare P4 evaluation runs (Base vs. Adapted).")
    parser.add_argument("--base", type=Path, required=True, help="Path to base evaluation report JSON")
    parser.add_argument("--adapted", type=Path, required=True, help="Path to adapted evaluation report JSON")
    parser.add_argument("--output-markdown", type=Path, default=None, help="Save markdown comparison table")
    parser.add_argument("--output-json", type=Path, default=None, help="Save structured JSON diff report")

    args = parser.parse_args(argv)

    base_summary, base_full = load_run_data(args.base)
    adapted_summary, adapted_full = load_run_data(args.adapted)

    base_m = extract_key_metrics(base_summary, base_full)
    adapted_m = extract_key_metrics(adapted_summary, adapted_full)

    comp = compute_comparison(base_m, adapted_m)
    md_table = generate_markdown_table(comp)

    print("\n" + md_table)

    if args.output_markdown:
        args.output_markdown.write_text(md_table, encoding="utf-8")
        print(f"Markdown report written to {args.output_markdown}")

    if args.output_json:
        args.output_json.write_text(json.dumps(comp, indent=2), encoding="utf-8")
        print(f"JSON diff report written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
