"""P23 Two-Stage Bilingual Evaluation Runner (Dev60 Prompt Selection + Exam72).

Stage 1 (Dev60, --stage dev):
  Evaluates 3 base prompt arms (`original`, `operational_v1`, `schema_fewshot_v1`)
  under `equal_ticket_delivery_v1`. Exports `dev_provisional_selection.json` and
  `dev_adjudication_queue.json` for mandatory human adjudication. Does NOT
  produce a final `dev_selection.json`.

Finalizer (--stage finalize-dev --adjudication-file ...):
  Validates complete human adjudication (no pending items, no duplicates, valid
  reviewer/date/reason, sealed input hashes), recomputes all 3 dev arms from raw
  episode traces, runs `select_dev_winner`, and writes sealed `dev_selection.json`.

Stage 2 (Exam72, --stage exam):
  Verifies all sealed hashes (model/revision/adapter, dev60, exam72 + manifest,
  prompts, rubric, runner, adjudication, dev traces) and recomputes the Stage 1
  winner BEFORE loading GPU weights. Evaluates `base_original`, `base_<winner>`
  (skipping duplicate inference if `original` won), and `v3_original`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
_SRC = _SCRIPTS.parent / "src"
for _p in (_SCRIPTS, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from audit_mission_intent import audit as audit_mission_intent
from equal_exposure import ticket_prelude
from factual_rubric import score_episode, summarize_arm
from prompt_variants import VARIANTS, instructions, policy_record

DEV_PROMPT_VARIANTS = ("original", "operational_v1", "schema_fewshot_v1")
VARIANT_TO_ARM_NAME = {
    "original": "base_original",
    "operational_v1": "base_operational",
    "schema_fewshot_v1": "base_schema_fewshot",
}
REQUIRED_EOS_TOKEN_IDS = [6, 261001, 3]
DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"
DEFAULT_MODEL_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
EXPECTED_V3_ADAPTER_SHA256 = "269f6cc06c31506e651fc146c06ec675b6e21f012d119dd5a93e76a3c724a81a"
EXPECTED_DEV60_CASES_SHA256 = "dfa8aebd6ef98e8002ac1fb7e8263ea2227e3cc515844987fa221f1d85d9b4ba"
EXPECTED_EXAM72_CASES_SHA256 = "40d814766433975505e8d17ee4764673ba6019f17b0371ff1302d883ca7330e3"
STAGE1_EXECUTED_RUNNER_SHA256 = "4bc3c725417b8d0114d33309bf9b206c44b56791a5c4c8588c29bc6ba3fce949"
MAX_CONTEXT_TOKENS = 4096
MAX_NEW_TOKENS = 128
MAX_TURNS = 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_case_coverage(cases: list[dict[str, Any]], episodes: list[dict[str, Any]], arm_name: str) -> None:
    expected = {c["case_id"] for c in cases}
    seen: list[str] = [ep["case_id"] for ep in episodes]
    if len(seen) != len(set(seen)):
        raise ValueError(f"Duplicate case_id in arm {arm_name}")
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        extra = sorted(set(seen) - expected)
        raise ValueError(f"Coverage mismatch in arm {arm_name}: missing={missing[:5]}, extra={extra[:5]}")


def verify_distinct_arm_uuids(arm_episodes: dict[str, list[dict[str, Any]]]) -> None:
    """Ensure distinct episode_id across arms (and non-default generation_id when non-deterministic)."""
    seen_ep: dict[str, str] = {}
    seen_gen: dict[str, str] = {}
    for arm_name, eps in arm_episodes.items():
        for ep in eps:
            eid = ep.get("episode_id")
            gid = ep.get("generation_id")
            cid = ep.get("case_id")
            if eid is not None:
                prev = seen_ep.get(eid)
                if prev is not None and prev != arm_name:
                    raise ValueError(f"Reused episode_id {eid!r} across arms {prev} and {arm_name}")
                seen_ep[eid] = arm_name
            if gid is not None and gid != f"gen-{cid}-g00":
                prev = seen_gen.get(gid)
                if prev is not None and prev != arm_name:
                    raise ValueError(f"Reused generation_id {gid!r} across arms {prev} and {arm_name}")
                seen_gen[gid] = arm_name


def score_arm_bundle(
    arm_name: str,
    prompt_variant: str,
    cases: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    adjudications_by_case: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    verify_case_coverage(cases, episodes, arm_name)
    tagged = [{**ep, "arm": arm_name} for ep in episodes]
    adj_map = adjudications_by_case or {}
    factual = summarize_arm(cases, tagged, adjudications_by_case=adj_map)
    intent_audit = audit_mission_intent(cases, tagged)
    intent_arm = intent_audit["arms"][arm_name]

    unauthorized_effects = sum(1 for ep in tagged if ep.get("unauthorized_effect"))
    historical_completed = sum(1 for ep in tagged if ep.get("task_completed"))
    parse_error_episodes = sum(
        1 for ep in tagged if any(a.get("parse_error") is not None for a in ep.get("attempts", []))
    )
    prompt_chars = len(instructions(prompt_variant, "en")) + len(instructions(prompt_variant, "es"))

    cases_by_id = {c["case_id"]: c for c in cases}
    ep_by_id = {ep["case_id"]: ep for ep in tagged}
    eval_by_id = {
        cid: score_episode(cases_by_id[cid], ep_by_id[cid], adjudication=adj_map.get(cid))
        for cid in cases_by_id
    }
    by_language: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    for cid, c in cases_by_id.items():
        lang = c["language"]
        fam = c["template_family"]
        passed = 1 if eval_by_id[cid]["factual_passed"] else 0
        for bucket, key in ((by_language, lang), (by_family, fam)):
            entry = bucket.setdefault(key, {"total": 0, "factual_passed": 0})
            entry["total"] += 1
            entry["factual_passed"] += passed

    pairs: dict[str, dict[str, bool]] = {}
    for cid, c in cases_by_id.items():
        pid = c.get("pair_id")
        if pid:
            pairs.setdefault(pid, {})[c["language"]] = bool(eval_by_id[cid]["factual_passed"])
    pair_sensitivity = None
    if pairs:
        counts = {"both_passed": 0, "en_only": 0, "es_only": 0, "neither": 0}
        for langs in pairs.values():
            en_ok = langs.get("en", False)
            es_ok = langs.get("es", False)
            if en_ok and es_ok:
                counts["both_passed"] += 1
            elif en_ok:
                counts["en_only"] += 1
            elif es_ok:
                counts["es_only"] += 1
            else:
                counts["neither"] += 1
        pair_sensitivity = {"total_pairs": len(pairs), **counts}

    return {
        "arm": arm_name,
        "prompt_variant": prompt_variant,
        "total_cases": len(cases),
        "screening_passed": factual["screening_passed"],
        "pending_semantic_review_episodes": factual["pending_semantic_review_episodes"],
        "adjudicated_approved_episodes": factual["adjudicated_approved_episodes"],
        "adjudicated_rejected_episodes": factual["adjudicated_rejected_episodes"],
        "factual_passed": factual["factual_passed"],
        "factual_pass_rate": round(factual["factual_passed"] / len(cases), 6) if cases else 0.0,
        "no_unauthorized_effect": factual["no_unauthorized_effect_episodes"],
        "historical_completed": historical_completed,
        "unauthorized_effect_episodes": unauthorized_effects,
        "intent_deviation_episodes": intent_arm["episodes_with_intent_deviation"],
        "parse_error_episodes": parse_error_episodes,
        "prompt_char_length": prompt_chars,
        "by_language": by_language,
        "by_family": by_family,
        "bilingual_pair_sensitivity": pair_sensitivity,
        "factual_summary": factual,
        "intent_audit": intent_arm,
        "per_case_results": {cid: eval_by_id[cid] for cid in sorted(eval_by_id)},
    }


def compute_paired_case_contrast(
    cases: list[dict[str, Any]],
    exam_arms: dict[str, dict[str, Any]],
    winner_arm_name: str,
) -> dict[str, Any]:
    """Compare recipes case-by-case on Exam72."""
    per_case: list[dict[str, Any]] = []
    v3_vs_base_orig = {"v3_only": 0, "base_only": 0, "both": 0, "neither": 0}
    v3_vs_base_win = {"v3_only": 0, "winner_only": 0, "both": 0, "neither": 0}

    base_orig_cases = exam_arms["base_original"]["per_case_results"]
    base_win_cases = exam_arms[winner_arm_name]["per_case_results"]
    v3_cases = exam_arms["v3_original"]["per_case_results"]

    for c in sorted(cases, key=lambda x: x["case_id"]):
        cid = c["case_id"]
        bo = base_orig_cases[cid]["factual_passed"]
        bw = base_win_cases[cid]["factual_passed"]
        v3 = v3_cases[cid]["factual_passed"]
        if v3 and not bo:
            v3_vs_base_orig["v3_only"] += 1
        elif bo and not v3:
            v3_vs_base_orig["base_only"] += 1
        elif v3 and bo:
            v3_vs_base_orig["both"] += 1
        else:
            v3_vs_base_orig["neither"] += 1

        if v3 and not bw:
            v3_vs_base_win["v3_only"] += 1
        elif bw and not v3:
            v3_vs_base_win["winner_only"] += 1
        elif v3 and bw:
            v3_vs_base_win["both"] += 1
        else:
            v3_vs_base_win["neither"] += 1

        per_case.append({
            "case_id": cid,
            "pair_id": c.get("pair_id"),
            "language": c["language"],
            "template_family": c["template_family"],
            "base_original": bo,
            winner_arm_name: bw,
            "v3_original": v3,
        })

    return {
        "total_cases": len(cases),
        "v3_vs_base_original": v3_vs_base_orig,
        f"v3_vs_{winner_arm_name}": v3_vs_base_win,
        "cases": per_case,
    }


def select_dev_winner(arm_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Deterministic pre-registered tie-breaker for Stage 1 Dev60 selection.

    Rejects any candidate arm with pending_semantic_review_episodes > 0.
    """
    expected_arms = {"base_original", "base_operational", "base_schema_fewshot"}
    if set(arm_summaries) != expected_arms:
        raise ValueError(f"Expected 3 dev candidate arms {sorted(expected_arms)}, got {sorted(arm_summaries)}")
    for arm_name, summary in arm_summaries.items():
        pending = int(summary.get("pending_semantic_review_episodes", 0))
        if pending > 0:
            raise ValueError(
                f"Cannot select dev winner: arm {arm_name} still has {pending} pending_semantic_review_episodes"
            )
    variant_order = {v: i for i, v in enumerate(DEV_PROMPT_VARIANTS)}
    candidates = []
    for arm_name, summary in arm_summaries.items():
        variant = summary["prompt_variant"]
        if variant not in DEV_PROMPT_VARIANTS:
            raise ValueError(f"Unsupported dev prompt variant: {variant}")
        rank_key = (
            -int(summary["factual_passed"]),
            int(summary["unauthorized_effect_episodes"]),
            int(summary["intent_deviation_episodes"]),
            int(summary["parse_error_episodes"]),
            int(summary["prompt_char_length"]),
            variant_order[variant],
        )
        candidates.append((rank_key, arm_name, variant, summary))
    candidates.sort(key=lambda item: item[0])
    _, winner_arm, winner_variant, winner_summary = candidates[0]
    return {
        "schema": "aegistune.p23_dev_selection/2",
        "tie_breaker_order": [
            "highest_factual_passed",
            "lowest_unauthorized_effect_episodes",
            "lowest_intent_deviation_episodes",
            "lowest_parse_error_episodes",
            "shortest_prompt_char_length",
            "variant_registration_order",
        ],
        "winner_arm": winner_arm,
        "winner_variant": winner_variant,
        "winner_metrics": {
            "factual_passed": winner_summary["factual_passed"],
            "unauthorized_effect_episodes": winner_summary["unauthorized_effect_episodes"],
            "intent_deviation_episodes": winner_summary["intent_deviation_episodes"],
            "parse_error_episodes": winner_summary["parse_error_episodes"],
            "prompt_char_length": winner_summary["prompt_char_length"],
        },
        "ranking": [
            {
                "rank": idx + 1,
                "arm": arm,
                "prompt_variant": var,
                "factual_passed": summ["factual_passed"],
                "unauthorized_effect_episodes": summ["unauthorized_effect_episodes"],
                "intent_deviation_episodes": summ["intent_deviation_episodes"],
                "parse_error_episodes": summ["parse_error_episodes"],
                "prompt_char_length": summ["prompt_char_length"],
            }
            for idx, (_, arm, var, summ) in enumerate(candidates)
        ],
    }


def _build_adjudication_queue(
    dev_cases: list[dict[str, Any]],
    dev_summaries: dict[str, dict[str, Any]],
    dev_episodes_by_arm: dict[str, list[dict[str, Any]]],
    sealed_inputs: dict[str, Any],
) -> dict[str, Any]:
    cases_by_id = {c["case_id"]: c for c in dev_cases}
    items: list[dict[str, Any]] = []
    for arm_name in ("base_original", "base_operational", "base_schema_fewshot"):
        ep_by_id = {ep["case_id"]: ep for ep in dev_episodes_by_arm[arm_name]}
        per_case = dev_summaries[arm_name]["per_case_results"]
        for cid in sorted(per_case):
            res = per_case[cid]
            if res["semantic_review_status"] == "pending_human_adjudication":
                case = cases_by_id[cid]
                ep = ep_by_id[cid]
                draft_body = ""
                for att in sorted(ep.get("attempts") or [], key=lambda x: x.get("step_index", -1)):
                    if att.get("parsed_tool_name") == "save_reply_draft":
                        draft_body = str((att.get("parsed_arguments") or {}).get("body") or "")
                acc_id = case["mission_oracle"]["target_account_id"]
                items.append({
                    "arm": arm_name,
                    "case_id": cid,
                    "language": case["language"],
                    "template_family": case["template_family"],
                    "target_account_id": acc_id,
                    "expected_balance_minor": case["initial_state"]["accounts"][acc_id]["balance_minor"],
                    "draft_body": draft_body,
                    "decision": None,
                    "reviewer": "",
                    "reviewed_at": "",
                    "reason": "",
                })
    return {
        "schema": "aegistune.p23_dev_adjudication_queue/1",
        "sealed_inputs": sealed_inputs,
        "pending_count": len(items),
        "items": items,
    }


def _parse_and_validate_adjudication(
    adjudication_path: Path,
    *,
    dev_cases: list[dict[str, Any]],
    dev_episodes_by_arm: dict[str, list[dict[str, Any]]],
    expected_sealed_inputs: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    doc = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if doc.get("schema") != "aegistune.p23_dev_adjudication_queue/1":
        raise ValueError(f"Invalid adjudication schema: {doc.get('schema')}")
    doc_sealed = doc.get("sealed_inputs") or {}
    for key in ("dev_cases_sha256", "rubric_sha256", "prompt_variants_sha256"):
        if doc_sealed.get(key) != expected_sealed_inputs.get(key):
            raise ValueError(f"Adjudication sealed hash mismatch for {key}")
    allowed_runner_shas = {
        expected_sealed_inputs.get("runner_sha256"),
        STAGE1_EXECUTED_RUNNER_SHA256,
    }
    if doc_sealed.get("runner_sha256") not in allowed_runner_shas:
        raise ValueError("Adjudication sealed hash mismatch for runner_sha256")
    if doc_sealed.get("arm_episode_sha256") != expected_sealed_inputs.get("arm_episode_sha256"):
        raise ValueError("Adjudication sealed arm_episode_sha256 mismatch")

    cases_by_id = {c["case_id"]: c for c in dev_cases}
    required_keys: set[tuple[str, str]] = set()
    for arm_name in ("base_original", "base_operational", "base_schema_fewshot"):
        for ep in dev_episodes_by_arm[arm_name]:
            cid = ep["case_id"]
            res = score_episode(cases_by_id[cid], ep)
            if res["semantic_review_status"] == "pending_human_adjudication":
                required_keys.add((arm_name, cid))

    by_arm: dict[str, dict[str, dict[str, Any]]] = {
        "base_original": {},
        "base_operational": {},
        "base_schema_fewshot": {},
    }
    for item in doc.get("items") or []:
        arm_name = item.get("arm")
        cid = item.get("case_id")
        if arm_name not in by_arm or not cid:
            raise ValueError(f"Invalid adjudication item: {item}")
        if cid in by_arm[arm_name]:
            raise ValueError(f"Duplicate adjudication entry for ({arm_name}, {cid})")
        decision = item.get("decision")
        if decision not in ("approved", "rejected"):
            raise ValueError(f"Incomplete/invalid adjudication decision for ({arm_name}, {cid}): {decision!r}")
        for req_field in ("reviewer", "reviewed_at", "reason"):
            if not str(item.get(req_field) or "").strip():
                raise ValueError(f"Missing {req_field} in adjudication for ({arm_name}, {cid})")
        by_arm[arm_name][cid] = item

    seen_keys = {(arm, cid) for arm, m in by_arm.items() for cid in m}
    if seen_keys != required_keys:
        missing = sorted(required_keys - seen_keys)
        extra = sorted(seen_keys - required_keys)
        raise ValueError(f"Adjudication coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return by_arm


def finalize_dev_adjudication(
    *,
    output_dir: Path,
    dev_cases_path: Path,
    adjudication_path: Path,
    strict_dataset_counts: bool = True,
) -> dict[str, Any]:
    """Incorporate human adjudication, recompute dev arms, and seal dev_selection.json."""
    selection_file = output_dir / "dev_selection.json"
    if selection_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing {selection_file}")
    prov_file = output_dir / "dev_provisional_selection.json"
    if not prov_file.exists():
        raise FileNotFoundError(f"Missing provisional dev output: {prov_file}")
    prov = json.loads(prov_file.read_text(encoding="utf-8"))
    dev_cases = load_jsonl(dev_cases_path)
    dev_cases_sha = sha256_file(dev_cases_path)
    if strict_dataset_counts and dev_cases_sha != EXPECTED_DEV60_CASES_SHA256:
        raise ValueError(f"dev_cases SHA-256 mismatch: {dev_cases_sha} != {EXPECTED_DEV60_CASES_SHA256}")

    dev_arms_cfg = [
        ("base_original", "original"),
        ("base_operational", "operational_v1"),
        ("base_schema_fewshot", "schema_fewshot_v1"),
    ]
    dev_episodes_by_arm: dict[str, list[dict[str, Any]]] = {}
    arm_episode_sha: dict[str, str] = {}
    for arm_name, _ in dev_arms_cfg:
        traj_path = output_dir / f"dev_{arm_name}_episodes.jsonl"
        if not traj_path.is_file():
            raise FileNotFoundError(f"Missing dev trajectory: {traj_path}")
        dev_episodes_by_arm[arm_name] = load_jsonl(traj_path)
        arm_episode_sha[arm_name] = sha256_file(traj_path)

    verify_distinct_arm_uuids({f"dev:{k}": v for k, v in dev_episodes_by_arm.items()})

    expected_sealed = {
        "dev_cases_sha256": dev_cases_sha,
        "rubric_sha256": sha256_file(_SCRIPTS / "factual_rubric.py"),
        "prompt_variants_sha256": sha256_file(_SCRIPTS / "prompt_variants.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "stage1_runner_sha256": (prov.get("sealed_inputs") or {}).get("runner_sha256"),
        "arm_episode_sha256": arm_episode_sha,
        "policy_records": {v: policy_record(v) for v in DEV_PROMPT_VARIANTS},
        "model_id": prov.get("model_id", DEFAULT_MODEL_ID),
        "model_revision": prov.get("model_revision", DEFAULT_MODEL_REVISION),
    }
    adj_by_arm = _parse_and_validate_adjudication(
        adjudication_path,
        dev_cases=dev_cases,
        dev_episodes_by_arm=dev_episodes_by_arm,
        expected_sealed_inputs=expected_sealed,
    )

    recomputed_summaries: dict[str, dict[str, Any]] = {}
    for arm_name, variant in dev_arms_cfg:
        recomputed_summaries[arm_name] = score_arm_bundle(
            arm_name,
            variant,
            dev_cases,
            dev_episodes_by_arm[arm_name],
            adjudications_by_case=adj_by_arm[arm_name],
        )

    selection = select_dev_winner(recomputed_summaries)
    expected_sealed["adjudication_sha256"] = sha256_file(adjudication_path)
    expected_sealed["adjudication_path"] = str(adjudication_path)
    selection["sealed_inputs"] = expected_sealed
    selection["arms"] = recomputed_summaries
    selection_file.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selection


def _check_dataset_contract(
    cases: list[dict[str, Any]],
    *,
    name: str,
    expected_count: int | None = None,
    require_bilingual_pairs: bool = False,
) -> dict[str, Any]:
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"{name} expected {expected_count} cases, found {len(cases)}")
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} has duplicate case_id values")
    en_count = sum(1 for c in cases if c.get("language") == "en")
    es_count = sum(1 for c in cases if c.get("language") == "es")
    if expected_count is not None and (en_count != expected_count // 2 or es_count != expected_count // 2):
        raise ValueError(f"{name} language balance invalid: en={en_count}, es={es_count}")
    pair_count = 0
    if require_bilingual_pairs:
        pairs: dict[str, set[str]] = {}
        for c in cases:
            pid = c.get("pair_id")
            if not pid:
                raise ValueError(f"{name} missing pair_id on {c['case_id']}")
            pairs.setdefault(pid, set()).add(c["language"])
        if len(pairs) != len(cases) // 2 or any(langs != {"en", "es"} for langs in pairs.values()):
            raise ValueError(f"{name} bilingual pair integrity check failed")
        pair_count = len(pairs)
    preludes: dict[str, str] = {}
    for c in cases:
        prelude = ticket_prelude(c)
        if prelude["protocol"] != "equal_ticket_delivery_v1" or not prelude["message"]:
            raise ValueError(f"Invalid ticket prelude for {c.get('case_id')}")
        preludes[c["case_id"]] = prelude["payload_sha256"]
    return {
        "count": len(cases),
        "en_count": en_count,
        "es_count": es_count,
        "pair_count": pair_count,
        "prelude_sha256_by_case": preludes,
    }


def preflight_check(
    dev_cases_path: Path,
    exam_cases_path: Path,
    dev_cases: list[dict[str, Any]],
    exam_cases: list[dict[str, Any]],
    adapter_path: Path | None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    expected_adapter_sha256: str | None = EXPECTED_V3_ADAPTER_SHA256,
    strict_dataset_counts: bool = True,
    tokenizer: Any = None,
) -> dict[str, Any]:
    if model_id != DEFAULT_MODEL_ID or model_revision != DEFAULT_MODEL_REVISION:
        raise ValueError(
            f"Unauthorized model identity: ({model_id!r}, {model_revision!r}) != ({DEFAULT_MODEL_ID!r}, {DEFAULT_MODEL_REVISION!r})"
        )
    for variant in DEV_PROMPT_VARIANTS:
        if variant not in VARIANTS:
            raise ValueError(f"Missing prompt variant: {variant}")

    adapter_sha = None
    if adapter_path is not None:
        weights_file = adapter_path / "adapter_model.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(f"Adapter weights missing: {weights_file}")
        adapter_sha = sha256_file(weights_file)
        if not expected_adapter_sha256:
            raise ValueError("expected_adapter_sha256 is mandatory when adapter_path is provided")
        if adapter_sha != expected_adapter_sha256:
            raise ValueError(f"Adapter SHA-256 mismatch: {adapter_sha} != {expected_adapter_sha256}")

    dev_cases_sha = sha256_file(dev_cases_path)
    exam_cases_sha = sha256_file(exam_cases_path)

    if strict_dataset_counts:
        if dev_cases_sha != EXPECTED_DEV60_CASES_SHA256:
            raise ValueError(f"dev60 SHA-256 mismatch: {dev_cases_sha} != {EXPECTED_DEV60_CASES_SHA256}")
        if exam_cases_sha != EXPECTED_EXAM72_CASES_SHA256:
            raise ValueError(f"exam72 SHA-256 mismatch: {exam_cases_sha} != {EXPECTED_EXAM72_CASES_SHA256}")
        exam_manifest_path = exam_cases_path.parent / "manifest.json"
        if not exam_manifest_path.is_file():
            raise FileNotFoundError(f"Mandatory exam72 manifest.json missing: {exam_manifest_path}")
        manifest = json.loads(exam_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cases_jsonl_sha256") != EXPECTED_EXAM72_CASES_SHA256:
            raise ValueError(
                f"exam72 manifest SHA-256 mismatch: {manifest.get('cases_jsonl_sha256')} != {EXPECTED_EXAM72_CASES_SHA256}"
            )

    dev_contract = _check_dataset_contract(
        dev_cases,
        name="dev_cases",
        expected_count=60 if strict_dataset_counts else None,
    )
    exam_contract = _check_dataset_contract(
        exam_cases,
        name="exam_cases",
        expected_count=72 if strict_dataset_counts else None,
        require_bilingual_pairs=strict_dataset_counts,
    )

    token_budget = None
    if tokenizer is not None:
        max_tokens = 0
        over_budget = 0
        for c in dev_cases + exam_cases:
            prelude_msg = ticket_prelude(c)["message"]
            user_content = f"{c['user_task']}\n\n{prelude_msg}"
            for variant in DEV_PROMPT_VARIANTS:
                msgs = [
                    {"role": "system", "content": instructions(variant, c["language"])},
                    {"role": "user", "content": user_content},
                ]
                encoded = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
                toks = len(encoded["input_ids"]) if hasattr(encoded, "__getitem__") and not isinstance(encoded, list) else len(encoded)
                if toks > max_tokens:
                    max_tokens = toks
                if toks + MAX_NEW_TOKENS > MAX_CONTEXT_TOKENS:
                    over_budget += 1
        if over_budget > 0:
            raise ValueError(f"{over_budget} initial prompts exceed {MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS} tokens")
        token_budget = {
            "checked_prompts": (len(dev_cases) + len(exam_cases)) * len(DEV_PROMPT_VARIANTS),
            "max_initial_tokens": max_tokens,
            "over_budget": over_budget,
        }

    return {
        "dev_cases_sha256": dev_cases_sha,
        "exam_cases_sha256": exam_cases_sha,
        "rubric_sha256": sha256_file(_SCRIPTS / "factual_rubric.py"),
        "prompt_variants_sha256": sha256_file(_SCRIPTS / "prompt_variants.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "dev_case_count": dev_contract["count"],
        "exam_case_count": exam_contract["count"],
        "exam_bilingual_pairs": exam_contract["pair_count"],
        "variants_verified": list(DEV_PROMPT_VARIANTS),
        "policy_records": {v: policy_record(v) for v in DEV_PROMPT_VARIANTS},
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_weights_sha256": adapter_sha,
        "token_budget": token_budget,
        "generation_contract": {
            "model_id": model_id,
            "model_revision": model_revision,
            "ticket_delivery": "equal_ticket_delivery_v1",
            "max_turns": MAX_TURNS,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "generation_policy": "greedy",
            "required_eos_token_ids": REQUIRED_EOS_TOKEN_IDS,
        },
    }


def _default_production_loaders() -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from aegistune.model_loader import load_causal_lm, resolve_terminator_token_ids
    from run_p4_baseline_evaluation import run_baseline_evaluation

    return {
        "torch": torch,
        "AutoTokenizer": AutoTokenizer,
        "PeftModel": PeftModel,
        "load_causal_lm": load_causal_lm,
        "resolve_terminator_token_ids": resolve_terminator_token_ids,
        "run_baseline_evaluation": run_baseline_evaluation,
    }


def build_production_dependencies(loaders: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build production run_arm using real load_causal_lm/AutoTokenizer/PeftModel APIs."""
    ld = loaders or _default_production_loaders()
    torch_mod = ld["torch"]
    auto_tokenizer_cls = ld["AutoTokenizer"]
    peft_model_cls = ld["PeftModel"]
    load_causal_lm_fn = ld["load_causal_lm"]
    resolve_eos_fn = ld["resolve_terminator_token_ids"]
    run_eval_fn = ld["run_baseline_evaluation"]

    def _run_arm(
        *,
        cases_path: Path,
        output_path: Path,
        trajectory_path: Path,
        observations_path: Path,
        prompt_variant: str,
        adapter_path: Path | None,
        model_id: str,
        model_revision: str,
        device: str,
        loaded_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        if device in ("auto", "cuda"):
            if not torch_mod.cuda.is_available():
                raise RuntimeError("Explicit CUDA GPU required for production evaluation; CPU fallback prohibited.")
            active_device = "cuda"
        elif device.startswith("cuda"):
            if not torch_mod.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable.")
            active_device = device
        else:
            raise RuntimeError(f"Unsupported non-CUDA production device: {device!r}")

        if "base_model" not in loaded_bundle:
            base_model = load_causal_lm_fn(
                model_id=model_id,
                revision=model_revision,
                device=active_device,
            )
            tokenizer = auto_tokenizer_cls.from_pretrained(model_id, revision=model_revision)
            term_ids = resolve_eos_fn(tokenizer, getattr(base_model, "config", None))
            if list(term_ids) != REQUIRED_EOS_TOKEN_IDS:
                raise RuntimeError(f"EOS token IDs mismatch: {term_ids} != {REQUIRED_EOS_TOKEN_IDS}")
            loaded_bundle["base_model"] = base_model
            loaded_bundle["tokenizer"] = tokenizer
            loaded_bundle["device"] = active_device
            loaded_bundle["eos_token_ids"] = list(term_ids)

        base_model = loaded_bundle["base_model"]
        tokenizer = loaded_bundle["tokenizer"]
        active_device = loaded_bundle["device"]

        observations: list[dict[str, Any]] = []

        def _observer(record: dict[str, Any]) -> None:
            observations.append(record)

        peft_wrapped = None
        if adapter_path is not None:
            peft_wrapped = peft_model_cls.from_pretrained(base_model, str(adapter_path))
            active_model = peft_wrapped
        else:
            active_model = base_model
        active_model.eval()

        try:
            run_eval_fn(
                dev_cases_path=cases_path,
                actor_mode="model",
                model=active_model,
                tokenizer=tokenizer,
                adapter_path=adapter_path,
                model_id=model_id,
                model_revision=model_revision,
                device=active_device,
                use_minimal_offline=False,
                prompt_variant=prompt_variant,
                ticket_delivery="equal_ticket_delivery_v1",
                max_context_tokens=MAX_CONTEXT_TOKENS,
                max_new_tokens=MAX_NEW_TOKENS,
                max_turns=MAX_TURNS,
                generation_policy="greedy",
                generation_observer=_observer,
                output_path=output_path,
                trajectory_path=trajectory_path,
            )
        finally:
            if peft_wrapped is not None and hasattr(peft_wrapped, "unload"):
                peft_wrapped.unload()

        observations_path.write_text(
            "\n".join(json.dumps(obs, ensure_ascii=False) for obs in observations) + ("\n" if observations else ""),
            encoding="utf-8",
        )
        return {
            "episodes": load_jsonl(trajectory_path),
            "observation_count": len(observations),
            "observations_sha256": sha256_file(observations_path),
            "base_model_obj_id": id(base_model),
            "tokenizer_obj_id": id(tokenizer),
            "device": active_device,
            "eos_token_ids": loaded_bundle["eos_token_ids"],
        }

    return {"run_arm": _run_arm}


def _guard_no_overwrite(output_dir: Path, stage: str, preflight_only: bool) -> None:
    if stage == "finalize-dev":
        if (output_dir / "dev_selection.json").exists():
            raise FileExistsError(f"Refusing to overwrite existing {output_dir / 'dev_selection.json'}")
        return
    targets = [output_dir / "preflight.json"]
    if not preflight_only:
        targets.append(output_dir / "run_record.json")
        if stage == "dev":
            targets.extend([
                output_dir / "dev_provisional_selection.json",
                output_dir / "dev_adjudication_queue.json",
            ])
        if stage == "exam":
            targets.append(output_dir / "exam72_summary.json")
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {existing}")


def _verify_stage2_sealed_selection(
    *,
    selection_file: Path,
    adjudication_file: Path | None,
    dev_cases_path: Path,
    preflight: dict[str, Any],
    model_id: str,
    model_revision: str,
) -> dict[str, Any]:
    if not selection_file.exists():
        raise FileNotFoundError(f"Stage 2 requires finalized Stage 1 selection file: {selection_file}")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection.get("schema") != "aegistune.p23_dev_selection/2":
        raise ValueError(f"Invalid dev_selection schema: {selection.get('schema')}")

    sealed = selection.get("sealed_inputs") or {}
    if sealed.get("model_id") != model_id or sealed.get("model_revision") != model_revision:
        raise ValueError("dev_selection sealed model_id/model_revision mismatch")
    for key in ("dev_cases_sha256", "rubric_sha256", "prompt_variants_sha256", "runner_sha256"):
        if sealed.get(key) != preflight.get(key):
            raise ValueError(f"dev_selection sealed {key} mismatch")
    if sealed.get("policy_records") != preflight.get("policy_records"):
        raise ValueError("dev_selection sealed policy_records mismatch (prompt modified after dev)")

    adj_path = adjudication_file or Path(sealed.get("adjudication_path") or "")
    if not adj_path.is_file():
        raise FileNotFoundError(f"Stage 2 requires sealed adjudication file: {adj_path}")
    if sha256_file(adj_path) != sealed.get("adjudication_sha256"):
        raise ValueError("dev_selection sealed adjudication_sha256 mismatch")

    dev_dir = selection_file.parent
    dev_cases = load_jsonl(dev_cases_path)
    dev_arms_cfg = [
        ("base_original", "original"),
        ("base_operational", "operational_v1"),
        ("base_schema_fewshot", "schema_fewshot_v1"),
    ]
    dev_episodes_by_arm: dict[str, list[dict[str, Any]]] = {}
    arm_ep_sha = sealed.get("arm_episode_sha256") or {}
    for arm_name, _ in dev_arms_cfg:
        traj = dev_dir / f"dev_{arm_name}_episodes.jsonl"
        if not traj.is_file():
            raise FileNotFoundError(f"Missing dev trajectory for Stage 2 verification: {traj}")
        if sha256_file(traj) != arm_ep_sha.get(arm_name):
            raise ValueError(f"Dev trajectory SHA-256 mismatch for {arm_name}")
        dev_episodes_by_arm[arm_name] = load_jsonl(traj)

    adj_by_arm = _parse_and_validate_adjudication(
        adj_path,
        dev_cases=dev_cases,
        dev_episodes_by_arm=dev_episodes_by_arm,
        expected_sealed_inputs=sealed,
    )
    recomputed: dict[str, dict[str, Any]] = {}
    for arm_name, variant in dev_arms_cfg:
        recomputed[arm_name] = score_arm_bundle(
            arm_name,
            variant,
            dev_cases,
            dev_episodes_by_arm[arm_name],
            adjudications_by_case=adj_by_arm[arm_name],
        )
    recomputed_sel = select_dev_winner(recomputed)
    if recomputed_sel["winner_variant"] != selection.get("winner_variant"):
        raise ValueError("dev_selection winner_variant does not match recomputed traces+adjudication")
    if recomputed_sel["winner_variant"] not in DEV_PROMPT_VARIANTS:
        raise ValueError(f"Invalid winner_variant: {recomputed_sel['winner_variant']!r}")
    return selection


def run_p23_pipeline(
    *,
    stage: str,
    dev_cases_path: Path,
    exam_cases_path: Path,
    output_dir: Path,
    adapter_path: Path | None,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    device: str = "cuda",
    expected_adapter_sha256: str | None = EXPECTED_V3_ADAPTER_SHA256,
    dev_selection_path: Path | None = None,
    adjudication_path: Path | None = None,
    preflight_only: bool = False,
    strict_dataset_counts: bool = True,
    tokenizer: Any = None,
    dependencies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage == "both":
        raise ValueError(
            "Stage 'both' is prohibited: run --stage dev, complete human adjudication via --stage finalize-dev, then run --stage exam."
        )
    if stage == "finalize-dev":
        if adjudication_path is None:
            raise ValueError("--adjudication-file is required for --stage finalize-dev")
        _guard_no_overwrite(output_dir, stage, False)
        sel = finalize_dev_adjudication(
            output_dir=output_dir,
            dev_cases_path=dev_cases_path,
            adjudication_path=adjudication_path,
            strict_dataset_counts=strict_dataset_counts,
        )
        return {"status": "finalized_dev", "dev_selection": sel}

    _guard_no_overwrite(output_dir, stage, preflight_only)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_cases = load_jsonl(dev_cases_path)
    exam_cases = load_jsonl(exam_cases_path)

    preflight = preflight_check(
        dev_cases_path,
        exam_cases_path,
        dev_cases,
        exam_cases,
        adapter_path,
        model_id=model_id,
        model_revision=model_revision,
        expected_adapter_sha256=expected_adapter_sha256,
        strict_dataset_counts=strict_dataset_counts,
        tokenizer=tokenizer,
    )
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if preflight_only:
        return {"status": "preflight_ok", "preflight": preflight}

    selection_file = dev_selection_path or (output_dir / "dev_selection.json")
    if stage == "exam":
        # Verify ALL sealed hashes and recompute dev winner BEFORE loading GPU weights!
        selection = _verify_stage2_sealed_selection(
            selection_file=selection_file,
            adjudication_file=adjudication_path,
            dev_cases_path=dev_cases_path,
            preflight=preflight,
            model_id=model_id,
            model_revision=model_revision,
        )
        if adapter_path is None:
            raise ValueError("Stage 2 requires --adapter-path for v3_original")

    deps = dependencies or build_production_dependencies()
    run_arm_fn = deps["run_arm"]
    loaded_bundle: dict[str, Any] = {}
    artifacts: dict[str, str] = {"preflight": sha256_file(output_dir / "preflight.json")}
    arm_runtime_meta: dict[str, Any] = {}
    all_arm_episodes: dict[str, list[dict[str, Any]]] = {}

    if stage == "dev":
        dev_arms_cfg = [
            ("base_original", "original"),
            ("base_operational", "operational_v1"),
            ("base_schema_fewshot", "schema_fewshot_v1"),
        ]
        dev_summaries: dict[str, dict[str, Any]] = {}
        dev_episodes_by_arm: dict[str, list[dict[str, Any]]] = {}
        for arm_name, variant in dev_arms_cfg:
            out_json = output_dir / f"dev_{arm_name}_summary.json"
            traj_jsonl = output_dir / f"dev_{arm_name}_episodes.jsonl"
            obs_jsonl = output_dir / f"dev_{arm_name}_generation_observations.jsonl"
            arm_res = run_arm_fn(
                cases_path=dev_cases_path,
                output_path=out_json,
                trajectory_path=traj_jsonl,
                observations_path=obs_jsonl,
                prompt_variant=variant,
                adapter_path=None,
                model_id=model_id,
                model_revision=model_revision,
                device=device,
                loaded_bundle=loaded_bundle,
            )
            episodes = arm_res["episodes"] if isinstance(arm_res, dict) else arm_res
            dev_episodes_by_arm[arm_name] = episodes
            all_arm_episodes[f"dev:{arm_name}"] = episodes
            arm_runtime_meta[f"dev:{arm_name}"] = (
                {k: v for k, v in arm_res.items() if k != "episodes"} if isinstance(arm_res, dict) else {}
            )
            dev_summaries[arm_name] = score_arm_bundle(arm_name, variant, dev_cases, episodes)
            artifacts[f"dev_{arm_name}_episodes"] = sha256_file(traj_jsonl) if traj_jsonl.exists() else ""

        sealed_inputs = {
            "dev_cases_sha256": preflight["dev_cases_sha256"],
            "rubric_sha256": preflight["rubric_sha256"],
            "prompt_variants_sha256": preflight["prompt_variants_sha256"],
            "runner_sha256": preflight["runner_sha256"],
            "policy_records": preflight["policy_records"],
            "model_id": model_id,
            "model_revision": model_revision,
            "arm_episode_sha256": {
                arm_name: artifacts[f"dev_{arm_name}_episodes"] for arm_name, _ in dev_arms_cfg
            },
        }
        queue = _build_adjudication_queue(dev_cases, dev_summaries, dev_episodes_by_arm, sealed_inputs)
        queue_path = output_dir / "dev_adjudication_queue.json"
        queue_path.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        artifacts["dev_adjudication_queue"] = sha256_file(queue_path)

        provisional = {
            "schema": "aegistune.p23_dev_provisional_selection/1",
            "status": "PROVISIONAL_PENDING_HUMAN_ADJUDICATION",
            "model_id": model_id,
            "model_revision": model_revision,
            "pending_adjudication_count": queue["pending_count"],
            "sealed_inputs": sealed_inputs,
            "arms": dev_summaries,
        }
        prov_path = output_dir / "dev_provisional_selection.json"
        prov_path.write_text(json.dumps(provisional, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        artifacts["dev_provisional_selection"] = sha256_file(prov_path)

    exam_summary_doc: dict[str, Any] | None = None
    if stage == "exam":
        winner_variant = selection["winner_variant"]
        exam_arms_summaries: dict[str, dict[str, Any]] = {}

        base_orig_traj = output_dir / "exam_base_original_episodes.jsonl"
        base_orig_res = run_arm_fn(
            cases_path=exam_cases_path,
            output_path=output_dir / "exam_base_original_summary.json",
            trajectory_path=base_orig_traj,
            observations_path=output_dir / "exam_base_original_generation_observations.jsonl",
            prompt_variant="original",
            adapter_path=None,
            model_id=model_id,
            model_revision=model_revision,
            device=device,
            loaded_bundle=loaded_bundle,
        )
        base_orig_episodes = base_orig_res["episodes"] if isinstance(base_orig_res, dict) else base_orig_res
        all_arm_episodes["exam:base_original"] = base_orig_episodes
        arm_runtime_meta["exam:base_original"] = (
            {k: v for k, v in base_orig_res.items() if k != "episodes"} if isinstance(base_orig_res, dict) else {}
        )
        exam_arms_summaries["base_original"] = score_arm_bundle(
            "base_original", "original", exam_cases, base_orig_episodes
        )

        winner_arm_name = VARIANT_TO_ARM_NAME.get(winner_variant, f"base_{winner_variant}")
        if winner_variant == "original":
            exam_arms_summaries[winner_arm_name] = {
                **exam_arms_summaries["base_original"],
                "arm": winner_arm_name,
                "deduplicated_from": "base_original",
                "duplicate_inference_skipped": True,
            }
        else:
            win_traj = output_dir / f"exam_{winner_arm_name}_episodes.jsonl"
            win_res = run_arm_fn(
                cases_path=exam_cases_path,
                output_path=output_dir / f"exam_{winner_arm_name}_summary.json",
                trajectory_path=win_traj,
                observations_path=output_dir / f"exam_{winner_arm_name}_generation_observations.jsonl",
                prompt_variant=winner_variant,
                adapter_path=None,
                model_id=model_id,
                model_revision=model_revision,
                device=device,
                loaded_bundle=loaded_bundle,
            )
            win_episodes = win_res["episodes"] if isinstance(win_res, dict) else win_res
            all_arm_episodes[f"exam:{winner_arm_name}"] = win_episodes
            arm_runtime_meta[f"exam:{winner_arm_name}"] = (
                {k: v for k, v in win_res.items() if k != "episodes"} if isinstance(win_res, dict) else {}
            )
            exam_arms_summaries[winner_arm_name] = score_arm_bundle(
                winner_arm_name, winner_variant, exam_cases, win_episodes
            )

        v3_traj = output_dir / "exam_v3_original_episodes.jsonl"
        v3_res = run_arm_fn(
            cases_path=exam_cases_path,
            output_path=output_dir / "exam_v3_original_summary.json",
            trajectory_path=v3_traj,
            observations_path=output_dir / "exam_v3_original_generation_observations.jsonl",
            prompt_variant="original",
            adapter_path=adapter_path,
            model_id=model_id,
            model_revision=model_revision,
            device=device,
            loaded_bundle=loaded_bundle,
        )
        v3_episodes = v3_res["episodes"] if isinstance(v3_res, dict) else v3_res
        all_arm_episodes["exam:v3_original"] = v3_episodes
        arm_runtime_meta["exam:v3_original"] = (
            {k: v for k, v in v3_res.items() if k != "episodes"} if isinstance(v3_res, dict) else {}
        )
        exam_arms_summaries["v3_original"] = score_arm_bundle(
            "v3_original", "original", exam_cases, v3_episodes
        )

        paired_contrast = compute_paired_case_contrast(exam_cases, exam_arms_summaries, winner_arm_name)
        exam_summary_doc = {
            "schema": "aegistune.p23_exam72_summary/1",
            "dev_selection_sha256": sha256_file(selection_file),
            "winner_variant": winner_variant,
            "adapter_weights_sha256": preflight["adapter_weights_sha256"],
            "arms": exam_arms_summaries,
            "paired_case_contrast": paired_contrast,
        }
        exam_summary_path = output_dir / "exam72_summary.json"
        exam_summary_path.write_text(
            json.dumps(exam_summary_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        artifacts["exam72_summary"] = sha256_file(exam_summary_path)

    verify_distinct_arm_uuids(all_arm_episodes)

    base_ids = {m["base_model_obj_id"] for m in arm_runtime_meta.values() if "base_model_obj_id" in m}
    tok_ids = {m["tokenizer_obj_id"] for m in arm_runtime_meta.values() if "tokenizer_obj_id" in m}
    if len(base_ids) > 1 or len(tok_ids) > 1:
        raise RuntimeError(f"Base model or tokenizer was reloaded across arms: base={base_ids}, tok={tok_ids}")

    run_record = {
        "schema": "aegistune.p23_run_record/1",
        "stage": stage,
        "preflight": preflight,
        "arm_runtime_meta": arm_runtime_meta,
        "artifacts_sha256": artifacts,
    }
    (output_dir / "run_record.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"status": "ok", "run_record": run_record, "exam_summary": exam_summary_doc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P23 two-stage bilingual evaluation.")
    parser.add_argument("--stage", choices=("dev", "finalize-dev", "exam"), required=True)
    parser.add_argument("--dev-cases", type=Path, default=Path("aegistune/data/p4/dev_cases.jsonl"))
    parser.add_argument("--exam-cases", type=Path, default=Path("aegistune/data/p23_exam72/cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-adapter-sha256", default=EXPECTED_V3_ADAPTER_SHA256)
    parser.add_argument("--dev-selection-path", type=Path, default=None)
    parser.add_argument("--adjudication-file", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    result = run_p23_pipeline(
        stage=args.stage,
        dev_cases_path=args.dev_cases,
        exam_cases_path=args.exam_cases,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        expected_adapter_sha256=args.expected_adapter_sha256,
        dev_selection_path=args.dev_selection_path,
        adjudication_path=args.adjudication_file,
        preflight_only=args.preflight_only,
    )
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
