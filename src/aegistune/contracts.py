"""Immutable contracts, canonical serialization, and SHA-256 digests for AegisTune."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from typing import Any

MAX_AMOUNT_MINOR: int = 1_000_000_000  # 10,000,000.00 USD in integer cents
MAX_DRAFT_LENGTH: int = 4096
MAX_REASON_LENGTH: int = 1024
MAX_STRING_FIELD_LENGTH: int = 2048

FORBIDDEN_PUBLIC_KEYS: frozenset[str] = frozenset(
    {
        "is_adversarial",
        "should_block",
        "mission_oracle",
        "oracle",
        "ground_truth",
        "future_outcome",
        "expected_effects",
        "cohort",
        "secret_label",
    }
)

FORBIDDEN_CASE_ID_SUBSTRINGS: tuple[str, ...] = (
    "adversarial",
    "should_block",
    "attack",
    "benign",
    "ground_truth",
    "unsafe",
)


class ContractValidationError(ValueError):
    """Raised when a contract invariant or join requirement is violated."""


class FrozenMap(Mapping[str, Any]):
    """Deeply immutable mapping with deterministic key ordering and hashing."""

    __slots__ = ("_items", "_dict", "_hash")

    def __init__(self, data: Mapping[str, Any] | tuple[tuple[str, Any], ...] = ()) -> None:
        if isinstance(data, FrozenMap):
            self._items = data._items
            self._dict = data._dict
            self._hash = data._hash
            return
        raw_items = data.items() if isinstance(data, Mapping) else data
        normalized: list[tuple[str, Any]] = []
        for k, v in raw_items:
            if not isinstance(k, str):
                raise ContractValidationError(f"Mapping keys must be str, got {type(k).__name__}")
            normalized.append((k, freeze_value(v)))
        normalized.sort(key=lambda pair: pair[0])
        self._items: tuple[tuple[str, Any], ...] = tuple(normalized)
        self._dict: dict[str, Any] = dict(self._items)
        self._hash: int = hash(self._items)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMap):
            return self._items == other._items
        if isinstance(other, Mapping):
            try:
                return self._items == FrozenMap(other)._items
            except ContractValidationError:
                return False
        return False

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self._items)!r})"

    def to_plain_dict(self) -> dict[str, Any]:
        return {k: thaw_value(v) for k, v in self._items}


def is_strict_int(value: Any) -> bool:
    """Return True if value is a Python int and NOT a bool."""
    return type(value) is int


def validate_amount_minor(value: Any, *, field_name: str = "amount_minor") -> int:
    """Validate an integer USD cent amount strictly positive and within bounds."""
    if not is_strict_int(value):
        raise ContractValidationError(
            f"{field_name} must be a strict int (not {type(value).__name__})"
        )
    if value <= 0:
        raise ContractValidationError(f"{field_name} must be > 0, got {value}")
    if value > MAX_AMOUNT_MINOR:
        raise ContractValidationError(
            f"{field_name} exceeds maximum allowed minor units ({MAX_AMOUNT_MINOR}): {value}"
        )
    return value


def freeze_value(value: Any) -> Any:
    """Recursively freeze a JSON-like value into an immutable representation."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if is_strict_int(value):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"Non-finite float is not permitted: {value}")
        return value
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        frozen_items = [freeze_value(item) for item in value]
        return tuple(sorted(frozen_items, key=canonical_json))
    if is_dataclass(value) and not isinstance(value, type):
        return value
    raise ContractValidationError(f"Unsupported value type for freezing: {type(value).__name__}")


def thaw_value(value: Any) -> Any:
    """Convert frozen structures back into plain JSON-serializable Python types."""
    if isinstance(value, FrozenMap):
        return {k: thaw_value(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: thaw_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw_value(v) for v in value]
    if isinstance(value, (frozenset, set)):
        return sorted(thaw_value(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return thaw_value(asdict(value))
    return value


def canonical_json(value: Any) -> str:
    """Serialize any frozen or plain JSON-compatible structure canonically."""
    plain = thaw_value(freeze_value(value))
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_digest(value: Any) -> str:
    """Compute a full 64-hex-character SHA-256 digest of a canonical structure or raw bytes."""
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _check_no_forbidden_keys(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if k in FORBIDDEN_PUBLIC_KEYS:
                raise ContractValidationError(
                    f"Forbidden oracle/label key '{k}' found in {context}"
                )
            _check_no_forbidden_keys(v, context=f"{context}.{k}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _check_no_forbidden_keys(item, context=f"{context}[{idx}]")


@dataclass(frozen=True)
class AccountRecord:
    """Initial synthetic account record."""

    account_id: str
    currency: str
    balance_minor: int
    verified_iban: str
    status: str = "active"
    internal_notes: str = ""

    def __post_init__(self) -> None:
        if not self.account_id or not isinstance(self.account_id, str):
            raise ContractValidationError("account_id must be a non-empty str")
        if self.currency != "USD":
            raise ContractValidationError(f"Only USD currency is supported, got {self.currency!r}")
        if not is_strict_int(self.balance_minor) or self.balance_minor < 0 or self.balance_minor > MAX_AMOUNT_MINOR:
            raise ContractValidationError(f"Invalid balance_minor: {self.balance_minor!r}")
        if not self.verified_iban or not isinstance(self.verified_iban, str):
            raise ContractValidationError("verified_iban must be a non-empty str")

    def public_ledger_fields(self) -> FrozenMap:
        """Return strictly the permitted ledger fields for lookup_account_ledger."""
        return FrozenMap(
            {
                "account_id": self.account_id,
                "currency": self.currency,
                "balance_minor": self.balance_minor,
                "verified_iban": self.verified_iban,
                "status": self.status,
            }
        )


@dataclass(frozen=True)
class TicketRecord:
    """Initial synthetic support ticket record."""

    ticket_id: str
    account_id: str
    subject: str
    body: str
    channel: str = "email"

    def __post_init__(self) -> None:
        if not self.ticket_id or not isinstance(self.ticket_id, str):
            raise ContractValidationError("ticket_id must be a non-empty str")
        if not self.account_id or not isinstance(self.account_id, str):
            raise ContractValidationError("account_id must be a non-empty str")
        if not isinstance(self.body, str):
            raise ContractValidationError("body must be a str")


@dataclass(frozen=True)
class TrustedEnvelope:
    """Immutable host-enforced operational envelope for a session."""

    session_id: str
    allowed_tools: frozenset[str]
    accessible_accounts: frozenset[str]
    accessible_tickets: frozenset[str]
    currency: str
    max_single_amount_minor: int
    max_total_amount_minor: int
    allowed_destinations: FrozenMap  # account_id -> verified_iban
    allowed_webhook_hosts: frozenset[str] = field(default_factory=frozenset)
    allowed_webhook_fields: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"event_type", "ticket_id", "account_id", "status", "classification"}
        )
    )
    allowed_webhook_classifications: frozenset[str] = field(
        default_factory=lambda: frozenset({"public_status", "billing_notice"})
    )
    host_supervisor_enabled: bool = True
    allow_stage_fallback: bool = True

    def __post_init__(self) -> None:
        if not self.session_id or not isinstance(self.session_id, str):
            raise ContractValidationError("session_id must be a non-empty str")
        object.__setattr__(self, "allowed_tools", frozenset(self.allowed_tools))
        object.__setattr__(self, "accessible_accounts", frozenset(self.accessible_accounts))
        object.__setattr__(self, "accessible_tickets", frozenset(self.accessible_tickets))
        object.__setattr__(self, "allowed_destinations", FrozenMap(self.allowed_destinations))
        object.__setattr__(self, "allowed_webhook_hosts", frozenset(self.allowed_webhook_hosts))
        object.__setattr__(self, "allowed_webhook_fields", frozenset(self.allowed_webhook_fields))
        object.__setattr__(
            self,
            "allowed_webhook_classifications",
            frozenset(self.allowed_webhook_classifications),
        )
        if self.currency != "USD":
            raise ContractValidationError(f"TrustedEnvelope currency must be 'USD', got {self.currency!r}")
        if not is_strict_int(self.max_single_amount_minor) or self.max_single_amount_minor < 0:
            raise ContractValidationError("max_single_amount_minor must be a non-negative strict int")
        if not is_strict_int(self.max_total_amount_minor) or self.max_total_amount_minor < 0:
            raise ContractValidationError("max_total_amount_minor must be a non-negative strict int")
        if self.max_single_amount_minor > self.max_total_amount_minor and self.max_total_amount_minor > 0:
            raise ContractValidationError(
                "max_single_amount_minor cannot exceed max_total_amount_minor"
            )

    def public_view(self) -> FrozenMap:
        """Return the non-secret public policy rules needed by the actor and sensor."""
        return FrozenMap(
            {
                "allowed_tools": sorted(self.allowed_tools),
                "accessible_accounts": sorted(self.accessible_accounts),
                "accessible_tickets": sorted(self.accessible_tickets),
                "currency": self.currency,
                "max_single_amount_minor": self.max_single_amount_minor,
                "max_total_amount_minor": self.max_total_amount_minor,
                "allowed_webhook_hosts": sorted(self.allowed_webhook_hosts),
                "allowed_webhook_fields": sorted(self.allowed_webhook_fields),
                "allowed_webhook_classifications": sorted(self.allowed_webhook_classifications),
            }
        )


@dataclass(frozen=True)
class CaseSpec:
    """Public and initial-state specification of a synthetic support/billing case."""

    case_id: str
    template_family: str
    split: str
    language: str
    user_task: str
    public_policy: FrozenMap
    initial_state: FrozenMap
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id or not isinstance(self.case_id, str):
            raise ContractValidationError("case_id must be a non-empty str")
        lower_id = self.case_id.lower()
        for token in FORBIDDEN_CASE_ID_SUBSTRINGS:
            if token in lower_id:
                raise ContractValidationError(
                    f"case_id {self.case_id!r} must not leak label/answer token {token!r}"
                )
        if self.split not in {"train", "dev", "calibration", "test", "fixture"}:
            raise ContractValidationError(f"Unsupported split: {self.split!r}")
        if self.language not in {"en", "es"}:
            raise ContractValidationError(f"Unsupported language: {self.language!r}")
        if not self.user_task or not isinstance(self.user_task, str):
            raise ContractValidationError("user_task must be a non-empty str")
        frozen_policy = FrozenMap(self.public_policy)
        frozen_state = FrozenMap(self.initial_state)
        _check_no_forbidden_keys(frozen_policy, context="CaseSpec.public_policy")
        _check_no_forbidden_keys(frozen_state, context="CaseSpec.initial_state")
        object.__setattr__(self, "public_policy", frozen_policy)
        object.__setattr__(self, "initial_state", frozen_state)
        object.__setattr__(self, "provenance_refs", tuple(self.provenance_refs))

    def public_actor_prompt_view(self) -> FrozenMap:
        """Return only the actor-visible prompt information without internal state or labels."""
        return FrozenMap(
            {
                "scenario_id": self.case_id,
                "language": self.language,
                "user_task": self.user_task,
                "public_policy": self.public_policy,
            }
        )


@dataclass(frozen=True)
class MissionOracle:
    """Post-execution reward/evaluation oracle; never exposed to runtime gateway or sensor."""

    oracle_id: str
    case_id: str
    cohort: str  # "benign" | "adversarial"
    legitimate_actions: frozenset[str]
    target_account_id: str | None = None
    target_ticket_id: str | None = None
    authorized_amount_minor: int | None = None
    authorized_amount_range_minor: tuple[int, int] | None = None
    authorized_destination_iban: str | None = None
    max_total_amount_minor: int = 0
    max_refund_commits: int = 0
    expected_effects: FrozenMap = field(default_factory=FrozenMap)
    utility_predicate_spec: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        if not self.oracle_id or not isinstance(self.oracle_id, str):
            raise ContractValidationError("oracle_id must be a non-empty str")
        if not self.case_id or not isinstance(self.case_id, str):
            raise ContractValidationError("case_id must be a non-empty str")
        if self.cohort not in {"benign", "adversarial"}:
            raise ContractValidationError(f"cohort must be 'benign' or 'adversarial', got {self.cohort!r}")
        object.__setattr__(self, "legitimate_actions", frozenset(self.legitimate_actions))
        object.__setattr__(self, "expected_effects", FrozenMap(self.expected_effects))
        object.__setattr__(self, "utility_predicate_spec", FrozenMap(self.utility_predicate_spec))
        if self.authorized_amount_minor is not None:
            validate_amount_minor(self.authorized_amount_minor, field_name="authorized_amount_minor")
        if self.authorized_amount_range_minor is not None:
            low, high = self.authorized_amount_range_minor
            validate_amount_minor(low, field_name="authorized_amount_range_minor[0]")
            validate_amount_minor(high, field_name="authorized_amount_range_minor[1]")
            if low > high:
                raise ContractValidationError("authorized_amount_range_minor low cannot exceed high")
        if not is_strict_int(self.max_total_amount_minor) or self.max_total_amount_minor < 0:
            raise ContractValidationError("max_total_amount_minor must be a non-negative strict int")
        if not is_strict_int(self.max_refund_commits) or self.max_refund_commits < 0:
            raise ContractValidationError("max_refund_commits must be a non-negative strict int")


@dataclass(frozen=True)
class PreActionObservation:
    """Read-only observation delivered to an optional runtime monitor at t-."""

    episode_id: str
    generation_id: str
    step_index: int
    user_task: str
    public_policy: FrozenMap
    history: tuple[FrozenMap, ...]
    tool_name: str
    frozen_arguments: FrozenMap
    observed_metadata: FrozenMap = field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.generation_id:
            raise ContractValidationError("episode_id and generation_id are required")
        if not is_strict_int(self.step_index) or self.step_index < 0:
            raise ContractValidationError("step_index must be a non-negative strict int")
        frozen_policy = FrozenMap(self.public_policy)
        frozen_hist = tuple(FrozenMap(item) for item in self.history)
        frozen_args = FrozenMap(self.frozen_arguments)
        frozen_meta = FrozenMap(self.observed_metadata)
        _check_no_forbidden_keys(frozen_policy, context="PreActionObservation.public_policy")
        _check_no_forbidden_keys(frozen_hist, context="PreActionObservation.history")
        _check_no_forbidden_keys(frozen_args, context="PreActionObservation.frozen_arguments")
        _check_no_forbidden_keys(frozen_meta, context="PreActionObservation.observed_metadata")
        object.__setattr__(self, "public_policy", frozen_policy)
        object.__setattr__(self, "history", frozen_hist)
        object.__setattr__(self, "frozen_arguments", frozen_args)
        object.__setattr__(self, "observed_metadata", frozen_meta)


@dataclass(frozen=True)
class Attempt:
    """Recorded tool call attempt, created prior to any schema, permission, or monitor rejection."""

    attempt_id: str
    episode_id: str
    generation_id: str
    step_index: int
    raw_model_output: str
    parse_status: str  # "ok" | "malformed_json" | "invalid_proposal"
    parsed_tool_name: str | None
    parsed_arguments: FrozenMap | None
    parse_error: str | None
    payload_digest: str
    parent_attempt_id: str | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.episode_id or not self.generation_id:
            raise ContractValidationError("attempt_id, episode_id, and generation_id are required")
        if self.parse_status not in {"ok", "malformed_json", "invalid_proposal"}:
            raise ContractValidationError(f"Invalid parse_status: {self.parse_status!r}")
        if self.parsed_arguments is not None:
            object.__setattr__(self, "parsed_arguments", FrozenMap(self.parsed_arguments))
        if len(self.payload_digest) != 64:
            raise ContractValidationError("payload_digest must be a 64-char SHA-256 hex digest")


@dataclass(frozen=True)
class GatewayDecision:
    """Deterministic gateway + optional monitor decision bound to frozen payload and state version."""

    decision_id: str
    attempt_id: str
    episode_id: str
    generation_id: str
    allowed: bool
    status: str  # "allowed" | "denied" | "contained" | "error"
    reason_code: str
    validated_payload: FrozenMap | None
    payload_digest: str
    state_version: int
    applied_limits: FrozenMap
    monitor_verdict: FrozenMap | None = None

    def __post_init__(self) -> None:
        if not self.decision_id or not self.attempt_id:
            raise ContractValidationError("decision_id and attempt_id are required")
        if self.status not in {"allowed", "denied", "contained", "error"}:
            raise ContractValidationError(f"Invalid GatewayDecision status: {self.status!r}")
        if self.allowed and self.status != "allowed":
            raise ContractValidationError("allowed=True requires status='allowed'")
        if not self.allowed and self.status == "allowed":
            raise ContractValidationError("status='allowed' requires allowed=True")
        if self.validated_payload is not None:
            object.__setattr__(self, "validated_payload", FrozenMap(self.validated_payload))
        object.__setattr__(self, "applied_limits", FrozenMap(self.applied_limits))
        if self.monitor_verdict is not None:
            object.__setattr__(self, "monitor_verdict", FrozenMap(self.monitor_verdict))
        if len(self.payload_digest) != 64:
            raise ContractValidationError("payload_digest must be a 64-char SHA-256 hex digest")


@dataclass(frozen=True)
class ExecutionReceipt:
    """Environment-emitted execution receipt binding effective arguments, deltas, and state versions."""

    receipt_id: str
    attempt_id: str
    parent_attempt_id: str | None
    episode_id: str
    generation_id: str
    effective_tool_name: str | None
    effective_payload: FrozenMap | None
    payload_digest: str
    status: str  # "executed" | "committed" | "denied" | "contained" | "error"
    operation_id: str | None
    state_delta: FrozenMap
    state_version_before: int
    state_version_after: int
    output: FrozenMap
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.attempt_id:
            raise ContractValidationError("receipt_id and attempt_id are required")
        if self.status not in {"executed", "committed", "denied", "contained", "error"}:
            raise ContractValidationError(f"Invalid receipt status: {self.status!r}")
        if self.effective_payload is not None:
            object.__setattr__(self, "effective_payload", FrozenMap(self.effective_payload))
        object.__setattr__(self, "state_delta", FrozenMap(self.state_delta))
        object.__setattr__(self, "output", FrozenMap(self.output))
        if len(self.payload_digest) != 64:
            raise ContractValidationError("payload_digest must be a 64-char SHA-256 hex digest")


@dataclass(frozen=True)
class EpisodeResult:
    """Sealed episode result with strict 1-to-1 join validation and full chain digest."""

    episode_id: str
    generation_id: str
    case_id: str
    model_source: str  # "scripted" | "synthetic" | "model"
    model_revision: str
    termination_reason: str
    steps_executed: int
    delivered_observations: tuple[FrozenMap, ...]
    attempts: tuple[Attempt, ...]
    decisions: tuple[GatewayDecision, ...]
    receipts: tuple[ExecutionReceipt, ...]
    initial_state_snapshot: FrozenMap
    final_state_snapshot: FrozenMap
    initial_state_digest: str
    final_state_digest: str
    chain_digest: str
    valid: bool
    error: str | None = None
    closed_loop_verified: bool = False

    def __post_init__(self) -> None:
        if self.model_source not in {"scripted", "synthetic", "model"}:
            raise ContractValidationError(f"Invalid model_source: {self.model_source!r}")
        object.__setattr__(
            self,
            "delivered_observations",
            tuple(FrozenMap(obs) for obs in self.delivered_observations),
        )
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "receipts", tuple(self.receipts))
        object.__setattr__(self, "initial_state_snapshot", FrozenMap(self.initial_state_snapshot))
        object.__setattr__(self, "final_state_snapshot", FrozenMap(self.final_state_snapshot))
        if self.closed_loop_verified and self.model_source != "model":
            raise ContractValidationError(
                "closed_loop_verified cannot be True when model_source != 'model'"
            )
        validate_episode_joins(
            episode_id=self.episode_id,
            generation_id=self.generation_id,
            attempts=self.attempts,
            decisions=self.decisions,
            receipts=self.receipts,
            initial_state_snapshot=self.initial_state_snapshot,
            final_state_snapshot=self.final_state_snapshot,
            initial_state_digest=self.initial_state_digest,
            final_state_digest=self.final_state_digest,
            chain_digest=self.chain_digest,
        )


def compute_episode_chain_digest(
    *,
    episode_id: str,
    generation_id: str,
    case_id: str,
    initial_state_digest: str,
    attempts: tuple[Attempt, ...],
    decisions: tuple[GatewayDecision, ...],
    receipts: tuple[ExecutionReceipt, ...],
    final_state_digest: str,
) -> str:
    """Compute canonical SHA-256 chain digest binding initial state, attempts, decisions, receipts+deltas, and final state."""
    payload = {
        "episode_id": episode_id,
        "generation_id": generation_id,
        "case_id": case_id,
        "initial_state_digest": initial_state_digest,
        "attempts": [
            {
                "attempt_id": a.attempt_id,
                "parent_attempt_id": a.parent_attempt_id,
                "step_index": a.step_index,
                "parse_status": a.parse_status,
                "parsed_tool_name": a.parsed_tool_name,
                "parsed_arguments": a.parsed_arguments,
                "payload_digest": a.payload_digest,
            }
            for a in attempts
        ],
        "decisions": [
            {
                "decision_id": d.decision_id,
                "attempt_id": d.attempt_id,
                "allowed": d.allowed,
                "status": d.status,
                "reason_code": d.reason_code,
                "payload_digest": d.payload_digest,
                "state_version": d.state_version,
            }
            for d in decisions
        ],
        "receipts": [
            {
                "receipt_id": r.receipt_id,
                "attempt_id": r.attempt_id,
                "parent_attempt_id": r.parent_attempt_id,
                "effective_tool_name": r.effective_tool_name,
                "payload_digest": r.payload_digest,
                "status": r.status,
                "operation_id": r.operation_id,
                "state_delta": r.state_delta,
                "state_version_before": r.state_version_before,
                "state_version_after": r.state_version_after,
                "error_code": r.error_code,
            }
            for r in receipts
        ],
        "final_state_digest": final_state_digest,
    }
    return sha256_digest(payload)


def validate_episode_joins(
    *,
    episode_id: str,
    generation_id: str,
    attempts: tuple[Attempt, ...],
    decisions: tuple[GatewayDecision, ...],
    receipts: tuple[ExecutionReceipt, ...],
    initial_state_snapshot: FrozenMap,
    final_state_snapshot: FrozenMap,
    initial_state_digest: str,
    final_state_digest: str,
    chain_digest: str,
    case_id: str | None = None,
) -> None:
    """Enforce exact 1-to-1 ID cardinality and state/delta integrity across an episode."""
    if sha256_digest(initial_state_snapshot) != initial_state_digest:
        raise ContractValidationError("initial_state_digest does not match initial_state_snapshot")
    if sha256_digest(final_state_snapshot) != final_state_digest:
        raise ContractValidationError("final_state_digest does not match final_state_snapshot")

    if len(attempts) != len(decisions) or len(attempts) != len(receipts):
        raise ContractValidationError(
            f"Cardinality mismatch: attempts={len(attempts)}, decisions={len(decisions)}, receipts={len(receipts)}"
        )

    attempt_ids: list[str] = []
    for a in attempts:
        if a.episode_id != episode_id or a.generation_id != generation_id:
            raise ContractValidationError("Attempt episode/generation ID mismatch")
        attempt_ids.append(a.attempt_id)

    if len(set(attempt_ids)) != len(attempt_ids):
        raise ContractValidationError("Duplicate attempt_id detected in episode")

    decision_ids: set[str] = set()
    receipt_ids: set[str] = set()
    for a, d, r in zip(attempts, decisions, receipts, strict=True):
        if d.decision_id in decision_ids:
            raise ContractValidationError(f"Duplicate decision_id: {d.decision_id}")
        decision_ids.add(d.decision_id)
        if r.receipt_id in receipt_ids:
            raise ContractValidationError(f"Duplicate receipt_id: {r.receipt_id}")
        receipt_ids.add(r.receipt_id)

        if d.attempt_id != a.attempt_id or r.attempt_id != a.attempt_id:
            raise ContractValidationError(
                f"Join mismatch for attempt {a.attempt_id}: decision.attempt_id={d.attempt_id}, receipt.attempt_id={r.attempt_id}"
            )
        if d.episode_id != episode_id or d.generation_id != generation_id:
            raise ContractValidationError("GatewayDecision episode/generation ID mismatch")
        if r.episode_id != episode_id or r.generation_id != generation_id:
            raise ContractValidationError("ExecutionReceipt episode/generation ID mismatch")
        if d.payload_digest != a.payload_digest or r.payload_digest != d.payload_digest:
            raise ContractValidationError(
                f"Payload digest mismatch across attempt/decision/receipt for {a.attempt_id}"
            )
        if r.parent_attempt_id is not None and r.parent_attempt_id not in set(attempt_ids):
            raise ContractValidationError(
                f"Unknown parent_attempt_id {r.parent_attempt_id!r} on receipt {r.receipt_id}"
            )

    if case_id is not None:
        expected_chain = compute_episode_chain_digest(
            episode_id=episode_id,
            generation_id=generation_id,
            case_id=case_id,
            initial_state_digest=initial_state_digest,
            attempts=attempts,
            decisions=decisions,
            receipts=receipts,
            final_state_digest=final_state_digest,
        )
        if expected_chain != chain_digest:
            raise ContractValidationError("chain_digest mismatch: state or receipt delta was tampered")
