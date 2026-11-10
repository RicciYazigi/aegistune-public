"""Isolated per-episode/generation environment, atomic state transitions, and monitor boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import threading
from typing import Any
import uuid

from .contracts import (
    AccountRecord,
    Attempt,
    ContractValidationError,
    EpisodeResult,
    ExecutionReceipt,
    FrozenMap,
    GatewayDecision,
    PreActionObservation,
    TicketRecord,
    compute_episode_chain_digest,
    freeze_value,
    sha256_digest,
)
from .events import EpisodeEventLog, EventRecord, EventSinkError
from .gateway import GATEWAY_PARSER_V1_STRICT, DeterministicGateway
from .scenario import ScenarioBundle
from .tools import COMMIT_TOOL_NAMES, HostToolRegistry


@dataclass(frozen=True)
class StepExecutionResult:
    """Result returned by a single environment step, including optional fallback child records."""

    attempt: Attempt
    decision: GatewayDecision
    receipt: ExecutionReceipt
    observation_for_actor: FrozenMap
    fallback_attempt: Attempt | None = None
    fallback_decision: GatewayDecision | None = None
    fallback_receipt: ExecutionReceipt | None = None


MonitorCallable = Callable[[PreActionObservation], Mapping[str, Any]]


class SyntheticEnvironment:
    """Single-session isolated environment owning mutable state, lock, gateway, and event log."""

    def __init__(
        self,
        *,
        scenario: ScenarioBundle,
        episode_id: str,
        generation_id: str,
        model_source: str = "scripted",
        model_revision: str = "deterministic-fixture",
        monitor: MonitorCallable | None = None,
        fallback_on_contain: bool = False,
        external_event_sink: Callable[[EventRecord], None] | None = None,
        parser_version: str = GATEWAY_PARSER_V1_STRICT,
    ) -> None:
        if model_source not in {"scripted", "synthetic", "model"}:
            raise ContractValidationError(f"Unsupported model_source: {model_source!r}")
        self.scenario = scenario
        self.episode_id = episode_id
        self.generation_id = generation_id
        self.model_source = model_source
        self.model_revision = model_revision
        self._monitor = monitor
        self._fallback_on_contain = fallback_on_contain
        self._lock = threading.RLock()

        # Deep-isolated mutable session state
        self._state: dict[str, Any] = {
            "accounts": {
                acc_id: AccountRecord(
                    account_id=rec.account_id,
                    currency=rec.currency,
                    balance_minor=rec.balance_minor,
                    verified_iban=rec.verified_iban,
                    status=rec.status,
                    internal_notes=rec.internal_notes,
                )
                for acc_id, rec in scenario.accounts.items()
            },
            "tickets": {
                t_id: TicketRecord(
                    ticket_id=rec.ticket_id,
                    account_id=rec.account_id,
                    subject=rec.subject,
                    body=rec.body,
                    channel=rec.channel,
                )
                for t_id, rec in scenario.tickets.items()
            },
            "read_tickets": [],
            "read_accounts": [],
            "reply_drafts": [],
            "staged_refunds": [],
            "committed_refunds": [],
            "committed_operation_ids": set(),
            "committed_refund_total_minor": 0,
            "dispatched_webhooks": [],
        }
        self._state_version: int = 0
        self._step_counter: int = 0
        self._attempt_counter: int = 0
        self._commits_suspended: bool = False
        self._closed: bool = False
        self._termination_reason: str = "in_progress"
        self._episode_error: str | None = None
        self._delivered_observations: list[FrozenMap] = []

        self.gateway = DeterministicGateway(
            episode_id=episode_id,
            generation_id=generation_id,
            envelope=scenario.trusted_envelope,
            parser_version=parser_version,
        )
        self._tool_registry = HostToolRegistry(episode_id=episode_id)
        self.event_log = EpisodeEventLog(
            episode_id=episode_id,
            generation_id=generation_id,
            external_sink=external_event_sink,
        )
        self._initial_state_snapshot: FrozenMap = self.snapshot_state()
        self._initial_state_digest: str = sha256_digest(self._initial_state_snapshot)

    @property
    def state_version(self) -> int:
        with self._lock:
            return self._state_version

    @property
    def commits_suspended(self) -> bool:
        with self._lock:
            return self._commits_suspended

    def snapshot_state(self) -> FrozenMap:
        """Return a deep-frozen canonical snapshot of the current environment state."""
        with self._lock:
            accounts_map = {
                acc_id: {
                    "account_id": rec.account_id,
                    "currency": rec.currency,
                    "balance_minor": rec.balance_minor,
                    "verified_iban": rec.verified_iban,
                    "status": rec.status,
                }
                for acc_id, rec in sorted(self._state["accounts"].items())
            }
            tickets_map = {
                t_id: {
                    "ticket_id": rec.ticket_id,
                    "account_id": rec.account_id,
                    "subject": rec.subject,
                    "body": rec.body,
                    "channel": rec.channel,
                }
                for t_id, rec in sorted(self._state["tickets"].items())
            }
            return FrozenMap(
                {
                    "state_version": self._state_version,
                    "commits_suspended": self._commits_suspended,
                    "accounts": accounts_map,
                    "tickets": tickets_map,
                    "read_tickets": tuple(self._state["read_tickets"]),
                    "read_accounts": tuple(self._state["read_accounts"]),
                    "reply_drafts": tuple(self._state["reply_drafts"]),
                    "staged_refunds": tuple(self._state["staged_refunds"]),
                    "committed_refunds": tuple(self._state["committed_refunds"]),
                    "committed_operation_ids": tuple(sorted(self._state["committed_operation_ids"])),
                    "committed_refund_total_minor": self._state["committed_refund_total_minor"],
                    "dispatched_webhooks": tuple(self._state["dispatched_webhooks"]),
                }
            )

    def _clone_internal_mutable_state(self) -> tuple[dict[str, Any], int, int, int, bool]:
        return (
            {
                "accounts": dict(self._state["accounts"]),
                "tickets": dict(self._state["tickets"]),
                "read_tickets": list(self._state["read_tickets"]),
                "read_accounts": list(self._state["read_accounts"]),
                "reply_drafts": list(self._state["reply_drafts"]),
                "staged_refunds": list(self._state["staged_refunds"]),
                "committed_refunds": list(self._state["committed_refunds"]),
                "committed_operation_ids": set(self._state["committed_operation_ids"]),
                "committed_refund_total_minor": self._state["committed_refund_total_minor"],
                "dispatched_webhooks": list(self._state["dispatched_webhooks"]),
            },
            self._state_version,
            self._step_counter,
            self._attempt_counter,
            self._commits_suspended,
        )

    def _restore_internal_mutable_state(
        self, saved: tuple[dict[str, Any], int, int, int, bool]
    ) -> None:
        (
            self._state,
            self._state_version,
            self._step_counter,
            self._attempt_counter,
            self._commits_suspended,
        ) = saved

    def prepare_and_validate(
        self, raw_proposal: str | Mapping[str, Any]
    ) -> tuple[Attempt, GatewayDecision]:
        """Parse and validate a proposal against current state without executing it yet."""
        with self._lock:
            if self._closed:
                raise ContractValidationError("Cannot validate on a closed environment")
            next_idx = self._attempt_counter + 1
            attempt_id = f"{self.episode_id}-att-{next_idx:04d}"
            decision_id = f"{self.episode_id}-dec-{next_idx:04d}"
            attempt = self.gateway.parse_attempt(
                raw_proposal,
                attempt_id=attempt_id,
                step_index=self._step_counter,
            )
            decision = self.gateway.evaluate_attempt(
                attempt,
                decision_id=decision_id,
                state=self._state,
                state_version=self._state_version,
                commits_suspended=self._commits_suspended,
            )
            return attempt, decision

    def execute_prepared(
        self, attempt: Attempt, decision: GatewayDecision
    ) -> StepExecutionResult:
        """Atomically execute a previously prepared (attempt, decision) pair, revalidating if state changed."""
        with self._lock:
            if self._closed:
                raise ContractValidationError("Cannot execute on a closed environment")
            if decision.attempt_id != attempt.attempt_id:
                raise ContractValidationError("GatewayDecision.attempt_id does not match Attempt.attempt_id")
            if decision.payload_digest != attempt.payload_digest:
                raise ContractValidationError("Payload digest mismatch between Attempt and GatewayDecision")
            if decision.validated_payload is not None:
                if sha256_digest(decision.validated_payload) != decision.payload_digest:
                    raise ContractValidationError("Tampered validated_payload detected prior to execution")

            return self._commit_step_locked(attempt=attempt, preliminary_decision=decision)

    def step(self, raw_proposal: str | Mapping[str, Any]) -> StepExecutionResult:
        """Parse, validate, run optional monitor, execute, and record events atomically."""
        with self._lock:
            attempt, decision = self.prepare_and_validate(raw_proposal)
            return self._commit_step_locked(attempt=attempt, preliminary_decision=decision)

    def _commit_step_locked(
        self,
        *,
        attempt: Attempt,
        preliminary_decision: GatewayDecision,
    ) -> StepExecutionResult:
        saved_state = self._clone_internal_mutable_state()
        log_checkpoint = self.event_log.checkpoint()
        obs_len_before = len(self._delivered_observations)

        try:
            self._attempt_counter += 1
            step_index = self._step_counter
            self._step_counter += 1

            # 1. Always record Attempt BEFORE any rejection or execution
            self.event_log.record_attempt(attempt)

            # 2. If state_version changed since preliminary validation, atomically revalidate!
            decision = preliminary_decision
            if decision.state_version != self._state_version:
                decision = self.gateway.evaluate_attempt(
                    attempt,
                    decision_id=decision.decision_id,
                    state=self._state,
                    state_version=self._state_version,
                    commits_suspended=self._commits_suspended,
                )

            # 3. If gateway allowed the call and a runtime monitor is configured, evaluate monitor at t-
            if decision.allowed and self._monitor is not None and decision.validated_payload is not None:
                decision = self._evaluate_monitor_locked(
                    attempt=attempt,
                    gateway_decision=decision,
                    step_index=step_index,
                )

            # 4. Record GatewayDecision
            self.event_log.record_decision(decision)

            # 5. Execute validated immutable payload if allowed, else emit non-mutating receipt
            receipt_id = f"{self.episode_id}-rcp-{self._attempt_counter:04d}"
            version_before = self._state_version

            if decision.allowed and decision.validated_payload is not None:
                tool_name: str = decision.validated_payload["tool_name"]
                frozen_args: FrozenMap = decision.validated_payload["arguments"]
                if sha256_digest(decision.validated_payload) != decision.payload_digest:
                    raise ContractValidationError("Immutable payload digest mismatch before tool dispatch")

                outcome = self._tool_registry.execute_validated(
                    tool_name=tool_name,
                    frozen_arguments=frozen_args,
                    step_index=step_index,
                    state=self._state,
                )
                self._state_version += 1
                receipt = ExecutionReceipt(
                    receipt_id=receipt_id,
                    attempt_id=attempt.attempt_id,
                    parent_attempt_id=attempt.parent_attempt_id,
                    episode_id=self.episode_id,
                    generation_id=self.generation_id,
                    effective_tool_name=tool_name,
                    effective_payload=decision.validated_payload,
                    payload_digest=decision.payload_digest,
                    status=outcome.receipt_status,
                    operation_id=outcome.operation_id,
                    state_delta=outcome.state_delta,
                    state_version_before=version_before,
                    state_version_after=self._state_version,
                    output=outcome.output,
                    error_code=None,
                )
            else:
                receipt = ExecutionReceipt(
                    receipt_id=receipt_id,
                    attempt_id=attempt.attempt_id,
                    parent_attempt_id=attempt.parent_attempt_id,
                    episode_id=self.episode_id,
                    generation_id=self.generation_id,
                    effective_tool_name=attempt.parsed_tool_name,
                    effective_payload=decision.validated_payload,
                    payload_digest=decision.payload_digest,
                    status=decision.status,  # "denied" | "contained" | "error"
                    operation_id=None,
                    state_delta=FrozenMap({}),
                    state_version_before=version_before,
                    state_version_after=version_before,
                    output=FrozenMap(
                        {
                            "allowed": False,
                            "status": decision.status,
                            "reason_code": decision.reason_code,
                        }
                    ),
                    error_code=decision.reason_code,
                )

            self.event_log.record_receipt(receipt)

            # 6. Handle optional fallback child call when execute_instant_refund is contained by monitor
            fb_attempt: Attempt | None = None
            fb_decision: GatewayDecision | None = None
            fb_receipt: ExecutionReceipt | None = None
            if (
                decision.status == "contained"
                and self._fallback_on_contain
                and attempt.parsed_tool_name == "execute_instant_refund"
                and attempt.parsed_arguments is not None
            ):
                fb_attempt, fb_decision, fb_receipt = self._execute_fallback_child_locked(
                    parent_attempt=attempt,
                    step_index=step_index,
                )

            obs = FrozenMap(
                {
                    "step_index": step_index,
                    "attempt_id": attempt.attempt_id,
                    "receipt_id": receipt.receipt_id,
                    "status": receipt.status,
                    "reason_code": decision.reason_code,
                    "output": receipt.output,
                    "fallback_receipt_id": fb_receipt.receipt_id if fb_receipt else None,
                    "fallback_status": fb_receipt.status if fb_receipt else None,
                }
            )
            self._delivered_observations.append(obs)

            return StepExecutionResult(
                attempt=attempt,
                decision=decision,
                receipt=receipt,
                observation_for_actor=obs,
                fallback_attempt=fb_attempt,
                fallback_decision=fb_decision,
                fallback_receipt=fb_receipt,
            )
        except EventSinkError as exc:
            self._restore_internal_mutable_state(saved_state)
            del self._delivered_observations[obs_len_before:]
            self.event_log.rollback(log_checkpoint, error_message=str(exc))
            self._termination_reason = "event_sink_error"
            self._episode_error = str(exc)
            self._closed = True
            raise

    def _evaluate_monitor_locked(
        self,
        *,
        attempt: Attempt,
        gateway_decision: GatewayDecision,
        step_index: int,
    ) -> GatewayDecision:
        assert self._monitor is not None
        assert gateway_decision.validated_payload is not None
        pre_obs = PreActionObservation(
            episode_id=self.episode_id,
            generation_id=self.generation_id,
            step_index=step_index,
            user_task=self.scenario.case_spec.user_task,
            public_policy=self.scenario.case_spec.public_policy,
            history=tuple(self._delivered_observations),
            tool_name=gateway_decision.validated_payload["tool_name"],
            frozen_arguments=gateway_decision.validated_payload["arguments"],
            observed_metadata=FrozenMap({"state_version": self._state_version}),
        )
        try:
            raw_verdict = self._monitor(pre_obs)
            if not isinstance(raw_verdict, Mapping):
                raise TypeError("Monitor verdict must be a Mapping")
            score = raw_verdict.get("score", 0.0)
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
                raise ValueError(f"Non-finite monitor score: {score!r}")
            contain = bool(raw_verdict.get("contain", False))
            frozen_verdict = FrozenMap(freeze_value(raw_verdict))
        except Exception as exc:
            # Monitor failure suspends commits and produces explicit error containment; never silent success
            self._commits_suspended = True
            self._termination_reason = "monitor_error"
            self._episode_error = f"{type(exc).__name__}: {exc}"
            return GatewayDecision(
                decision_id=gateway_decision.decision_id,
                attempt_id=attempt.attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                allowed=False,
                status="error",
                reason_code="MONITOR_ERROR",
                validated_payload=None,
                payload_digest=gateway_decision.payload_digest,
                state_version=self._state_version,
                applied_limits=gateway_decision.applied_limits,
                monitor_verdict=FrozenMap({"error": f"{type(exc).__name__}: {exc}"}),
            )

        if contain:
            self._commits_suspended = True
            if gateway_decision.validated_payload["tool_name"] in COMMIT_TOOL_NAMES:
                return GatewayDecision(
                    decision_id=gateway_decision.decision_id,
                    attempt_id=attempt.attempt_id,
                    episode_id=self.episode_id,
                    generation_id=self.generation_id,
                    allowed=False,
                    status="contained",
                    reason_code="MONITOR_CONTAINED",
                    validated_payload=None,
                    payload_digest=gateway_decision.payload_digest,
                    state_version=self._state_version,
                    applied_limits=gateway_decision.applied_limits,
                    monitor_verdict=frozen_verdict,
                )

        return GatewayDecision(
            decision_id=gateway_decision.decision_id,
            attempt_id=attempt.attempt_id,
            episode_id=self.episode_id,
            generation_id=self.generation_id,
            allowed=True,
            status="allowed",
            reason_code="OK",
            validated_payload=gateway_decision.validated_payload,
            payload_digest=gateway_decision.payload_digest,
            state_version=self._state_version,
            applied_limits=gateway_decision.applied_limits,
            monitor_verdict=frozen_verdict,
        )

    def _execute_fallback_child_locked(
        self,
        *,
        parent_attempt: Attempt,
        step_index: int,
    ) -> tuple[Attempt, GatewayDecision, ExecutionReceipt]:
        assert parent_attempt.parsed_arguments is not None
        self._attempt_counter += 1
        child_attempt_id = f"{self.episode_id}-att-{self._attempt_counter:04d}"
        child_decision_id = f"{self.episode_id}-dec-{self._attempt_counter:04d}"
        child_receipt_id = f"{self.episode_id}-rcp-{self._attempt_counter:04d}"

        parent_args = parent_attempt.parsed_arguments
        fallback_proposal = {
            "tool_name": "stage_refund_proposal",
            "arguments": {
                "account_id": parent_args.get("account_id", ""),
                "amount_minor": parent_args.get("amount_minor", 0),
                "currency": parent_args.get("currency", "USD"),
                "destination_iban": parent_args.get("destination_iban", ""),
                "reason": f"Fallback staged proposal from contained attempt {parent_attempt.attempt_id}",
            },
        }
        child_attempt = self.gateway.parse_attempt(
            fallback_proposal,
            attempt_id=child_attempt_id,
            step_index=step_index,
            parent_attempt_id=parent_attempt.attempt_id,
        )
        self.event_log.record_attempt(child_attempt)

        # Child fallback call does NOT inherit permission from execute_instant_refund
        child_decision = self.gateway.evaluate_attempt(
            child_attempt,
            decision_id=child_decision_id,
            state=self._state,
            state_version=self._state_version,
            commits_suspended=self._commits_suspended,
            is_fallback=True,
        )
        self.event_log.record_decision(child_decision)

        version_before = self._state_version
        if child_decision.allowed and child_decision.validated_payload is not None:
            outcome = self._tool_registry.execute_validated(
                tool_name="stage_refund_proposal",
                frozen_arguments=child_decision.validated_payload["arguments"],
                step_index=step_index,
                state=self._state,
            )
            self._state_version += 1
            child_receipt = ExecutionReceipt(
                receipt_id=child_receipt_id,
                attempt_id=child_attempt.attempt_id,
                parent_attempt_id=parent_attempt.attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                effective_tool_name="stage_refund_proposal",
                effective_payload=child_decision.validated_payload,
                payload_digest=child_decision.payload_digest,
                status=outcome.receipt_status,
                operation_id=outcome.operation_id,
                state_delta=outcome.state_delta,
                state_version_before=version_before,
                state_version_after=self._state_version,
                output=outcome.output,
                error_code=None,
            )
        else:
            child_receipt = ExecutionReceipt(
                receipt_id=child_receipt_id,
                attempt_id=child_attempt.attempt_id,
                parent_attempt_id=parent_attempt.attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                effective_tool_name="stage_refund_proposal",
                effective_payload=None,
                payload_digest=child_decision.payload_digest,
                status="denied",
                operation_id=None,
                state_delta=FrozenMap({}),
                state_version_before=version_before,
                state_version_after=version_before,
                output=FrozenMap(
                    {
                        "allowed": False,
                        "status": "denied",
                        "reason_code": child_decision.reason_code,
                    }
                ),
                error_code=child_decision.reason_code,
            )

        self.event_log.record_receipt(child_receipt)
        return child_attempt, child_decision, child_receipt

    def close(
        self,
        *,
        termination_reason: str = "completed",
        error: str | None = None,
    ) -> EpisodeResult:
        """Seal the episode, validate 1-to-1 joins and digests, and return EpisodeResult."""
        with self._lock:
            if not self._closed:
                if self._termination_reason == "in_progress":
                    self._termination_reason = termination_reason
                if error is not None and self._episode_error is None:
                    self._episode_error = error
                self._closed = True

            final_snapshot = self.snapshot_state()
            final_digest = sha256_digest(final_snapshot)
            attempts = self.event_log.attempts
            decisions = self.event_log.decisions
            receipts = self.event_log.receipts
            chain_digest = compute_episode_chain_digest(
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                case_id=self.scenario.case_spec.case_id,
                initial_state_digest=self._initial_state_digest,
                attempts=attempts,
                decisions=decisions,
                receipts=receipts,
                final_state_digest=final_digest,
            )
            coverage = self.event_log.coverage_summary()
            is_valid = (
                bool(coverage["complete_joins"])
                and not self.event_log.sink_failed
                and self._termination_reason not in {"event_sink_error", "monitor_error"}
                and not self._termination_reason.startswith("actor_exception:")
                and self._episode_error is None
            )
            closed_loop = (
                is_valid
                and self.model_source == "model"
                and len(self._delivered_observations) > 0
            )
            return EpisodeResult(
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                case_id=self.scenario.case_spec.case_id,
                model_source=self.model_source,
                model_revision=self.model_revision,
                termination_reason=self._termination_reason,
                steps_executed=self._step_counter,
                delivered_observations=tuple(self._delivered_observations),
                attempts=attempts,
                decisions=decisions,
                receipts=receipts,
                initial_state_snapshot=self._initial_state_snapshot,
                final_state_snapshot=final_snapshot,
                initial_state_digest=self._initial_state_digest,
                final_state_digest=final_digest,
                chain_digest=chain_digest,
                valid=is_valid,
                error=self._episode_error,
                closed_loop_verified=closed_loop,
            )


class EnvironmentFactory:
    """Factory creating isolated SyntheticEnvironment instances per episode and generation."""

    def __init__(self, *, run_prefix: str = "aegis") -> None:
        self._run_prefix = run_prefix
        self._counter = 0
        self._lock = threading.Lock()

    def create(
        self,
        scenario: ScenarioBundle,
        *,
        episode_id: str | None = None,
        generation_id: str | None = None,
        model_source: str = "scripted",
        model_revision: str = "deterministic-fixture",
        monitor: MonitorCallable | None = None,
        fallback_on_contain: bool = False,
        external_event_sink: Callable[[EventRecord], None] | None = None,
        parser_version: str = GATEWAY_PARSER_V1_STRICT,
    ) -> SyntheticEnvironment:
        with self._lock:
            self._counter += 1
            seq = self._counter
        uid = uuid.uuid4().hex[:8]
        ep_id = episode_id or f"{self._run_prefix}-ep-{seq:04d}-{uid}"
        gen_id = generation_id or f"{self._run_prefix}-gen-{seq:04d}-{uid}"
        return SyntheticEnvironment(
            scenario=scenario,
            episode_id=ep_id,
            generation_id=gen_id,
            model_source=model_source,
            model_revision=model_revision,
            monitor=monitor,
            fallback_on_contain=fallback_on_contain,
            external_event_sink=external_event_sink,
            parser_version=parser_version,
        )

