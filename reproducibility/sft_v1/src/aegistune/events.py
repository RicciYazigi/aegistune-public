"""Deterministic event logging, coverage tracking, and failure-safe event sinks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .contracts import (
    Attempt,
    ExecutionReceipt,
    FrozenMap,
    GatewayDecision,
    canonical_json,
    freeze_value,
    sha256_digest,
)


class EventSinkError(RuntimeError):
    """Raised when an event cannot be durably recorded in the session event log."""


@dataclass(frozen=True)
class EventRecord:
    """Single immutable event entry with deterministic sequence index and digest."""

    sequence_index: int
    episode_id: str
    generation_id: str
    event_type: str  # "attempt" | "decision" | "receipt" | "monitor" | "error"
    attempt_id: str
    payload: FrozenMap
    event_digest: str


class EpisodeEventLog:
    """Per-episode event log that records attempts before rejection and tracks coverage."""

    def __init__(
        self,
        *,
        episode_id: str,
        generation_id: str,
        external_sink: Callable[[EventRecord], None] | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.generation_id = generation_id
        self._external_sink = external_sink
        self._records: list[EventRecord] = []
        self._attempts: list[Attempt] = []
        self._decisions: list[GatewayDecision] = []
        self._receipts: list[ExecutionReceipt] = []
        self._sink_failed: bool = False
        self._sink_error_message: str | None = None

    @property
    def sink_failed(self) -> bool:
        return self._sink_failed

    @property
    def sink_error_message(self) -> str | None:
        return self._sink_error_message

    @property
    def records(self) -> tuple[EventRecord, ...]:
        return tuple(self._records)

    @property
    def attempts(self) -> tuple[Attempt, ...]:
        return tuple(self._attempts)

    @property
    def decisions(self) -> tuple[GatewayDecision, ...]:
        return tuple(self._decisions)

    @property
    def receipts(self) -> tuple[ExecutionReceipt, ...]:
        return tuple(self._receipts)

    def checkpoint(self) -> tuple[int, int, int, int]:
        """Return current lengths so a failed atomic transition can roll back cleanly."""
        return (
            len(self._records),
            len(self._attempts),
            len(self._decisions),
            len(self._receipts),
        )

    def rollback(self, checkpoint: tuple[int, int, int, int], *, error_message: str) -> None:
        """Roll back partial transition entries and mark sink failure."""
        rec_len, att_len, dec_len, rcp_len = checkpoint
        del self._records[rec_len:]
        del self._attempts[att_len:]
        del self._decisions[dec_len:]
        del self._receipts[rcp_len:]
        self._sink_failed = True
        self._sink_error_message = error_message

    def _append_record(self, event_type: str, attempt_id: str, payload_dict: dict[str, Any]) -> EventRecord:
        if self._sink_failed:
            raise EventSinkError(
                f"EpisodeEventLog is in failed state: {self._sink_error_message}"
            )
        frozen_payload = FrozenMap(freeze_value(payload_dict))
        seq = len(self._records)
        digest = sha256_digest(
            {
                "sequence_index": seq,
                "episode_id": self.episode_id,
                "generation_id": self.generation_id,
                "event_type": event_type,
                "attempt_id": attempt_id,
                "payload": frozen_payload,
            }
        )
        record = EventRecord(
            sequence_index=seq,
            episode_id=self.episode_id,
            generation_id=self.generation_id,
            event_type=event_type,
            attempt_id=attempt_id,
            payload=frozen_payload,
            event_digest=digest,
        )
        if self._external_sink is not None:
            try:
                self._external_sink(record)
            except Exception as exc:
                self._sink_failed = True
                self._sink_error_message = f"{type(exc).__name__}: {exc}"
                raise EventSinkError(self._sink_error_message) from exc
        self._records.append(record)
        return record

    def record_attempt(self, attempt: Attempt) -> EventRecord:
        record = self._append_record(
            "attempt",
            attempt.attempt_id,
            {
                "attempt_id": attempt.attempt_id,
                "parent_attempt_id": attempt.parent_attempt_id,
                "step_index": attempt.step_index,
                "raw_model_output": attempt.raw_model_output,
                "parse_status": attempt.parse_status,
                "parsed_tool_name": attempt.parsed_tool_name,
                "parsed_arguments": attempt.parsed_arguments,
                "parse_error": attempt.parse_error,
                "payload_digest": attempt.payload_digest,
            },
        )
        self._attempts.append(attempt)
        return record

    def record_decision(self, decision: GatewayDecision) -> EventRecord:
        record = self._append_record(
            "decision",
            decision.attempt_id,
            {
                "decision_id": decision.decision_id,
                "attempt_id": decision.attempt_id,
                "allowed": decision.allowed,
                "status": decision.status,
                "reason_code": decision.reason_code,
                "validated_payload": decision.validated_payload,
                "payload_digest": decision.payload_digest,
                "state_version": decision.state_version,
                "applied_limits": decision.applied_limits,
                "monitor_verdict": decision.monitor_verdict,
            },
        )
        self._decisions.append(decision)
        return record

    def record_receipt(self, receipt: ExecutionReceipt) -> EventRecord:
        record = self._append_record(
            "receipt",
            receipt.attempt_id,
            {
                "receipt_id": receipt.receipt_id,
                "attempt_id": receipt.attempt_id,
                "parent_attempt_id": receipt.parent_attempt_id,
                "effective_tool_name": receipt.effective_tool_name,
                "effective_payload": receipt.effective_payload,
                "payload_digest": receipt.payload_digest,
                "status": receipt.status,
                "operation_id": receipt.operation_id,
                "state_delta": receipt.state_delta,
                "state_version_before": receipt.state_version_before,
                "state_version_after": receipt.state_version_after,
                "output": receipt.output,
                "error_code": receipt.error_code,
            },
        )
        self._receipts.append(receipt)
        return record

    def coverage_summary(self) -> FrozenMap:
        """Compute coverage and completeness metrics for the episode event log."""
        total_attempts = len(self._attempts)
        total_decisions = len(self._decisions)
        total_receipts = len(self._receipts)
        complete_joins = (
            not self._sink_failed
            and total_attempts == total_decisions == total_receipts
            and all(
                a.attempt_id == d.attempt_id == r.attempt_id
                for a, d, r in zip(self._attempts, self._decisions, self._receipts)
            )
        )
        return FrozenMap(
            {
                "total_attempts": total_attempts,
                "total_decisions": total_decisions,
                "total_receipts": total_receipts,
                "sink_failed": self._sink_failed,
                "sink_error_message": self._sink_error_message,
                "complete_joins": complete_joins,
            }
        )

    def to_jsonl(self) -> str:
        """Serialize all recorded events deterministically to JSON Lines."""
        lines = [
            canonical_json(
                {
                    "sequence_index": r.sequence_index,
                    "episode_id": r.episode_id,
                    "generation_id": r.generation_id,
                    "event_type": r.event_type,
                    "attempt_id": r.attempt_id,
                    "payload": r.payload,
                    "event_digest": r.event_digest,
                }
            )
            for r in self._records
        ]
        return "\n".join(lines)

