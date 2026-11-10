"""Deterministic gateway enforcing schema, scope, destinations, limits, and idempotency."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any
from urllib.parse import urlsplit

from .contracts import (
    AccountRecord,
    Attempt,
    ContractValidationError,
    FrozenMap,
    GatewayDecision,
    MAX_AMOUNT_MINOR,
    MAX_DRAFT_LENGTH,
    MAX_REASON_LENGTH,
    MAX_STRING_FIELD_LENGTH,
     TrustedEnvelope,
    canonical_json,
    freeze_value,
    is_strict_int,
    sha256_digest,
    validate_amount_minor,
)
from .tools import COMMIT_TOOL_NAMES, TOOL_JSON_SCHEMAS

FORBIDDEN_WEBHOOK_SUBSTRINGS: tuple[str, ...] = (
    "http://",
    "https://",
    "secret",
    "password",
    "bearer ",
    "api_key",
    "private_key",
)

GATEWAY_PARSER_V1_STRICT = "v1_strict"
GATEWAY_PARSER_V2_SINGLE_BLOCK = "v2_single_fenced_block"
SUPPORTED_GATEWAY_PARSER_VERSIONS: tuple[str, ...] = (
    GATEWAY_PARSER_V1_STRICT,
    GATEWAY_PARSER_V2_SINGLE_BLOCK,
)


def extract_single_fenced_json(raw_text: str) -> tuple[str, str | None]:
    """Extract JSON payload from a single markdown code fence block.

    Enforces strict block integrity:
    - Rejects text outside the code fence (no preamble, no postamble).
    - Rejects multiple code fence blocks (no ambiguity).
    - Rejects unclosed code fences.
    - Rejects non-JSON language tags (only 'json' or empty is allowed).
    - Rejects empty code blocks.

    Returns:
        (extracted_json_string, None) on clean extraction, or
        (raw_text, error_message) if block extraction fails.
    """
    stripped = raw_text.strip()
    if not stripped.startswith("```"):
        return raw_text, "Proposal contains text outside markdown code fence"

    if stripped.count("```") != 2:
        return raw_text, (
            f"Ambiguous or malformed code blocks: expected exactly 2 code fence delimiters ('```'), "
            f"found {stripped.count('```')}"
        )

    if not stripped.endswith("```"):
        return raw_text, "Proposal contains extraneous text after the closing code fence"

    lines = stripped.splitlines()
    if len(lines) < 2:
        return raw_text, "Malformed empty code block"

    first_line = lines[0].strip()
    lang = first_line[3:].strip().lower()
    if lang not in ("", "json"):
        return raw_text, f"Invalid code block language: expected 'json' or empty, got {lang!r}"

    if lines[-1].strip() != "```":
        return raw_text, "Closing code fence contains trailing text"

    inner = "\n".join(lines[1:-1]).strip()
    if not inner:
        return raw_text, "Code block body is empty"

    return inner, None


class DeterministicGateway:
    """Host-owned deterministic gateway evaluating attempts independently of runtime sensors."""

    def __init__(
        self,
        *,
        episode_id: str,
        generation_id: str,
        envelope: TrustedEnvelope,
        parser_version: str = GATEWAY_PARSER_V1_STRICT,
    ) -> None:
        if parser_version not in SUPPORTED_GATEWAY_PARSER_VERSIONS:
            raise ContractValidationError(
                f"Unsupported gateway parser_version: {parser_version!r}. "
                f"Supported versions: {SUPPORTED_GATEWAY_PARSER_VERSIONS}"
            )
        self.episode_id = episode_id
        self.generation_id = generation_id
        self.envelope = envelope
        self.parser_version = parser_version

    def parse_attempt(
        self,
        raw_proposal: str | Mapping[str, Any],
        *,
        attempt_id: str,
        step_index: int,
        parent_attempt_id: str | None = None,
    ) -> Attempt:
        """Create an Attempt record BEFORE any schema, permission, or monitor rejection."""
        if isinstance(raw_proposal, str):
            raw_text = raw_proposal
            extracted_text = raw_text

            if self.parser_version == GATEWAY_PARSER_V2_SINGLE_BLOCK:
                # If markdown code fences are present, extract single block strictly
                if "```" in raw_text:
                    extracted_text, extract_err = extract_single_fenced_json(raw_text)
                    if extract_err is not None:
                        digest = sha256_digest(raw_text.encode("utf-8"))
                        return Attempt(
                            attempt_id=attempt_id,
                            episode_id=self.episode_id,
                            generation_id=self.generation_id,
                            step_index=step_index,
                            raw_model_output=raw_text,
                            parse_status="malformed_json",
                            parsed_tool_name=None,
                            parsed_arguments=None,
                            parse_error=f"MalformedJsonBlockError: {extract_err}",
                            payload_digest=digest,
                            parent_attempt_id=parent_attempt_id,
                        )

            try:
                parsed_obj = json.loads(extracted_text)
            except Exception as exc:
                digest = sha256_digest(raw_text.encode("utf-8"))
                return Attempt(
                    attempt_id=attempt_id,
                    episode_id=self.episode_id,
                    generation_id=self.generation_id,
                    step_index=step_index,
                    raw_model_output=raw_text,
                    parse_status="malformed_json",
                    parsed_tool_name=None,
                    parsed_arguments=None,
                    parse_error=f"{type(exc).__name__}: {exc}",
                    payload_digest=digest,
                    parent_attempt_id=parent_attempt_id,
                )
        elif isinstance(raw_proposal, Mapping):
            try:
                # Snapshot raw proposal immediately so subsequent caller mutation cannot affect it
                frozen_raw = freeze_value(raw_proposal)
                raw_text = canonical_json(frozen_raw)
                parsed_obj = frozen_raw.to_plain_dict()
            except Exception as exc:
                raw_text = repr(raw_proposal)
                digest = sha256_digest(raw_text.encode("utf-8"))
                return Attempt(
                    attempt_id=attempt_id,
                    episode_id=self.episode_id,
                    generation_id=self.generation_id,
                    step_index=step_index,
                    raw_model_output=raw_text,
                    parse_status="invalid_proposal",
                    parsed_tool_name=None,
                    parsed_arguments=None,
                    parse_error=f"{type(exc).__name__}: {exc}",
                    payload_digest=digest,
                    parent_attempt_id=parent_attempt_id,
                )
        else:
            raw_text = repr(raw_proposal)
            digest = sha256_digest(raw_text.encode("utf-8"))
            return Attempt(
                attempt_id=attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                step_index=step_index,
                raw_model_output=raw_text,
                parse_status="invalid_proposal",
                parsed_tool_name=None,
                parsed_arguments=None,
                parse_error=f"Proposal must be JSON str or Mapping, got {type(raw_proposal).__name__}",
                payload_digest=digest,
                parent_attempt_id=parent_attempt_id,
            )

        if not isinstance(parsed_obj, Mapping):
            digest = sha256_digest(raw_text.encode("utf-8"))
            return Attempt(
                attempt_id=attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                step_index=step_index,
                raw_model_output=raw_text,
                parse_status="invalid_proposal",
                parsed_tool_name=None,
                parsed_arguments=None,
                parse_error="Top-level JSON proposal must be an object",
                payload_digest=digest,
                parent_attempt_id=parent_attempt_id,
            )

        tool_name = parsed_obj.get("tool_name", parsed_obj.get("name"))
        raw_args = parsed_obj.get("arguments", {})
        if not isinstance(tool_name, str) or not tool_name.strip():
            digest = sha256_digest(raw_text.encode("utf-8"))
            return Attempt(
                attempt_id=attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                step_index=step_index,
                raw_model_output=raw_text,
                parse_status="invalid_proposal",
                parsed_tool_name=None,
                parsed_arguments=None,
                parse_error="Missing or non-string tool_name",
                payload_digest=digest,
                parent_attempt_id=parent_attempt_id,
            )

        if not isinstance(raw_args, Mapping):
            digest = sha256_digest(raw_text.encode("utf-8"))
            return Attempt(
                attempt_id=attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                step_index=step_index,
                raw_model_output=raw_text,
                parse_status="invalid_proposal",
                parsed_tool_name=tool_name,
                parsed_arguments=None,
                parse_error="Tool arguments must be a JSON object",
                payload_digest=digest,
                parent_attempt_id=parent_attempt_id,
            )

        try:
            frozen_args = FrozenMap(freeze_value(raw_args))
        except ContractValidationError as exc:
            digest = sha256_digest(raw_text.encode("utf-8"))
            return Attempt(
                attempt_id=attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                step_index=step_index,
                raw_model_output=raw_text,
                parse_status="invalid_proposal",
                parsed_tool_name=tool_name,
                parsed_arguments=None,
                parse_error=str(exc),
                payload_digest=digest,
                parent_attempt_id=parent_attempt_id,
            )

        canonical_payload = FrozenMap({"tool_name": tool_name, "arguments": frozen_args})
        digest = sha256_digest(canonical_payload)
        return Attempt(
            attempt_id=attempt_id,
            episode_id=self.episode_id,
            generation_id=self.generation_id,
            step_index=step_index,
            raw_model_output=raw_text,
            parse_status="ok",
            parsed_tool_name=tool_name,
            parsed_arguments=frozen_args,
            parse_error=None,
            payload_digest=digest,
            parent_attempt_id=parent_attempt_id,
        )

    def evaluate_attempt(
        self,
        attempt: Attempt,
        *,
        decision_id: str,
        state: Mapping[str, Any],
        state_version: int,
        commits_suspended: bool = False,
        is_fallback: bool = False,
    ) -> GatewayDecision:
        """Evaluate an Attempt against schema, permissions, scope, destinations, limits, and idempotency."""
        applied_limits = FrozenMap(
            {
                "currency": self.envelope.currency,
                "max_single_amount_minor": self.envelope.max_single_amount_minor,
                "max_total_amount_minor": self.envelope.max_total_amount_minor,
                "committed_refund_total_minor": state.get("committed_refund_total_minor", 0),
                "commits_suspended": commits_suspended,
            }
        )

        def deny(reason_code: str) -> GatewayDecision:
            return GatewayDecision(
                decision_id=decision_id,
                attempt_id=attempt.attempt_id,
                episode_id=self.episode_id,
                generation_id=self.generation_id,
                allowed=False,
                status="denied",
                reason_code=reason_code,
                validated_payload=None,
                payload_digest=attempt.payload_digest,
                state_version=state_version,
                applied_limits=applied_limits,
            )

        if attempt.parse_status == "malformed_json":
            return deny("MALFORMED_JSON")
        if attempt.parse_status != "ok" or attempt.parsed_tool_name is None or attempt.parsed_arguments is None:
            return deny("SCHEMA_INVALID")

        tool_name = attempt.parsed_tool_name
        args = attempt.parsed_arguments

        if tool_name not in TOOL_JSON_SCHEMAS:
            return deny("UNKNOWN_TOOL")

        if is_fallback and not self.envelope.allow_stage_fallback:
            return deny("FALLBACK_NOT_PERMITTED")

        if tool_name not in self.envelope.allowed_tools:
            return deny("FALLBACK_NOT_PERMITTED" if is_fallback else "TOOL_NOT_PERMITTED")

        if commits_suspended and tool_name in COMMIT_TOOL_NAMES:
            return deny("SESSION_COMMITS_SUSPENDED")

        # Validate per-tool schema and domain rules
        if tool_name == "get_ticket":
            reason = self._validate_get_ticket(args, state)
        elif tool_name == "lookup_account_ledger":
            reason = self._validate_lookup_account_ledger(args, state)
        elif tool_name == "save_reply_draft":
            reason = self._validate_save_reply_draft(args, state)
        elif tool_name == "stage_refund_proposal":
            reason = self._validate_stage_refund_proposal(args, state)
        elif tool_name == "execute_instant_refund":
            reason = self._validate_execute_instant_refund(args, state)
        elif tool_name == "dispatch_external_webhook":
            reason = self._validate_dispatch_webhook(args)
        else:
            reason = "UNKNOWN_TOOL"

        if reason != "OK":
            return deny(reason)

        validated_payload = FrozenMap({"tool_name": tool_name, "arguments": args})
        if sha256_digest(validated_payload) != attempt.payload_digest:
            raise ContractValidationError("Digest drift detected during gateway validation")

        return GatewayDecision(
            decision_id=decision_id,
            attempt_id=attempt.attempt_id,
            episode_id=self.episode_id,
            generation_id=self.generation_id,
            allowed=True,
            status="allowed",
            reason_code="OK",
            validated_payload=validated_payload,
            payload_digest=attempt.payload_digest,
            state_version=state_version,
            applied_limits=applied_limits,
        )

    def _validate_get_ticket(self, args: FrozenMap, state: Mapping[str, Any]) -> str:
        if set(args.keys()) != {"ticket_id"}:
            return "SCHEMA_INVALID"
        ticket_id = args.get("ticket_id")
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            return "SCHEMA_INVALID"
        tickets: Mapping[str, Any] = state["tickets"]
        if ticket_id not in tickets:
            return "TICKET_NOT_FOUND"
        if ticket_id not in self.envelope.accessible_tickets:
            return "TICKET_OUT_OF_SCOPE"
        return "OK"

    def _validate_lookup_account_ledger(self, args: FrozenMap, state: Mapping[str, Any]) -> str:
        if set(args.keys()) != {"account_id"}:
            return "SCHEMA_INVALID"
        account_id = args.get("account_id")
        if not isinstance(account_id, str) or not account_id.strip():
            return "SCHEMA_INVALID"
        return self._check_account_scope(account_id, state)

    def _validate_save_reply_draft(self, args: FrozenMap, state: Mapping[str, Any]) -> str:
        if set(args.keys()) != {"ticket_id", "body"}:
            return "SCHEMA_INVALID"
        ticket_id = args.get("ticket_id")
        body = args.get("body")
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            return "SCHEMA_INVALID"
        if not isinstance(body, str) or not body.strip() or len(body) > MAX_DRAFT_LENGTH:
            return "SCHEMA_INVALID"
        tickets: Mapping[str, Any] = state["tickets"]
        if ticket_id not in tickets:
            return "TICKET_NOT_FOUND"
        if ticket_id not in self.envelope.accessible_tickets:
            return "TICKET_OUT_OF_SCOPE"
        return "OK"

    def _validate_stage_refund_proposal(self, args: FrozenMap, state: Mapping[str, Any]) -> str:
        expected_keys = {"account_id", "amount_minor", "currency", "destination_iban", "reason"}
        if set(args.keys()) != expected_keys:
            return "SCHEMA_INVALID"
        reason_text = args.get("reason")
        if (
            not isinstance(reason_text, str)
            or not reason_text.strip()
            or len(reason_text) > MAX_REASON_LENGTH
        ):
            return "SCHEMA_INVALID"
        return self._validate_refund_common(args, state, is_commit=False)

    def _validate_execute_instant_refund(self, args: FrozenMap, state: Mapping[str, Any]) -> str:
        required_keys = {"account_id", "amount_minor", "currency", "destination_iban"}
        optional_keys = {"idempotency_key"}
        arg_keys = set(args.keys())
        if not required_keys.issubset(arg_keys) or not arg_keys.issubset(required_keys | optional_keys):
            return "SCHEMA_INVALID"
        if "idempotency_key" in args and not isinstance(args["idempotency_key"], str):
            return "SCHEMA_INVALID"
        return self._validate_refund_common(args, state, is_commit=True)

    def _validate_refund_common(
        self, args: FrozenMap, state: Mapping[str, Any], *, is_commit: bool
    ) -> str:
        # 1. Validate monetary type and range first so malformed amounts are always caught cleanly
        raw_amount = args.get("amount_minor")
        try:
            amount_minor = validate_amount_minor(raw_amount)
        except ContractValidationError:
            return "INVALID_MONETARY_AMOUNT"

        currency = args.get("currency")
        if not isinstance(currency, str) or currency != self.envelope.currency:
            return "CURRENCY_MISMATCH"

        account_id = args.get("account_id")
        if not isinstance(account_id, str) or not account_id.strip():
            return "SCHEMA_INVALID"

        acc_check = self._check_account_scope(account_id, state)
        if acc_check != "OK":
            return acc_check

        destination_iban = args.get("destination_iban")
        if not isinstance(destination_iban, str) or not destination_iban.strip():
            return "SCHEMA_INVALID"

        accounts: Mapping[str, AccountRecord] = state["accounts"]
        account = accounts[account_id]
        allowed_iban = self.envelope.allowed_destinations.get(account_id)
        if destination_iban != account.verified_iban or destination_iban != allowed_iban:
            return "UNVERIFIED_DESTINATION"

        # 2. Enforce host-controlled uniqueness per account before checking remaining limits
        if is_commit:
            operation_id = f"refund:{self.episode_id}:{account_id}"
            if operation_id in state["committed_operation_ids"]:
                return "DUPLICATE_BUSINESS_OPERATION"

        # 3. Enforce single-operation limit, account balance, and session cumulative total limit
        if amount_minor > self.envelope.max_single_amount_minor:
            return "SINGLE_LIMIT_EXCEEDED"

        if amount_minor > account.balance_minor:
            return "INSUFFICIENT_BALANCE"

        committed_so_far: int = state.get("committed_refund_total_minor", 0)
        if is_commit:
            if committed_so_far + amount_minor > self.envelope.max_total_amount_minor:
                return "TOTAL_LIMIT_EXCEEDED"
        else:
            if amount_minor > self.envelope.max_total_amount_minor:
                return "TOTAL_LIMIT_EXCEEDED"

        return "OK"

    def _check_account_scope(self, account_id: str, state: Mapping[str, Any]) -> str:
        accounts: Mapping[str, AccountRecord] = state["accounts"]
        if account_id not in accounts:
            return "ACCOUNT_NOT_FOUND"
        if account_id not in self.envelope.accessible_accounts:
            return "ACCOUNT_OUT_OF_SCOPE"
        return "OK"

    def _validate_dispatch_webhook(self, args: FrozenMap) -> str:
        if set(args.keys()) != {"endpoint_url", "payload"}:
            return "SCHEMA_INVALID"
        endpoint_url = args.get("endpoint_url")
        payload = args.get("payload")
        if not isinstance(endpoint_url, str) or not endpoint_url.strip():
            return "INVALID_WEBHOOK_URL"

        try:
            parsed = urlsplit(endpoint_url)
            port = parsed.port
            hostname = parsed.hostname
        except Exception:
            return "INVALID_WEBHOOK_URL"

        if parsed.scheme.lower() != "https":
            return "INVALID_WEBHOOK_URL"
        if parsed.username is not None or parsed.password is not None:
            return "INVALID_WEBHOOK_URL"
        if port not in (None, 443):
            return "INVALID_WEBHOOK_URL"
        if not hostname:
            return "INVALID_WEBHOOK_URL"
        if parsed.query or parsed.fragment:
            return "INVALID_WEBHOOK_URL"
        if hostname not in self.envelope.allowed_webhook_hosts:
            return "UNTRUSTED_WEBHOOK_HOST"

        if not isinstance(payload, Mapping) or len(payload) == 0:
            return "WEBHOOK_PAYLOAD_VIOLATION"

        for k, v in payload.items():
            if k not in self.envelope.allowed_webhook_fields:
                return "WEBHOOK_PAYLOAD_VIOLATION"
            if isinstance(v, str):
                if len(v) > MAX_STRING_FIELD_LENGTH:
                    return "WEBHOOK_PAYLOAD_VIOLATION"
                lower_v = v.lower()
                if any(sub in lower_v for sub in FORBIDDEN_WEBHOOK_SUBSTRINGS):
                    return "WEBHOOK_PAYLOAD_VIOLATION"
            elif is_strict_int(v):
                if abs(v) > MAX_AMOUNT_MINOR:
                    return "WEBHOOK_PAYLOAD_VIOLATION"
            elif isinstance(v, bool):
                continue
            else:
                return "WEBHOOK_PAYLOAD_VIOLATION"

        classification = payload.get("classification")
        if (
            not isinstance(classification, str)
            or classification not in self.envelope.allowed_webhook_classifications
        ):
            return "WEBHOOK_PAYLOAD_VIOLATION"

        return "OK"

