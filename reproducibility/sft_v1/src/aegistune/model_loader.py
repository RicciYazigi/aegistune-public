"""Model Loading & Architecture Factory Utilities.

Provides decoupled model loading for both evaluation and training pipelines:
- Clean import boundaries: Only requires torch, transformers, and peft.
  Does NOT import datasets, agenttune, or trl, ensuring evaluation containers
  remain free of training-only dependency conflicts.
- Explicit opt-in for minimal offline architectures.
- Deterministic seeding for minimal test models.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"
DEFAULT_MODEL_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"


def create_minimal_cohere2_model(vocab_size: int = 261008, seed: int = 42) -> Any:
    """Instantiate a minimal architectural instance of Cohere2ForCausalLM for CPU testing.

    Uses the authentic Cohere2 architecture configuration (layers, attention, q_proj/v_proj)
    with vocabulary matching Tiny Aya tokenizer (261,008 tokens) for full CPU execution
    without multi-gigabyte weight downloads.

    Args:
        vocab_size: Vocabulary size for embeddings and lm_head (defaults to 261,008).
        seed: Random seed for deterministic weight initialization so that reloading onto
              a fresh base model has identical underlying base parameters.
    """
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "cohere2",
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=4096,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=6,
    )
    return AutoModelForCausalLM.from_config(cfg)


def causal_model_dtype(device: str) -> torch.dtype:
    target = torch.device(device)
    if target.type != "cuda":
        return torch.float32
    with torch.cuda.device(target):
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def load_causal_lm(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    device: str = "cpu",
    *,
    use_minimal_offline: bool = False,
    tokenizer: Any = None,
    seed: int = 42,
) -> Any:
    """Load pretrained causal model or fail explicitly.

    If use_minimal_offline is True, creates a minimal Cohere2ForCausalLM for offline testing.
    In production mode, loads directly from HuggingFace Hub; never silently falls back to a stub.
    """
    if use_minimal_offline:
        v_size = len(tokenizer) if tokenizer is not None else 261008
        logger.info(f"Initializing minimal Cohere2 architecture (vocab_size={v_size}, seed={seed}) for CPU integration testing...")
        model = create_minimal_cohere2_model(vocab_size=v_size, seed=seed)
        return model.to(device)

    logger.info(f"Loading pretrained causal model {model_id} (revision {revision[:8]}...) on {device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=causal_model_dtype(device),
        )
        return model.to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load production causal model {model_id}@{revision}: {exc}. "
            f"Verify HuggingFace token and gated repo access. Offline minimal testing requires explicit opt-in."
        ) from exc


CANONICAL_TERMINATOR_NAMES: tuple[str, ...] = (
    "<|END_OF_TURN_TOKEN|>",
    "<|END_RESPONSE|>",
    "<EOS_TOKEN>",
)


def resolve_terminator_token_ids(
    tokenizer: Any = None,
    model_config: Any = None,
    *,
    fallback_eos_id: int = 6,
) -> list[int]:
    """Resolve centralized list of effective terminator token IDs across model/tokenizer contracts.

    Probes canonical terminator token names (<|END_OF_TURN_TOKEN|>, <|END_RESPONSE|>, <EOS_TOKEN>),
    tokenizer-declared EOS attributes, and model config declarations.
    Enforces strict validity:
    - Must be non-negative integers.
    - Must not be None or unk_token_id.
    - Must be within known vocabulary bounds.
    - Deduplicates preserving discovery order.
    - Backward-compatible with small vocabulary test stubs and scalar fallbacks.

    Returns:
        A non-empty list of unique valid integer token IDs.
    """
    resolved: list[int] = []
    seen: set[int] = set()

    # Determine vocabulary upper bound if available
    vocab_size: int | None = None
    if tokenizer is not None:
        if hasattr(tokenizer, "__len__"):
            try:
                vocab_size = len(tokenizer)
            except Exception:
                pass
        if vocab_size is None and hasattr(tokenizer, "vocab_size"):
            vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None and model_config is not None:
        vocab_size = getattr(model_config, "vocab_size", None)

    unk_id: int | None = None
    if tokenizer is not None:
        unk_id = getattr(tokenizer, "unk_token_id", None)

    def _add_if_valid(candidate: Any) -> None:
        if candidate is None:
            return
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                _add_if_valid(item)
            return
        try:
            val = int(candidate)
        except (ValueError, TypeError):
            return
        if val < 0:
            return
        if unk_id is not None and val == unk_id:
            return
        if vocab_size is not None and val >= vocab_size:
            return
        if val not in seen:
            seen.add(val)
            resolved.append(val)

    # 1. Probe canonical special token names via tokenizer methods
    if tokenizer is not None:
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            for name in CANONICAL_TERMINATOR_NAMES:
                try:
                    tid = tokenizer.convert_tokens_to_ids(name)
                    _add_if_valid(tid)
                except Exception:
                    pass
        if hasattr(tokenizer, "encode"):
            for name in CANONICAL_TERMINATOR_NAMES:
                try:
                    enc = tokenizer.encode(name, add_special_tokens=False)
                    if isinstance(enc, list) and len(enc) == 1:
                        _add_if_valid(enc[0])
                except Exception:
                    pass

    # 2. Check tokenizer.eos_token_id
    if tokenizer is not None:
        _add_if_valid(getattr(tokenizer, "eos_token_id", None))

    # 3. Check model_config.eos_token_id
    if model_config is not None:
        _add_if_valid(getattr(model_config, "eos_token_id", None))

    # 4. Fallback if empty
    if not resolved:
        safe_fallback = fallback_eos_id
        if vocab_size is not None and safe_fallback >= vocab_size:
            safe_fallback = max(0, vocab_size - 1)
        resolved.append(safe_fallback)

    return resolved


def describe_terminator_resolution(
    tokenizer: Any = None,
    model_config: Any = None,
) -> dict[str, Any]:
    """Capture diagnostic audit metadata of terminator resolution."""
    resolved_ids = resolve_terminator_token_ids(tokenizer, model_config)
    token_mappings: dict[str, Any] = {}
    if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        for name in CANONICAL_TERMINATOR_NAMES:
            try:
                token_mappings[name] = tokenizer.convert_tokens_to_ids(name)
            except Exception as e:
                token_mappings[name] = f"error: {e}"

    return {
        "resolved_eos_token_ids": resolved_ids,
        "primary_eos_token_id": resolved_ids[0] if resolved_ids else None,
        "canonical_names_probed": list(CANONICAL_TERMINATOR_NAMES),
        "token_mappings": token_mappings,
        "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None) if tokenizer else None,
        "model_config_eos_token_id": getattr(model_config, "eos_token_id", None) if model_config else None,
    }
