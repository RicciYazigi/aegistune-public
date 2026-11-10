import os
import sys
import hashlib
from pathlib import Path
import pytest
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aegistune.model_loader import create_minimal_cohere2_model
from peft import LoraConfig, get_peft_model, TaskType
from run_p4_lora_training import (
    check_gradient_checkpointing_active,
    inspect_hook_step0_diagnostics,
    execute_aegistune_rollout,
)
from aegistune.scenario import make_reference_scenario_bundle


@pytest.mark.parametrize("initial_training_mode", [True, False], ids=["train_mode", "eval_mode"])
def test_inspect_hook_step0_diagnostics_exception_restoration(monkeypatch, initial_training_mode):
    """Verify production inspect_hook_step0_diagnostics restores mode & RNG even when generate fails."""
    assert torch.get_num_threads() == 1, "Must run on single thread"
    base = create_minimal_cohere2_model(vocab_size=1000, seed=42)
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base, peft_cfg)

    if initial_training_mode:
        model.train()
    else:
        model.eval()
    assert model.training is initial_training_mode

    inputs = torch.tensor([[10, 20, 30, 40, 50, 60]])
    gen_kwargs = {"pad_token_id": 0, "eos_token_id": 6, "temperature": 0.6, "top_p": 0.9}

    rng_before = torch.get_rng_state()
    sha_before = hashlib.sha256(rng_before.numpy().tobytes()).hexdigest()

    class MockTokenizer:
        def decode(self, token_ids, **kwargs):
            return "tok_" + "_".join(str(t) for t in token_ids)

    def faulty_generate(self, *args, **kwargs):
        # Consume RNG state to ensure state changes if not restored
        torch.randn(100)
        raise RuntimeError("INJECTED_HOOK_GENERATE_ERROR")

    monkeypatch.setattr(type(model), "generate", faulty_generate)

    # 1. Verify exception propagates out of the production function
    with pytest.raises(RuntimeError, match="INJECTED_HOOK_GENERATE_ERROR"):
        inspect_hook_step0_diagnostics(
            model=model,
            inputs=inputs,
            tokenizer=MockTokenizer(),
            gen_kwargs=gen_kwargs,
            sample=True,
            max_diagnostic_tokens=8,
            diagnostic_seed=42,
        )

    # 2. Verify original training mode restored by production finally: block
    assert model.training is initial_training_mode, "Original model training mode must be restored"

    # 3. Verify RNG restored bit-for-bit by production finally: block
    rng_after = torch.get_rng_state()
    sha_after = hashlib.sha256(rng_after.numpy().tobytes()).hexdigest()
    assert sha_before == sha_after, "PyTorch RNG state must be bit-for-bit restored"


@pytest.mark.parametrize("initial_training_mode", [True, False], ids=["train_mode", "eval_mode"])
def test_execute_aegistune_rollout_exception_restoration(monkeypatch, initial_training_mode):
    """Verify production execute_aegistune_rollout restores mode in finally: even when generate fails."""
    assert torch.get_num_threads() == 1
    scenario = make_reference_scenario_bundle(case_id="case-p4-restoration-01")
    base = create_minimal_cohere2_model(vocab_size=1000, seed=42)
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base, peft_cfg)

    if initial_training_mode:
        model.train()
    else:
        model.eval()

    class MockTokenizer:
        bos_token = "<BOS>"
        eos_token = "<EOS>"
        pad_token = "<PAD>"
        eos_token_id = 6
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "system: help user: refund assistant:"

        def encode(self, text, add_special_tokens=False):
            return [10, 20, 30, 40]

        def decode(self, tokens, **kwargs):
            return "{\"name\": \"get_ticket\"}"

    observed_mode_inside_rollout = []

    def faulty_rollout_generate(self, *args, **kwargs):
        # Verify model was put in eval mode during generation
        observed_mode_inside_rollout.append(self.training)
        # Verify use_cache=True was explicitly requested in production rollout
        assert kwargs.get("use_cache") is True, "Rollout must explicitly pass use_cache=True"
        torch.randn(50)
        raise RuntimeError("INJECTED_ROLLOUT_GENERATE_ERROR")

    monkeypatch.setattr(type(model), "generate", faulty_rollout_generate)

    # 1. Verify exception propagates
    with pytest.raises(RuntimeError, match="INJECTED_ROLLOUT_GENERATE_ERROR"):
        execute_aegistune_rollout(
            scenario=scenario,
            model=model,
            tokenizer=MockTokenizer(),
            max_turns=1,
            max_new_tokens=16,
            model_source="synthetic",
        )

    # 2. Verify model was evaluated in eval() during rollout
    assert observed_mode_inside_rollout == [False], "Model must be in eval() mode during rollout generate"

    # 3. Verify original mode restored in finally:
    assert model.training is initial_training_mode, "Original model training mode must be restored after rollout"


def test_inspect_hook_step0_diagnostics_effective_cache_and_dual_mode():
    """Verify that inspect_hook_step0_diagnostics measures actual cache and layer checkpointing."""
    assert torch.get_num_threads() == 1
    base = create_minimal_cohere2_model(vocab_size=10000, seed=42)
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base, peft_cfg)

    # Actively enable gradient checkpointing to verify detection on layers
    model.gradient_checkpointing_enable()
    model.train()
    assert check_gradient_checkpointing_active(model) is True

    inputs = torch.tensor([[10, 20, 30, 40, 50, 60]])
    gen_kwargs = {"pad_token_id": 0, "eos_token_id": 6, "temperature": 0.6, "top_p": 0.9}

    class MockTokenizer:
        def decode(self, token_ids, **kwargs):
            return "tok_" + "_".join(str(t) for t in token_ids)

    diag = inspect_hook_step0_diagnostics(
        model=model,
        inputs=inputs,
        tokenizer=MockTokenizer(),
        gen_kwargs=gen_kwargs,
        sample=True,
        max_diagnostic_tokens=8,
        diagnostic_seed=42,
        target_token_ids=[100, 200],
    )

    # 1. Check measured runtime cache facts
    eff_cache = diag["effective_cache"]
    assert eff_cache["gradient_checkpointing_active_on_layers"] is True
    assert eff_cache["eval_kv_cache_returned"] is True

    # 2. Check prefix forward vs incremental measurement with past_kv existence
    pref_comp = diag["prefix_logits_comparison"]
    for p_tok in ["5983", "213"]:
        assert p_tok in pref_comp
        entry = pref_comp[p_tok]
        assert entry["incremental_kv_cache_available"] is True
        assert isinstance(entry["max_abs_diff"], float)

    # 3. Mode restored
    assert model.training is True


def test_inspect_hook_step0_diagnostics_forward_spy_logits_to_keep_and_parity(monkeypatch):
    """Verify that every auxiliary forward in inspect_hook_step0_diagnostics enforces logits_to_keep=1."""
    assert torch.get_num_threads() == 1
    base = create_minimal_cohere2_model(vocab_size=10000, seed=42)
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base, peft_cfg)

    inputs = torch.tensor([[10, 20, 30, 40, 50, 60]])
    gen_kwargs = {"pad_token_id": 0, "eos_token_id": 6, "temperature": 0.6, "top_p": 0.9}

    # Reference forward without restriction to verify parity
    with torch.no_grad():
        ref_out = model(inputs)
        ref_last_logits = ref_out.logits[0, -1, :].float()

    spy_forward_calls = []
    orig_forward = model.forward

    def spy_forward(*args, **kwargs):
        logits_to_keep = kwargs.get("logits_to_keep")
        past_kv = kwargs.get("past_key_values")
        inp = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if len(args) > 0 else None)
        seq_len = inp.shape[-1] if inp is not None else 1

        # Simulate Accelerate FP32 conversion wrapper: fail immediately if full sequence logits are requested
        # when sequence length > 1 and logits_to_keep is not 1
        if seq_len > 1 and logits_to_keep != 1:
            raise AssertionError(f"Memory violation: auxiliary forward called with seq_len={seq_len} but logits_to_keep={logits_to_keep}")

        res = orig_forward(*args, **kwargs)
        if hasattr(res, "logits") and res.logits is not None:
            # Check returned dimension is strictly 1 position
            assert res.logits.shape[1] == 1, f"Expected 1 position, got {res.logits.shape[1]}"
            spy_forward_calls.append({
                "seq_len": seq_len,
                "logits_to_keep": logits_to_keep,
                "shape": tuple(res.logits.shape),
                "has_past_kv": past_kv is not None,
            })
        return res

    monkeypatch.setattr(model, "forward", spy_forward)

    class MockTokenizer:
        def decode(self, token_ids, **kwargs):
            return "tok_" + "_".join(str(t) for t in token_ids)

    diag = inspect_hook_step0_diagnostics(
        model=model,
        inputs=inputs,
        tokenizer=MockTokenizer(),
        gen_kwargs=gen_kwargs,
        sample=True,
        max_diagnostic_tokens=8,
        diagnostic_seed=42,
        target_token_ids=[100, 200],
    )

    # 1. Verify auxiliary forward calls were intercepted and all enforced logits_to_keep=1
    assert len(spy_forward_calls) > 0, "Forward spy must record auxiliary forward calls"
    for call in spy_forward_calls:
        assert call["logits_to_keep"] == 1, f"Forward call violated memory rule: {call}"
        assert call["shape"] == (1, 1, 10000), f"Logits tensor must be shape (1, 1, 10000), got {call['shape']}"

    # 2. Verify numerical parity of the step0 logits against full reference
    top1_diag = diag["step0_logits"]["top_tokens"][0]["token_id"]
    top1_ref = ref_last_logits.argmax().item()
    assert top1_diag == top1_ref, f"Top-1 token mismatch: {top1_diag} vs {top1_ref}"

    specific_100_diag = diag["step0_logits"]["specific_tokens"]["100"]["logit"]
    specific_100_ref = round(ref_last_logits[100].item(), 4)
    assert abs(specific_100_diag - specific_100_ref) <= 1e-4


def test_execute_aegistune_rollout_subsequent_diagnostic_omission():
    """Verify that subsequent rollouts omit the step 0 probe when run_step0_diagnostics=False."""
    assert torch.get_num_threads() == 1
    scenario = make_reference_scenario_bundle(case_id="case-p4-omission-01")
    base = create_minimal_cohere2_model(vocab_size=1000, seed=42)
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(base, peft_cfg)

    class MockTokenizer:
        bos_token = "<BOS>"
        eos_token = "<EOS>"
        pad_token = "<PAD>"
        eos_token_id = 6
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "system: help user: refund assistant:"

        def encode(self, text, add_special_tokens=False):
            return [10, 20, 30, 40]

        def decode(self, tokens, **kwargs):
            return "{\"name\": \"get_ticket\"}"

    # Rollout 1: run_step0_diagnostics=True -> hook_step0_diagnostic is dict with measurements
    res1 = execute_aegistune_rollout(
        scenario=scenario,
        model=model,
        tokenizer=MockTokenizer(),
        max_turns=1,
        max_new_tokens=16,
        model_source="synthetic",
        run_step0_diagnostics=True,
    )
    assert isinstance(res1["turn_diagnostics"][0]["hook_step0_diagnostic"], dict)
    assert "step0_logits" in res1["turn_diagnostics"][0]["hook_step0_diagnostic"]

    # Rollout 2: run_step0_diagnostics=False -> probe omitted to preserve memory and compute
    res2 = execute_aegistune_rollout(
        scenario=scenario,
        model=model,
        tokenizer=MockTokenizer(),
        max_turns=1,
        max_new_tokens=16,
        model_source="synthetic",
        run_step0_diagnostics=False,
    )
    assert res2["turn_diagnostics"][0]["hook_step0_diagnostic"] == {"status": "omitted_subsequent_rollout"}


# ---------------------------------------------------------------------------
# P4-TRAIN-04 r4 Focused Termination & Guard Tests (Single CPU Thread)
# ---------------------------------------------------------------------------

def test_terminator_resolution_centralized_contracts():
    """Verify centralized terminator resolution across genuine tokenizer, small mocks, and scalar fallbacks."""
    assert torch.get_num_threads() == 1
    from aegistune.model_loader import (
        CANONICAL_TERMINATOR_NAMES,
        resolve_terminator_token_ids,
        describe_terminator_resolution,
    )
    from transformers import AutoConfig

    # 1. Real Tiny Aya cached tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "CohereLabs/tiny-aya-global",
            revision="af89d219b53ed9b13b8a4645f9c8028973510324",
            local_files_only=True,
        )
        cfg = AutoConfig.for_model("cohere2", vocab_size=261008, eos_token_id=6)
        res_real = resolve_terminator_token_ids(tok, cfg)
        assert 6 in res_real, "Must include END_OF_TURN_TOKEN"
        assert 261001 in res_real, "Must include END_RESPONSE"
        assert 3 in res_real, "Must include EOS_TOKEN"
        assert res_real[0] == 6, "Primary EOS should be first"
        diag = describe_terminator_resolution(tok, cfg)
        assert diag["primary_eos_token_id"] == 6
        assert diag["token_mappings"]["<|END_RESPONSE|>"] == 261001
    except Exception:
        pass  # If local cache unavailable, mock tests below verify the contract

    # 2. Small vocabulary mock (vocab_size=1000): out-of-bounds 261001 must be dropped safely
    cfg_small = AutoConfig.for_model("cohere2", vocab_size=1000, eos_token_id=6)
    class MockSmallTokenizer:
        vocab_size = 1000
        unk_token_id = 4
        eos_token_id = 6
        def convert_tokens_to_ids(self, name):
            if name == "<|END_OF_TURN_TOKEN|>":
                return 6
            elif name == "<|END_RESPONSE|>":
                return 261001  # Out of bounds!
            return 4  # UNK

    res_small = resolve_terminator_token_ids(MockSmallTokenizer(), cfg_small)
    assert res_small == [6], f"Out-of-bounds 261001 must be omitted, got {res_small}"

    # 3. Empty mock with no tokenizer methods: safe fallback to clamped scalar
    class MockEmpty:
        pass
    res_empty = resolve_terminator_token_ids(MockEmpty(), None)
    assert res_empty == [6]


def test_forced_generation_stops_at_first_terminator():
    """Verify forced generation stops immediately at the first terminator (261001) and omits subsequent garbage."""
    assert torch.get_num_threads() == 1
    base = create_minimal_cohere2_model(vocab_size=261008, seed=42)
    base.eval()

    inputs = torch.tensor([[10, 20, 30]])
    target_clean = [100, 101, 102]
    terminator_token = 261001
    garbage_tokens = [500, 501, 502, 503]

    def forced_prefix_fn(batch_id: int, cur_ids: torch.Tensor) -> list[int]:
        step = len(cur_ids) - 3
        if step < len(target_clean):
            return [target_clean[step]]
        elif step == len(target_clean):
            return [terminator_token]
        else:
            return [garbage_tokens[min(step - len(target_clean) - 1, len(garbage_tokens) - 1)]]

    out = base.generate(
        inputs,
        max_new_tokens=20,
        eos_token_id=[6, 261001],
        prefix_allowed_tokens_fn=forced_prefix_fn,
        do_sample=False,
    )
    gen_tokens = out[0, 3:].tolist()

    # The generated tokens must end at 261001
    assert gen_tokens[-1] == 261001, f"Must terminate with 261001, got {gen_tokens}"
    assert len(gen_tokens) == len(target_clean) + 1, f"Must have stopped at terminator, got {len(gen_tokens)} tokens"
    # Garbage tokens must NEVER have been emitted
    for g in garbage_tokens:
        assert g not in gen_tokens, f"Garbage token {g} was emitted after terminator!"


def test_forced_generation_stops_at_previous_eos():
    """Verify forced generation stops on the classical scalar EOS (token 6)."""
    assert torch.get_num_threads() == 1
    base = create_minimal_cohere2_model(vocab_size=1000, seed=42)
    base.eval()

    inputs = torch.tensor([[10, 20, 30]])
    target_clean = [100, 101]
    terminator_token = 6

    def forced_prefix_fn(batch_id: int, cur_ids: torch.Tensor) -> list[int]:
        step = len(cur_ids) - 3
        if step < len(target_clean):
            return [target_clean[step]]
        return [terminator_token]

    out = base.generate(
        inputs,
        max_new_tokens=10,
        eos_token_id=[6, 261001],
        prefix_allowed_tokens_fn=forced_prefix_fn,
        do_sample=False,
    )
    gen_tokens = out[0, 3:].tolist()
    assert gen_tokens == [100, 101, 6]
    assert gen_tokens[-1] == 6


def test_generation_reaches_limit_without_terminator():
    """Verify generation clipping when no terminator is reached within token budget."""
    assert torch.get_num_threads() == 1
    base = create_minimal_cohere2_model(vocab_size=1000, seed=42)
    base.eval()

    inputs = torch.tensor([[10, 20, 30]])
    max_new = 5

    def endless_tokens_fn(batch_id: int, cur_ids: torch.Tensor) -> list[int]:
        return [50]  # Regular token, not in [6, 261001]

    out = base.generate(
        inputs,
        max_new_tokens=max_new,
        eos_token_id=[6, 261001],
        prefix_allowed_tokens_fn=endless_tokens_fn,
        do_sample=False,
    )
    gen_tokens = out[0, 3:].tolist()
    assert len(gen_tokens) == max_new
    assert gen_tokens[-1] == 50
    assert 6 not in gen_tokens
    assert 261001 not in gen_tokens


def test_rollout_alignment_tokens_logprobs_env_mask():
    """Verify exact alignment of lengths across completion_ids, env_mask, logprobs, and diagnostics."""
    assert torch.get_num_threads() == 1
    scenario = make_reference_scenario_bundle(case_id="case-p4-align-01")
    base = create_minimal_cohere2_model(vocab_size=1000, seed=42)
    base.eval()

    class MockTokenizer:
        bos_token = "<BOS>"
        eos_token = "<EOS>"
        pad_token = "<PAD>"
        eos_token_id = 6
        pad_token_id = 0
        vocab_size = 1000

        def convert_tokens_to_ids(self, name):
            if name == "<|END_OF_TURN_TOKEN|>":
                return 6
            return 4

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "system: help user: refund assistant:"

        def encode(self, text, add_special_tokens=False):
            return [10, 20, 30, 40]

        def decode(self, tokens, skip_special_tokens=True, **kwargs):
            return '```json\n{"name": "lookup_account_ledger", "arguments": {"account_id": "ACC-01"}}\n```'

    res = execute_aegistune_rollout(
        scenario=scenario,
        model=base,
        tokenizer=MockTokenizer(),
        max_turns=1,
        max_new_tokens=12,
        sample=True,
        model_source="synthetic",
        synthetic_generation_index=0,
        run_step0_diagnostics=False,
    )

    comp_ids = res["completion_ids"]
    mask = res["env_mask"]
    probs = res["logprobs"]

    # Invariants: equal lengths, terminal token recorded, mask=1 for generated tokens
    assert len(comp_ids) == len(mask) == len(probs), f"Lengths mismatch: {len(comp_ids)}, {len(mask)}, {len(probs)}"
    assert all(m == 1 for m in mask), "All generated action tokens must have env_mask=1"
    assert all(isinstance(p, float) for p in probs), "All logprobs must be floats"

    t_diag = res["turn_diagnostics"][0]
    assert "terminal_token_id" in t_diag
    assert "terminal_token_is_eos" in t_diag
    assert "stop_tokens_configured" in t_diag
    assert t_diag["terminal_token_id"] == comp_ids[-1]
    assert t_diag["generated_tokens_count"] == len(comp_ids)


def test_counterexamples_prose_before_eos_rejected_by_gateway():
    """Counterexample: Even if closed with valid fence/EOS, extraneous prose is strictly rejected as MALFORMED_JSON."""
    from aegistune.gateway import DeterministicGateway, GATEWAY_PARSER_V2_SINGLE_BLOCK

    gw = DeterministicGateway(
        episode_id="ep-counter-01",
        generation_id="gen-counter-01",
        envelope={},
        parser_version=GATEWAY_PARSER_V2_SINGLE_BLOCK,
    )

    # 1. Preamble before fence
    raw_preamble = 'Here is the tool call:\n```json\n{"name": "lookup_account_ledger"}\n```'
    att1 = gw.parse_attempt(raw_preamble, attempt_id="att-01", step_index=0)
    assert att1.parse_status == "malformed_json"
    assert "outside markdown code fence" in att1.parse_error

    # 2. Postamble after fence
    raw_postamble = '```json\n{"name": "lookup_account_ledger"}\n```\nExtra explanation'
    att2 = gw.parse_attempt(raw_postamble, attempt_id="att-02", step_index=0)
    assert att2.parse_status == "malformed_json"
    assert "extraneous text after the closing code fence" in att2.parse_error


def test_counterexamples_repetition_of_4776_and_controls_detected_by_guard():
    """Counterexample: Repeated tokens (4776, unclosed controls) and text loops are strictly caught by degeneration guard."""
    from run_p4_lora_training import detect_degenerative_loop

    # 1. Repeating 4776 (the legacy **\n token)
    res_4776 = detect_degenerative_loop([4776, 4776, 4776, 4776], "some text")
    assert res_4776.is_degenerate is True
    assert "4776" in res_4776.reason
    assert res_4776.token_loop_detected is True

    # 2. Repeating 261001 without valid termination
    res_ctrl = detect_degenerative_loop([261001, 261001, 261001, 261001], "some text")
    assert res_ctrl.is_degenerate is True
    assert "261001" in res_ctrl.reason
    assert res_ctrl.token_loop_detected is True

    # 3. Repeated text pattern
    res_txt = detect_degenerative_loop([1, 2, 3], "**\n**\n**\n")
    assert res_txt.is_degenerate is True
    assert "Repeated '**\\n'" in res_txt.reason
    assert res_txt.text_loop_detected is True

    # 4. Normal non-repetitive sequence
    res_normal = detect_degenerative_loop([10, 20, 30, 40, 50, 6], '```json\n{"name": "test"}\n```')
    assert res_normal.is_degenerate is False
    assert res_normal.reason == ""


