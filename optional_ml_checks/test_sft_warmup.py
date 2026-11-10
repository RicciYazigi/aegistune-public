"""Focused checks of causal loss and verified supervised demonstrations."""
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aegistune.model_loader import create_minimal_cohere2_model, DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from run_p4_sft_warmup import ActionOnlyTrainer, LabelCollator, build_demonstrations

torch.set_num_threads(1)


def test_selected_position_loss_equals_full_loss_and_backpropagates():
    model = create_minimal_cohere2_model(vocab_size=128, seed=42).eval()
    ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 6]])
    labels = torch.tensor([[-100, -100, 30, 40, -100, -100, 70, 6]])
    full = model(input_ids=ids, use_cache=False).logits[:, :-1]
    reference = torch.nn.functional.cross_entropy(full.reshape(-1, 128), labels[:, 1:].reshape(-1))
    trainer = object.__new__(ActionOnlyTrainer)
    selected = trainer.compute_loss(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    assert torch.allclose(selected, reference, atol=1e-6)
    selected.backward()
    assert model.get_input_embeddings().weight.grad is not None
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()


def test_padding_does_not_add_supervised_targets():
    batch = LabelCollator(0)([
        {"input_ids": [2, 10, 6], "attention_mask": [1, 1, 1], "labels": [-100, 10, 6]},
        {"input_ids": [2, 6], "attention_mask": [1, 1], "labels": [-100, 6]},
    ])
    assert batch["labels"].tolist() == [[-100, 10, 6], [-100, 6, -100]]
    assert batch["attention_mask"][1, -1] == 0


def test_weighted_loss_matches_full_causal_reference_and_masks_host():
    model = create_minimal_cohere2_model(vocab_size=128, seed=42).eval()
    ids = torch.tensor([[10, 20, 30, 40, 50, 6]])
    labels = torch.tensor([[-100, -100, 30, 40, 50, 6]])
    weights = torch.tensor([[0., 0., 8., 8., 1., 1.]])
    full = model(input_ids=ids, use_cache=False).logits[:, :-1]
    losses = torch.nn.functional.cross_entropy(full.reshape(-1, 128), labels[:, 1:].reshape(-1), reduction="none")
    reference = (losses * weights[:, 1:].flatten()).sum() / weights.sum()
    trainer = object.__new__(ActionOnlyTrainer)
    actual = trainer.compute_loss(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                                         "labels": labels, "loss_weights": weights})
    assert torch.allclose(actual, reference, atol=1e-6)
    actual.backward()
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()


def test_tool_weighting_only_current_assistant_name_with_real_tokenizer(tmp_path):
    families = {}
    for line in (ROOT / "data/p4/train_cases.jsonl").read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        families.setdefault(case["template_family"], case)
    source = tmp_path / "six.jsonl"
    source.write_text("".join(json.dumps(c) + "\n" for c in families.values()), encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION, local_files_only=True)
    normal, _ = build_demonstrations(source, tok)
    weighted, diagnostics = build_demonstrations(source, tok, tool_name_weight=8)
    for before, row in zip(normal, weighted, strict=True):
        # Fresh environments have independent random attempt identifiers in host
        # history; the supervised action must remain identical across builds.
        assert [v for v in before["labels"] if v != -100] == [v for v in row["labels"] if v != -100]
        indices = [i for i, w in enumerate(row["loss_weights"]) if w > 1]
        assert indices
        assert all(row["labels"][i] != -100 for i in indices)
        text = tok.decode([row["labels"][i] for i in indices])
        target = json.loads(tok.decode([v for v in row["labels"] if v != -100], skip_special_tokens=True))
        assert target["name"] in text
        assert all(w == 0 for w, label in zip(row["loss_weights"], row["labels"]) if label == -100)
        if target["name"] == "stage_refund_proposal":
            assert "exceeds" not in target["arguments"]["reason"]
            assert "supervisory" in target["arguments"]["reason"]
    assert len(diagnostics) == 6


def test_training_time_budget_requests_graceful_stop(monkeypatch):
    from types import SimpleNamespace
    import run_p4_sft_warmup as sft
    now = [100.0]
    monkeypatch.setattr(sft.time, "monotonic", lambda: now[0])
    callback = sft.TrainingTimeBudget(10)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_train_begin(None, None, control)
    now[0] = 105
    assert not callback.on_step_end(None, None, control).should_training_stop
    now[0] = 110
    assert callback.on_step_end(None, None, control).should_training_stop
    assert callback.expired


def test_demonstrations_cover_six_families_with_only_action_labels(tmp_path):
    families = {}
    for line in (ROOT / "data/p4/train_cases.jsonl").read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        families.setdefault(case["template_family"], case)
    source = tmp_path / "six_train.jsonl"
    source.write_text("".join(json.dumps(c) + "\n" for c in families.values()), encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION, local_files_only=True)
    rows, diagnostics = build_demonstrations(source, tokenizer)
    assert len(diagnostics) == len(families) == 6
    assert all(d["task_completed"] for d in diagnostics)
    for row in rows:
        first = next(i for i, value in enumerate(row["labels"]) if value != -100)
        assert first > 0
        assert all(x == -100 for x in row["labels"][:first])
        action = tokenizer.decode(row["labels"][first:], skip_special_tokens=True)
        parsed = json.loads(action)
        assert set(parsed) == {"name", "arguments"}
        assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])
    first_case = next(iter(families.values()))
    first_case["split"] = "dev"
    source.write_text(json.dumps(first_case) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        build_demonstrations(source, tokenizer)


@pytest.mark.parametrize("paired", [False, True])
def test_training_save_reload_pipeline_on_minimal_model(tmp_path, monkeypatch, paired):
    """Exercise the actual Trainer, optimizer, PEFT save and fresh reload on CPU."""
    import argparse
    import run_p4_sft_warmup as sft
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 6
        unk_token_id = 1
        vocab_size = 128
    tok = Tokenizer()
    monkeypatch.setattr(sft.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    rows = [{"input_ids": [2, 10, 20, 30, 6], "attention_mask": [1] * 5,
             "labels": [-100, -100, 20, 30, 6],
             "loss_weights": [0., 0., 8., 1., 1.]} for _ in range(4)]
    monkeypatch.setattr(sft, "build_demonstrations", lambda *a, **k: (rows, [{"case_id": "fixture"}]))
    monkeypatch.setattr(sft, "load_causal_lm", lambda *a, **k: create_minimal_cohere2_model(128, seed=k.get("seed", 42)))
    def evaluation(*args, **kwargs):
        kwargs["output_path"].write_text('{"summary":{"fixture":true}}')
        kwargs["trajectory_path"].write_text('{"fixture":true}\n')
        return {"summary": {"fixture": True}}
    monkeypatch.setattr(sft, "run_baseline_evaluation", evaluation)
    import run_p4_paired_dev60_evaluation as paired_module
    def paired_evaluation(**kwargs):
        kwargs["adapter_summary_path"].write_text('{"summary":{"fixture":true}}')
        kwargs["trajectory_path"].write_text('{"fixture":true,"arm":"base"}\n{"fixture":true,"arm":"adapter"}\n')
        return {"adapter_summary": {"fixture": True}, "paired_metrics_comparison": {"fixture": True}}
    monkeypatch.setattr(paired_module, "run_paired_dev60_evaluation", paired_evaluation)
    source = tmp_path / "train.jsonl"
    source.write_text("fixture")
    initial_adapter = None
    initial_sha = None
    if paired:
        from peft import LoraConfig, get_peft_model
        initial_adapter = tmp_path / "initial"
        initial = get_peft_model(create_minimal_cohere2_model(128, seed=42),
                                 LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        initial.save_pretrained(initial_adapter)
        initial_sha = sft.sha(initial_adapter / "adapter_model.safetensors")
    args = argparse.Namespace(seed=42, output_dir=tmp_path / "adapter", model_id="offline-minimal",
                              model_revision="fixture", use_minimal_offline=True, train_cases=source,
                              dev_cases=source, prepare_only=False, device="cpu", max_steps=1,
                              learning_rate=1e-3, paired_evaluation=paired,
                              initial_adapter=initial_adapter, expected_initial_sha256=initial_sha)
    report = sft.train(args)
    assert report["total_parameter_delta_l2"] > 0
    assert report["reload_max_parameter_diff"] == 0
    assert report["is_genuine_tiny_aya_weights"] is False
    assert (report["paired_evaluation"] is not None) is paired
    assert (args.output_dir / "adapter_model.safetensors").is_file()
