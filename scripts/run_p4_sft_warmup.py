"""Supervised PEFT warmup from sandbox-verified synthetic demonstrations.

This is SFT, not GRPO. Privileged teacher labels are generated offline; oracle
metadata is never serialized into model inputs. Real model loading is mandatory
unless --use-minimal-offline is explicitly selected for integration tests.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from aegistune.environment import EnvironmentFactory
from aegistune.integrations import TinyAyaMessageAdapter, TinyAyaPromptFormatter
from aegistune.model_loader import (
    DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, describe_terminator_resolution,
    load_causal_lm, resolve_terminator_token_ids,
)
from aegistune.scenario import evaluate_mission_oracle
from run_p4_baseline_evaluation import (
    PrivilegedOracleCeilingActor, deserialize_scenario_bundle, run_baseline_evaluation,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_name_loss_weights(tokenizer, full_text, input_ids, prompt_length, action, weight):
    """Weight the generated JSON name, never a name mentioned in host history.

    Offsets bind weights to the actual tokenization, including tokens that span
    a JSON quote boundary. No family/oracle features are added to model inputs.
    """
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("Tool-name loss weight must be finite and at least one")
    result = [0.0] * prompt_length + [1.0] * (len(input_ids) - prompt_length)
    if weight == 1:
        return result
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"][:len(input_ids)] != input_ids:
        raise ValueError("Offset tokenization differs from demonstration")
    marker = json.dumps({"name": action["name"]}, separators=(",", ":"))[:-1]
    # The action is the final assistant message; previous actions can repeat.
    start = full_text.rfind(marker)
    if start < 0:
        raise ValueError("Generated tool-name span missing")
    start += len('{"name":"')
    end = start + len(action["name"])
    selected = []
    for index, (left, right) in enumerate(encoded["offset_mapping"][:len(input_ids)]):
        if right > start and left < end:
            if index < prompt_length:
                raise ValueError("Tool-name weighting crossed the prompt boundary")
            result[index] = float(weight)
            selected.append(index)
    if not selected:
        raise ValueError("Tool-name span has no target tokens")
    return result


def build_demonstrations(cases_path: Path, tokenizer, max_context_tokens=4096, tool_name_weight=1.0):
    """One training row per action, retaining all prior host feedback masked.

    Every episode must pass the independent outcome oracle before any of its
    action rows are accepted. Train-only input prevents accidental dev/test use.
    """
    rows, diagnostics = [], []
    seen = set()
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scenario = deserialize_scenario_bundle(json.loads(line))
        spec = scenario.case_spec
        if spec.split != "train" or spec.case_id in seen:
            raise ValueError("Demonstrations require unique train-only cases")
        seen.add(spec.case_id)
        teacher = PrivilegedOracleCeilingActor(scenario)
        if not teacher.target_account or not teacher.target_ticket:
            raise ValueError(f"Teacher lacks required entities: {spec.case_id}")
        if spec.template_family == "legitimate_refund" and not scenario.mission_oracle.authorized_amount_minor:
            raise ValueError("Missing exact refund label")
        env = EnvironmentFactory().create(scenario)
        messages = TinyAyaPromptFormatter(language=spec.language).render_messages(
            task_prompt=spec.user_task, public_policy=spec.public_policy,
        )
        candidate_rows, observation = [], None
        actions = []
        for _ in range(4):
            action = teacher.act(observation)
            text = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": text}],
                tokenize=False, add_generation_prompt=False,
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(full, add_special_tokens=False)
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("Tokenizer changed the prompt boundary")
            target = full_ids[len(prompt_ids):]
            stops = set(resolve_terminator_token_ids(tokenizer))
            terminal = next((i for i, t in enumerate(target) if t in stops), None)
            if terminal is None:
                raise ValueError("Chat template did not supply a response terminator")
            target = target[:terminal + 1]
            ids = prompt_ids + target
            if len(ids) > max_context_tokens:
                raise ValueError(f"Untruncated demonstration exceeds context: {spec.case_id}: {len(ids)}")
            candidate_rows.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                                   "labels": [-100] * len(prompt_ids) + target,
                                   "loss_weights": tool_name_loss_weights(
                                       tokenizer, full, ids, len(prompt_ids), action, tool_name_weight)})
            result = env.step(text)
            actions.append(action["name"])
            if not result.decision.allowed:
                raise ValueError(f"Expert demonstration denied: {spec.case_id}: {result.decision.reason_code}")
            observation = result.observation_for_actor
            messages.append({"role": "assistant", "content": text})
            if action["name"] in {"save_reply_draft", "stage_refund_proposal", "execute_instant_refund"}:
                break
            messages.append(TinyAyaMessageAdapter.wrap_tool_observation(
                observation.to_plain_dict(), tool_name=result.attempt.parsed_tool_name,
                attempt_id=result.attempt.attempt_id, language=spec.language,
            ))
        episode = env.close(termination_reason="verified_teacher_demonstration")
        # Same independent oracle used by runtime evaluation, not teacher assertion.
        verdict = evaluate_mission_oracle(
            scenario=scenario, initial_state_snapshot=episode.initial_state_snapshot,
            final_state_snapshot=episode.final_state_snapshot, receipts=episode.receipts,
        )
        if not episode.valid or not verdict.task_completed or verdict.unauthorized_effect:
            raise ValueError(f"Expert mission invalid: {spec.case_id}")
        rows.extend(candidate_rows)
        diagnostics.append({"case_id": spec.case_id, "family": spec.template_family,
                            "language": spec.language, "tools": actions,
                            "task_completed": True, "teacher_source": "privileged_offline_labels"})
    if not rows:
        raise ValueError("No training demonstrations")
    return rows, diagnostics


class LabelCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        length = max(len(x["input_ids"]) for x in features)
        result = {}
        for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor([x[key] + [pad] * (length - len(x[key])) for x in features])
        if any("loss_weights" in x for x in features):
            result["loss_weights"] = torch.tensor([
                x.get("loss_weights", [float(t != -100) for t in x["labels"]])
                + [0.0] * (length - len(x["input_ids"])) for x in features
            ], dtype=torch.float32)
        return result


class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ActionOnlyTrainer(Trainer):
    """Exact causal CE on assistant targets, without logits for masked host tokens.

    Microbatch size is one; gradient accumulation supplies the effective batch.
    Cohere2 selects hidden states before lm_head via logits_to_keep tensor indices.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        if labels.shape[0] != 1:
            raise ValueError("ActionOnlyTrainer requires microbatch size one")
        positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError("No supervised assistant targets")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        use_cache=False, logits_to_keep=positions)
        targets = labels[:, positions + 1]
        token_loss = torch.nn.functional.cross_entropy(
            outputs.logits.float().reshape(-1, outputs.logits.size(-1)), targets.reshape(-1), reduction="none",
        )
        weights = inputs.get("loss_weights")
        if weights is None:
            loss = token_loss.mean()
        else:
            selected_weights = weights[:, positions + 1].reshape(-1)
            if not torch.isfinite(selected_weights).all() or (selected_weights <= 0).any():
                raise ValueError("Supervised loss weights must be finite and positive")
            loss = (token_loss * selected_weights).sum() / selected_weights.sum()
        return (loss, outputs) if return_outputs else loss


class TrainingTimeBudget(TrainerCallback):
    """Leave time for adapter serialization, fresh reload and evaluation."""
    def __init__(self, seconds):
        self.seconds = seconds
        self.started = None
        self.expired = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.started >= self.seconds:
            self.expired = True
            control.should_training_stop = True
        return control


def train(args):
    torch.manual_seed(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision,
                                      local_files_only=args.use_minimal_offline)
    name_weight = getattr(args, "tool_name_weight", 1.0)
    rows, diagnostics = build_demonstrations(args.train_cases, tok, tool_name_weight=name_weight)
    if getattr(args, "operational_reweighting", False):
        original_rows = list(rows)
        offset = 0
        for episode in diagnostics:
            offset += len(episode["tools"])
            copies = {"stage_refund_proposal": 3, "execute_instant_refund": 1}.get(episode["tools"][-1], 0)
            rows.extend([original_rows[offset - 1]] * copies)
        if offset != len(original_rows):
            raise ValueError("Demonstration/action join mismatch")
    (out / "rollout_diagnostics.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in diagnostics), encoding="utf-8")
    if args.prepare_only:
        report = {"cases": len(diagnostics), "action_rows": len(rows),
                  "max_sequence_tokens": max(len(r["input_ids"]) for r in rows),
                  "target_tokens": sum(sum(t != -100 for t in r["labels"]) for r in rows),
                  "tool_name_weight": name_weight,
                  "weighted_name_tokens": sum(sum(w > 1 for w in r.get("loss_weights", [])) for r in rows),
                  "train_sha256": sha(args.train_cases), "teacher": "privileged_offline_labels",
                  "algorithm": "SFT_PREPARATION_ONLY", "trained": False}
        (out / "demonstration_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        return report
    base = load_causal_lm(args.model_id, args.model_revision, args.device,
                          use_minimal_offline=args.use_minimal_offline, tokenizer=tok, seed=args.seed)
    base.config.use_cache = False
    base.generation_config.eos_token_id = resolve_terminator_token_ids(tok, base.config)
    initial_adapter = getattr(args, "initial_adapter", None)
    if initial_adapter:
        expected = getattr(args, "expected_initial_sha256", None)
        if not expected or sha(initial_adapter / "adapter_model.safetensors") != expected:
            raise ValueError("Initial adapter hash mismatch")
        model = PeftModel.from_pretrained(base, initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                               target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM))
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}
    cuda = torch.device(args.device).type == "cuda"
    dtype = next(model.parameters()).dtype
    cfg = TrainingArguments(
        output_dir=str(out / "checkpoints"), max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=args.learning_rate, warmup_steps=min(args.max_steps - 1, int(args.max_steps * 0.05)),
        logging_steps=5, save_steps=25, save_total_limit=2,
        gradient_checkpointing=True, bf16=cuda and dtype == torch.bfloat16,
        fp16=cuda and dtype == torch.float16, use_cpu=not cuda,
        report_to="none", seed=args.seed, remove_unused_columns=False,
    )
    budget = TrainingTimeBudget(getattr(args, "max_training_seconds", 9000))
    trainer = ActionOnlyTrainer(model=model, args=cfg, train_dataset=ActionDataset(rows),
                      data_collator=LabelCollator(tok.pad_token_id or 0), callbacks=[budget])
    trainer.model_accepts_loss_kwargs = False
    result = trainer.train()
    completed_steps = trainer.state.global_step
    if completed_steps <= 0 or (completed_steps != args.max_steps and not budget.expired):
        raise RuntimeError(f"Unexpected training stop after {completed_steps}/{args.max_steps} steps")
    trained = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}
    delta = math.sqrt(sum(float(((trained[n].float() - p.float()) ** 2).sum()) for n, p in initial.items()))
    if not math.isfinite(delta) or delta <= 0 or not all(torch.isfinite(p).all() for p in trained.values()):
        raise RuntimeError("SFT did not produce finite nonzero parameter change")
    model.save_pretrained(out)
    del trainer, model, base, initial
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    fresh = load_causal_lm(args.model_id, args.model_revision, args.device,
                           use_minimal_offline=args.use_minimal_offline, tokenizer=tok, seed=args.seed)
    reloaded = PeftModel.from_pretrained(fresh, out)
    for name, p in reloaded.named_parameters():
        if name in trained and not torch.equal(p.detach().cpu(), trained[name]):
            raise RuntimeError(f"SFT reload mismatch: {name}")
    reloaded.eval()
    terminator_audit = describe_terminator_resolution(tok, reloaded.config)
    paired = None
    if getattr(args, "paired_evaluation", False):
        del reloaded, fresh, trained
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        from run_p4_paired_dev60_evaluation import run_paired_dev60_evaluation
        paired = run_paired_dev60_evaluation(
            dev_cases_path=args.dev_cases, adapter_path=out, output_path=out / "paired_report.json",
            trajectory_path=out / "dev_trajectory.jsonl", base_summary_path=out / "base_report.json",
            adapter_summary_path=out / "dev_evaluation_report.json",
            base_trajectory_path=out / "base_trajectory.jsonl", adapter_trajectory_path=out / "adapter_trajectory.jsonl",
            expected_adapter_sha256=sha(out / "adapter_model.safetensors"),
            model_id=args.model_id, model_revision=args.model_revision, device=args.device,
            max_context_tokens=4096, max_new_tokens=128, max_turns=4,
            use_minimal_offline=args.use_minimal_offline,
        )
        evaluation = {"summary": paired["adapter_summary"]}
    else:
        evaluation = run_baseline_evaluation(
            args.dev_cases, "causal_model", model=reloaded, tokenizer=tok,
            output_path=out / "dev_evaluation_report.json", trajectory_path=out / "dev_trajectory.jsonl",
            model_id=args.model_id, model_revision=args.model_revision, adapter_path=out,
            device=args.device, max_new_tokens=128, max_turns=4,
            use_minimal_offline=args.use_minimal_offline,
        )
    report = {"algorithm": "SUPERVISED_SFT_TRANSFORMERS_PEFT", "not_grpo": True,
              "initial_adapter_sha256": getattr(args, "expected_initial_sha256", None),
              "operational_reweighting": getattr(args, "operational_reweighting", False),
              "tool_name_weight": name_weight,
              "is_genuine_tiny_aya_weights": not args.use_minimal_offline,
              "steps_completed": completed_steps, "steps_requested": args.max_steps,
              "termination_reason": "training_time_budget" if budget.expired else "max_steps_completed",
              "training_metrics": result.metrics,
              "total_parameter_delta_l2": delta, "reload_max_parameter_diff": 0.0,
              "cases": len(diagnostics), "action_rows": len(rows),
              "train_sha256": sha(args.train_cases), "adapter_safetensors_sha256": sha(out / "adapter_model.safetensors"),
              "terminator_audit": terminator_audit, "paired_evaluation": paired,
              "dev_evaluation_summary": evaluation["summary"]}
    (out / "training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cases", type=Path, required=True)
    p.add_argument("--dev-cases", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--max-training-seconds", type=float, default=9000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--paired-evaluation", action="store_true")
    p.add_argument("--operational-reweighting", action="store_true")
    p.add_argument("--tool-name-weight", type=float, default=1.0)
    p.add_argument("--initial-adapter", type=Path)
    p.add_argument("--expected-initial-sha256")
    p.add_argument("--use-minimal-offline", action="store_true")
    args = p.parse_args()
    if args.max_steps <= 0 or args.max_training_seconds <= 0:
        p.error("step and time limits must be positive")
    if not math.isfinite(args.tool_name_weight) or args.tool_name_weight < 1:
        p.error("tool-name weight must be finite and at least one")
    train(args)


if __name__ == "__main__":
    main()
