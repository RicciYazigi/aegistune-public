"""Episode rollout execution, scripted actor protocols, and isolated batch runners."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ContractValidationError,
    EpisodeResult,
    FrozenMap,
    PreActionObservation,
    freeze_value,
)
from .environment import EnvironmentFactory, StepExecutionResult, SyntheticEnvironment
from .scenario import ScenarioBundle

ActorProtocol = Callable[[PreActionObservation | FrozenMap], str | Mapping[str, Any] | None]


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for episode rollout loop termination and execution bounds."""

    max_steps: int = 10
    stop_on_contain: bool = False
    stop_on_error: bool = True
    stop_on_first_unauthorized_effect: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ContractValidationError(f"max_steps must be > 0, got {self.max_steps}")


class ScriptedSequenceActor:
    """Deterministic actor that emits a pre-configured sequence of tool proposals."""

    def __init__(self, proposals: Sequence[str | Mapping[str, Any]]) -> None:
        self._proposals = list(proposals)
        self._cursor = 0

    def __call__(self, _obs: PreActionObservation | FrozenMap) -> str | Mapping[str, Any] | None:
        if self._cursor >= len(self._proposals):
            return None
        proposal = self._proposals[self._cursor]
        self._cursor += 1
        return proposal


class ScriptedReactiveActor:
    """Deterministic actor driven by a step-indexed reactive decision callback."""

    def __init__(
        self,
        callback: Callable[[int, FrozenMap], str | Mapping[str, Any] | None],
    ) -> None:
        self._callback = callback
        self._step = 0

    def __call__(self, obs: PreActionObservation | FrozenMap) -> str | Mapping[str, Any] | None:
        frozen_obs = obs if isinstance(obs, FrozenMap) else FrozenMap({"step_index": self._step})
        proposal = self._callback(self._step, frozen_obs)
        self._step += 1
        return proposal


def execute_rollout(
    *,
    environment: SyntheticEnvironment,
    actor: ActorProtocol,
    config: RolloutConfig | None = None,
) -> EpisodeResult:
    """Execute a single episode rollout loop until completion, step limit, or error."""
    cfg = config or RolloutConfig()
    current_obs: FrozenMap = environment.scenario.case_spec.public_actor_prompt_view()
    termination_reason = "actor_completed"
    episode_error: str | None = None

    for step_idx in range(cfg.max_steps):
        try:
            proposal = actor(current_obs)
        except Exception as exc:
            termination_reason = f"actor_exception:{type(exc).__name__}"
            episode_error = f"{type(exc).__name__}: {exc}"
            break

        if proposal is None:
            termination_reason = "actor_finished"
            break

        step_res: StepExecutionResult = environment.step(proposal)
        current_obs = step_res.observation_for_actor

        if step_res.receipt.status == "error" and cfg.stop_on_error:
            termination_reason = "environment_error"
            episode_error = step_res.decision.reason_code or "ENVIRONMENT_ERROR"
            break

        if step_res.decision.status == "contained" and cfg.stop_on_contain:
            termination_reason = "contained_by_monitor"
            break

        if cfg.stop_on_first_unauthorized_effect and step_res.receipt.status == "committed":
            # Early check if single commit breached target
            if (
                step_res.receipt.effective_tool_name == "execute_instant_refund"
                and environment.scenario.mission_oracle.max_refund_commits == 0
            ):
                termination_reason = "unauthorized_effect_contained"
                break
    else:
        termination_reason = "max_steps_exceeded"

    return environment.close(termination_reason=termination_reason, error=episode_error)


class BatchRolloutRunner:
    """Manages isolated multi-episode and multi-generation rollouts with strict ID discipline."""

    def __init__(self, factory: EnvironmentFactory | None = None) -> None:
        self.factory = factory or EnvironmentFactory()
        self._seen_episode_ids: set[str] = set()
        self._seen_generation_ids: set[str] = set()

    def run_episode(
        self,
        scenario: ScenarioBundle,
        actor: ActorProtocol,
        *,
        episode_id: str | None = None,
        generation_id: str | None = None,
        model_source: str = "scripted",
        model_revision: str = "deterministic-fixture",
        monitor: Any = None,
        fallback_on_contain: bool = False,
        config: RolloutConfig | None = None,
    ) -> EpisodeResult:
        """Run a single episode with unique ID checking and sandbox isolation."""
        env = self.factory.create(
            scenario,
            episode_id=episode_id,
            generation_id=generation_id,
            model_source=model_source,
            model_revision=model_revision,
            monitor=monitor,
            fallback_on_contain=fallback_on_contain,
        )
        if env.episode_id in self._seen_episode_ids:
            raise ContractValidationError(f"Duplicate episode_id detected: {env.episode_id}")
        if env.generation_id in self._seen_generation_ids:
            raise ContractValidationError(f"Duplicate generation_id detected: {env.generation_id}")

        self._seen_episode_ids.add(env.episode_id)
        self._seen_generation_ids.add(env.generation_id)

        return execute_rollout(environment=env, actor=actor, config=config)

    def run_case_generations(
        self,
        scenario: ScenarioBundle,
        actor_factory: Callable[[str, str], ActorProtocol],
        *,
        generation_count: int = 1,
        model_source: str = "scripted",
        model_revision: str = "deterministic-fixture",
        monitor: Any = None,
        fallback_on_contain: bool = False,
        config: RolloutConfig | None = None,
    ) -> tuple[EpisodeResult, ...]:
        """Run N isolated generations for the same case, ensuring fresh environments per generation."""
        if generation_count <= 0:
            raise ContractValidationError(f"generation_count must be > 0, got {generation_count}")

        results: list[EpisodeResult] = []
        for gen_idx in range(generation_count):
            case_id = scenario.case_spec.case_id
            ep_id = f"ep-{case_id}-g{gen_idx:02d}"
            gen_id = f"gen-{case_id}-g{gen_idx:02d}"
            actor = actor_factory(ep_id, gen_id)
            res = self.run_episode(
                scenario=scenario,
                actor=actor,
                episode_id=ep_id,
                generation_id=gen_id,
                model_source=model_source,
                model_revision=model_revision,
                monitor=monitor,
                fallback_on_contain=fallback_on_contain,
                config=config,
            )
            results.append(res)

        return tuple(results)

