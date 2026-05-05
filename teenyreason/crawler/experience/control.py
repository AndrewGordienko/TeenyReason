"""Continuous-control crawler experience shared by memory, imagination, and skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...cognition.imagination import ImaginationMemory, TargetBank, ValidationResult
from ...multidomain.planning.generic.collection.trajectory import (
    ReplayTrajectory,
    rows_to_trajectory,
)
from ...multidomain.planning.gym_mpc import TransitionBatch
from ...cognition.scenario import ScenarioMemory, ScenarioVariant, ScenarioWindow, score_variant_weights
from ...cognition.scenario.variation import final_value, generate_variants, rollout_actions
from ...cognition.skills import SkillMemory, SkillPracticeResult
from ...cognition.worldmap import WorldMap, build_control_worldmap, graph_guided_sequences
from .control_support import (
    batch_to_replay_trajectories,
    build_target_bank,
    dedupe_vectors,
    first_or_none,
    graph_variant_count,
    proposal_from_variant,
    rollout_arrays,
    sync_targets_to_mindmap,
    sync_worldmap_to_mindmap,
)
from .intrinsic import IntrinsicDrive, PracticeTarget
from .mindmap import CrawlerMindMap


@dataclass
class ControlCrawlerExperience:
    """One evolving crawler-side world model for a continuous-control run.

    The object keeps the pieces that were previously managed separately:
    real trace memory, imagined proposals, graph/worldmap belief, target memory,
    and validated skills. The solver still owns policy updates; this object owns
    the crawler's reusable understanding of what has been seen and imagined.
    """

    config: Any
    batch: TransitionBatch
    action_low: np.ndarray
    action_high: np.ndarray
    replay_trajectories: list[ReplayTrajectory]
    memory: ScenarioMemory
    skill_memory: SkillMemory
    imagination: ImaginationMemory
    mindmap: CrawlerMindMap
    drive: IntrinsicDrive
    targets: TargetBank
    drive_targets: list[PracticeTarget]
    world_map: WorldMap | None
    diagnostics: dict[str, object] = field(default_factory=dict)
    guided_variant_count: int = 0
    trainable_variant_count: int = 0
    focus_target_count: int = 0
    rollout_validation_count: int = 0

    def refresh_batch(self, batch: TransitionBatch) -> None:
        self.batch = batch
        self.world_map = build_world_map(self.config, batch, self.action_low, self.action_high)
        self.refresh_targets()

    def refresh_targets(self) -> None:
        self.targets = build_target_bank(self.memory, self.world_map)
        sync_targets_to_mindmap(self.mindmap, self.targets)
        sync_worldmap_to_mindmap(self.mindmap, self.world_map)
        self.drive_targets = self.drive.refresh(self.targets, self.mindmap, self.skill_memory, self.world_map)

    def focus_observations(
        self,
        base_focus: list[np.ndarray],
        *,
        count: int,
    ) -> list[np.ndarray]:
        """Blend frontier focus with remembered high-value targets."""
        obs_dim = int(np.asarray(self.batch.observations).shape[1])
        target_focus: list[np.ndarray] = []
        for target in self.drive.top(count=max(1, int(count))):
            latent = np.asarray(target.latent, dtype=np.float32).reshape(-1)
            if latent.shape[0] == obs_dim:
                target_focus.append(latent.copy())
        if not target_focus:
            for target in self.targets.top(count=max(1, int(count)), kind="high_return_state"):
                latent = np.asarray(target.latent, dtype=np.float32).reshape(-1)
                if latent.shape[0] == obs_dim:
                    target_focus.append(latent.copy())
        self.focus_target_count = len(target_focus)
        return dedupe_vectors(target_focus + list(base_focus), count=max(1, int(count)))

    def absorb_skill_round(self, result: SkillPracticeResult) -> list[ReplayTrajectory]:
        before = self.skill_memory.records()
        self.skill_memory.extend(result.skills)
        if result.goal_actor is not None:
            self.skill_memory.set_goal_actor(result.goal_actor)
        for skill in result.skills:
            self.mindmap.add_skill_node(skill)
        self.mindmap.add_skill_compositions(before + result.skills)
        accepted = list(result.accepted_trajectories)
        for trajectory in accepted:
            self.memory.add_trajectory(trajectory, source="real", surprise=0.0)
            self.replay_trajectories.append(trajectory)
            self.mindmap.add_episode_node(
                f"skill:{trajectory.seed}:{trajectory.length}",
                "real_skill",
                vector=tuple(float(item) for item in trajectory.observations[0][:32]),
                utility=float(trajectory.episode_return),
                support=float(trajectory.length),
                metadata={"seed": int(trajectory.seed), "length": int(trajectory.length)},
            )
        self.refresh_targets()
        return accepted

    def generate_variants(
        self,
        model,
        value_model,
        action_value_model,
        windows: list[ScenarioWindow],
        *,
        round_idx: int,
    ) -> list[ScenarioVariant]:
        local = generate_variants(
            self.config,
            model,
            value_model,
            action_value_model,
            windows,
            self.action_low,
            self.action_high,
            round_idx=round_idx,
        )
        guided = self._graph_guided_variants(
            model,
            value_model,
            action_value_model,
            windows,
            round_idx=round_idx,
        )
        self.guided_variant_count = len(guided)
        self.trainable_variant_count = len(local) + len(guided)
        return local + guided

    def trainable_variants(self, variants: list[ScenarioVariant]) -> list[ScenarioVariant]:
        """Return imagined variants allowed to train the actor this round."""
        if not variants:
            self.trainable_variant_count = 0
            return []
        if len(self.mindmap.nearest_residuals((), (), k=1)) < 1:
            self.trainable_variant_count = len(variants)
            return variants
        target = first_or_none(self.targets.top(count=1))
        scored = []
        for index, variant in enumerate(variants):
            proposal = proposal_from_variant(
                self.config,
                variant,
                target=target,
                proposal_id=f"candidate:{index}",
            )
            correction = self.mindmap.correction_for_proposal(proposal)
            score = self.drive.score_proposal(proposal, correction)
            scored.append((score, variant))
        scored.sort(key=lambda item: item[0], reverse=True)
        keep = [variant for score, variant in scored if score > 0.0]
        if not keep:
            fallback_count = max(1, min(len(scored), int(self.config.scenario_window_count)))
            keep = [variant for _score, variant in scored[:fallback_count]]
        self.trainable_variant_count = len(keep)
        return keep

    def record_imagined_variants(
        self,
        variants: list[ScenarioVariant],
        *,
        round_idx: int,
        surprise: float,
    ) -> None:
        target = first_or_none(self.targets.top(count=1))
        for index, variant in enumerate(variants):
            self.memory.add_rows(variant.rows, source="imagined", surprise=surprise)
            proposal = proposal_from_variant(
                self.config,
                variant,
                target=target,
                proposal_id=f"{self.config.env_name}:r{int(round_idx)}:v{int(index)}",
            )
            self.imagination.add_proposal(proposal)
            correction = self.mindmap.correction_for_proposal(proposal)
            self.mindmap.add_proposal(proposal, correction)
            self.mindmap.add_episode_node(
                proposal.proposal_id,
                "imagined",
                vector=proposal.predicted_latent,
                utility=correction.corrected_predicted_lift,
                support=float(proposal.horizon),
                metadata={"variant_kind": str(variant.variant_kind), "round": int(round_idx)},
            )

    def record_real_rollout(
        self,
        rows: list[dict[str, np.ndarray | float]],
        *,
        seed: int,
        discount: float,
        surprise: float,
    ) -> ReplayTrajectory | None:
        trajectory = rows_to_trajectory(rows, seed=int(seed), discount=float(discount))
        if trajectory is not None:
            self.memory.add_trajectory(trajectory, source="real", surprise=surprise)
            self.replay_trajectories.append(trajectory)
            self.mindmap.add_episode_node(
                f"rollout:{seed}:{trajectory.length}",
                "real_rollout",
                vector=tuple(float(item) for item in trajectory.observations[0][:32]),
                utility=float(trajectory.episode_return),
                support=float(trajectory.length),
                metadata={"seed": int(seed), "surprise": float(surprise)},
            )
            self.refresh_targets()
        return trajectory

    def validate_imagination_against_rollout(
        self,
        rows: list[dict[str, np.ndarray | float]],
        *,
        max_validations: int = 16,
    ) -> None:
        """Calibrate imagined proposals against nearby real rollout segments."""
        rollout = rollout_arrays(rows)
        if rollout is None:
            return
        observations, actions, rewards = rollout
        pending = [sample for sample in self.imagination.samples() if not sample.validated]
        if not pending:
            return
        obs_std = np.maximum(np.std(self.batch.observations, axis=0), 1e-4).reshape(1, -1)
        action_scale = np.maximum(self.action_high - self.action_low, 1e-4).reshape(1, -1)
        scored: list[tuple[float, Any]] = []
        for sample in pending:
            intervention = np.asarray(sample.proposal.intervention, dtype=np.float32)
            if intervention.size == 0:
                continue
            context = np.asarray(sample.proposal.context_latent, dtype=np.float32).reshape(1, -1)
            first_action = intervention.reshape(-1, self.action_low.shape[0])[:1]
            if context.shape[1] != observations.shape[1]:
                continue
            obs_distance = np.mean(np.square((observations - context) / obs_std), axis=1)
            action_distance = np.mean(np.square((actions - first_action) / action_scale), axis=1)
            distance = obs_distance + 0.35 * action_distance
            best_idx = int(np.argmin(distance))
            scored.append((float(distance[best_idx]), (sample, best_idx)))
        scored.sort(key=lambda item: item[0])
        for distance, payload in scored[: max(0, int(max_validations))]:
            sample, start = payload
            horizon = max(1, int(sample.proposal.horizon))
            end = min(int(rewards.shape[0]), int(start) + horizon)
            if end <= int(start):
                continue
            real_utility = float(np.sum(rewards[int(start) : end]))
            observed_return = float(sample.proposal.metadata.get("observed_return", 0.0))
            real_lift = float(real_utility - observed_return)
            correction = self.mindmap.correction_for_proposal(sample.proposal)
            validation = ValidationResult(
                proposal_id=sample.proposal.proposal_id,
                accepted=real_lift > 0.0,
                real_utility=real_utility,
                real_lift=real_lift,
                validation_cost=float(end - int(start)),
                validator="own_rollout_nearest_trace",
                rejected_reason="" if real_lift > 0.0 else "no_real_lift",
                metadata={"nearest_distance": float(distance), "matched_step": int(start)},
            )
            self.imagination.add_validation(validation)
            self.mindmap.add_residual(
                sample.proposal,
                real_lift=real_lift,
                accepted=real_lift > 0.0,
                validation_cost=float(end - int(start)),
                correction=correction,
                metadata={"nearest_rollout_distance": float(distance)},
            )
            self.mindmap.add_value_evidence(
                f"proposal:{sample.proposal.proposal_id}",
                "real_lift",
                value=float(real_lift),
                cost=float(end - int(start)),
                accepted=real_lift > 0.0,
            )
            self.rollout_validation_count += 1

    def summary(self, *, prefix: str = "crawler_experience") -> dict[str, float]:
        out = {
            f"{prefix}_contract_active": 1.0,
            f"{prefix}_real_trajectory_count": float(len(self.replay_trajectories)),
            f"{prefix}_target_focus_count": float(self.focus_target_count),
            f"{prefix}_guided_variant_count": float(self.guided_variant_count),
            f"{prefix}_trainable_variant_count": float(self.trainable_variant_count),
            f"{prefix}_rollout_validation_count": float(self.rollout_validation_count),
            f"{prefix}_memory_real_count": float(len(self.memory.real_tracelets())),
            f"{prefix}_memory_imagined_count": float(len(self.memory.imagined_tracelets())),
        }
        out.update(self.imagination.summary(prefix="general_imagination"))
        out.update(self.mindmap.summary(prefix="crawler_mindmap"))
        out.update(self.drive.summary(prefix="intrinsic_drive"))
        out.update(self.targets.summary(prefix="imagination_targets"))
        if self.world_map is not None:
            out.update(self.world_map.summary())
        return out

    def _graph_guided_variants(
        self,
        model,
        value_model,
        action_value_model,
        windows: list[ScenarioWindow],
        *,
        round_idx: int,
    ) -> list[ScenarioVariant]:
        del round_idx
        if self.world_map is None or not windows or not bool(getattr(self.config, "worldmap_enabled", True)):
            return []
        count = graph_variant_count(self.config, self.world_map)
        if count <= 0:
            return []
        sequences = graph_guided_sequences(
            self.world_map,
            self.action_low,
            self.action_high,
            horizon=max(1, int(self.config.scenario_variant_horizon)),
            count=count,
        )
        variants: list[ScenarioVariant] = []
        for index, actions in enumerate(sequences):
            window = windows[index % len(windows)]
            rows, predicted_return, done_risk, uncertainty = rollout_actions(
                model,
                window.start_observation,
                actions,
            )
            value = final_value(
                self.config,
                model,
                value_model,
                action_value_model,
                window.start_observation,
                actions,
            )
            predicted_lift = float(predicted_return + value - window.observed_return)
            weights = score_variant_weights(
                window,
                predicted_lift=predicted_lift,
                uncertainty=uncertainty,
                done_risk=done_risk,
                variant_surprise=0.5 * window.mean_surprise,
                advantage_temperature=float(self.config.scenario_advantage_temperature),
                uncertainty_scale=float(self.config.scenario_uncertainty_scale),
                surprise_scale=float(self.config.scenario_surprise_scale),
            )
            variants.append(
                ScenarioVariant(
                    window=window,
                    actions=actions,
                    rows=tuple(rows),
                    predicted_return=float(predicted_return),
                    predicted_value=float(value),
                    predicted_lift=predicted_lift,
                    uncertainty=float(uncertainty),
                    done_risk=float(done_risk),
                    weights=weights,
                    variant_kind=f"worldmap_guided_{index}",
                )
            )
        return variants


def build_control_crawler_experience(
    config,
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
    collector_stats: dict[str, object],
    *,
    replay_trajectories: list[ReplayTrajectory] | None = None,
) -> ControlCrawlerExperience:
    trajectories = list(replay_trajectories or [])
    if not trajectories:
        trajectories = batch_to_replay_trajectories(
            batch,
            seed_start=int(config.seed + 229_000),
            discount=float(config.discount),
        )
    memory = ScenarioMemory.from_trajectories(trajectories)
    if not memory.tracelets():
        memory = ScenarioMemory.from_batch(batch, discount=float(config.discount))
    world_map = build_world_map(config, batch, action_low, action_high)
    experience = ControlCrawlerExperience(
        config=config,
        batch=batch,
        action_low=np.asarray(action_low, dtype=np.float32).reshape(-1),
        action_high=np.asarray(action_high, dtype=np.float32).reshape(-1),
        replay_trajectories=trajectories,
        memory=memory,
        skill_memory=SkillMemory(),
        imagination=ImaginationMemory(),
        mindmap=CrawlerMindMap(),
        drive=IntrinsicDrive(),
        targets=TargetBank(),
        drive_targets=[],
        world_map=world_map,
        diagnostics=dict(collector_stats),
    )
    experience.refresh_targets()
    return experience


def build_world_map(config, batch: TransitionBatch, action_low: np.ndarray, action_high: np.ndarray) -> WorldMap | None:
    if not bool(getattr(config, "worldmap_enabled", True)):
        return None
    return build_control_worldmap(
        batch,
        action_low,
        action_high,
        min_effect=float(getattr(config, "worldmap_edge_min_effect", 0.05)),
    )


__all__ = ["ControlCrawlerExperience", "build_control_crawler_experience"]
