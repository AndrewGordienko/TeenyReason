"""Support helpers for continuous-control crawler experience."""

from __future__ import annotations

import numpy as np

from ...cognition.imagination import Proposal, Target, TargetBank
from ...cognition.imagination.schema import as_latent
from ...multidomain.planning.generic.collection.trajectory import ReplayTrajectory, make_trajectory
from ...multidomain.planning.gym_mpc import TransitionBatch
from ...cognition.scenario import ScenarioMemory, ScenarioVariant
from ...cognition.worldmap import WorldMap
from .mindmap import CrawlerMindMap


def build_target_bank(memory: ScenarioMemory, world_map: WorldMap | None, *, count: int = 32) -> TargetBank:
    bank = TargetBank()
    rows = sorted(
        memory.real_tracelets(),
        key=lambda item: float(item.return_to_go) - 2.0 * float(item.done) - 0.25 * float(item.surprise),
        reverse=True,
    )
    for index, row in enumerate(rows[: max(1, int(count))]):
        bank.add(
            Target(
                target_id=f"real:{index}",
                kind="high_return_state",
                latent=as_latent(row.observation),
                utility=float(row.return_to_go),
                stability=float(1.0 - row.done),
                source="scenario_memory",
                metadata={"trajectory_id": int(row.trajectory_id), "step": int(row.step)},
            )
        )
    if world_map is not None:
        for index, edge in enumerate(world_map.ranked_edges(count=8)):
            bank.add(
                Target(
                    target_id=f"edge:{index}",
                    kind="worldmap_edge",
                    utility=max(0.0, float(edge.utility)),
                    stability=float(edge.confidence),
                    source="worldmap",
                    metadata={"edge_id": edge.edge_id, "relation": edge.relation},
                )
            )
    return bank


def sync_targets_to_mindmap(mindmap: CrawlerMindMap, targets: TargetBank) -> None:
    for target in targets.top(count=48):
        mindmap.add_target(
            target.target_id,
            target.kind,
            latent=target.latent,
            utility=target.utility,
            stability=target.stability,
            source=target.source,
        )


def sync_worldmap_to_mindmap(mindmap: CrawlerMindMap, world_map: WorldMap | None) -> None:
    if world_map is None:
        return
    for node in world_map.nodes():
        mindmap.add_factor_node(
            node.node_id,
            node.kind,
            vector=node.vector,
            confidence=float(node.confidence),
            utility=float(node.utility),
            support=float(node.support),
            metadata={"label": node.label, "source": "worldmap"},
        )
    for edge in world_map.edges():
        mindmap.add_edge(
            f"factor:{edge.source}",
            edge.relation,
            f"factor:{edge.target}",
            strength=float(edge.effect_size),
            confidence=float(edge.confidence),
            utility=float(edge.utility),
            cost=float(edge.sample_cost),
            metadata={"edge_id": edge.edge_id, "source": "worldmap"},
        )


def proposal_from_variant(config, variant: ScenarioVariant, *, target: Target | None, proposal_id: str) -> Proposal:
    predicted_latent = ()
    if variant.rows:
        predicted_latent = as_latent(variant.rows[-1]["next_observation"])
    return Proposal(
        proposal_id=str(proposal_id),
        domain=str(config.env_name),
        context_id=str(variant.variant_kind),
        context_latent=as_latent(variant.window.start_observation),
        target=target,
        intervention=np.asarray(variant.actions, dtype=np.float32).copy(),
        predicted_latent=predicted_latent,
        predicted_utility=float(variant.predicted_return + variant.predicted_value),
        predicted_lift=float(variant.predicted_lift),
        uncertainty=float(variant.uncertainty),
        support_confidence=float(variant.weights.familiarity),
        reachability=float(variant.weights.plausibility),
        consistency=float(variant.weights.inverse_surprise),
        trust_score=float(variant.weights.combined),
        expected_solver_utility=float(max(0.0, variant.predicted_lift)),
        horizon=int(np.asarray(variant.actions).shape[0]),
        generation_cost=0.05 * float(len(variant.rows)),
        metadata={
            "variant_kind": str(variant.variant_kind),
            "done_risk": float(variant.done_risk),
            "observed_return": float(variant.window.observed_return),
        },
    )


def graph_variant_count(config, world_map: WorldMap) -> int:
    configured = int(getattr(config, "worldmap_prior_count", 0))
    if configured > 0:
        return configured
    useful = [
        edge
        for edge in world_map.ranked_edges(count=8)
        if float(edge.utility) > 0.0 and float(edge.confidence) >= 0.10
    ]
    return min(2, len(useful))


def batch_to_replay_trajectories(batch: TransitionBatch, *, seed_start: int, discount: float) -> list[ReplayTrajectory]:
    trajectories: list[ReplayTrajectory] = []
    start = 0
    count = int(batch.actions.shape[0])
    for index in range(count):
        done = float(batch.dones[index]) > 0.5
        if done or index == count - 1:
            end = index + 1
            if end > start:
                trajectories.append(
                    make_trajectory(
                        seed=int(seed_start + len(trajectories)),
                        observations=[batch.observations[idx].copy() for idx in range(start, end)],
                        actions=[batch.actions[idx].copy() for idx in range(start, end)],
                        rewards=[float(batch.rewards[idx]) for idx in range(start, end)],
                        next_observations=[batch.next_observations[idx].copy() for idx in range(start, end)],
                        dones=[float(batch.dones[idx]) for idx in range(start, end)],
                        discount=float(discount),
                    )
                )
            start = end
    return trajectories


def dedupe_vectors(values: list[np.ndarray], *, count: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for value in values:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        key = tuple(float(item) for item in np.round(vector, 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(vector.copy())
        if len(out) >= int(count):
            break
    return out


def rollout_arrays(
    rows: list[dict[str, np.ndarray | float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not rows:
        return None
    observations = np.asarray([row["observation"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float32)
    if observations.ndim != 2 or actions.ndim != 2 or rewards.ndim != 1:
        return None
    return observations, actions, rewards


def first_or_none(values: list[Target]) -> Target | None:
    return values[0] if values else None


__all__ = [
    "batch_to_replay_trajectories",
    "build_target_bank",
    "dedupe_vectors",
    "first_or_none",
    "graph_variant_count",
    "proposal_from_variant",
    "rollout_arrays",
    "sync_targets_to_mindmap",
    "sync_worldmap_to_mindmap",
]
