"""Build and query world maps from generic continuous-control data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teenyreason.multidomain.planning.gym_mpc import TransitionBatch, normalize_actions

from .graph import WorldMap
from .schema import EvidenceRef, WorldNode


@dataclass(frozen=True)
class GraphPlanScore:
    trust: float
    utility: float
    risk: float
    support: float
    edge_ids: tuple[str, ...]


def build_control_worldmap(
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    min_effect: float = 0.05,
    max_state_factors: int = 32,
) -> WorldMap:
    """Extract generic factor/action/event relations from real transitions."""
    world = WorldMap()
    observations = np.asarray(batch.observations, dtype=np.float32)
    actions = np.asarray(batch.actions, dtype=np.float32)
    next_observations = np.asarray(batch.next_observations, dtype=np.float32)
    rewards = np.asarray(batch.rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(batch.dones, dtype=np.float32).reshape(-1)
    if observations.size == 0 or actions.size == 0:
        return world

    action_z = normalize_actions(actions, action_low, action_high)
    obs_std = np.maximum(np.std(observations, axis=0), 1e-4)
    delta_z = (next_observations - observations) / obs_std.reshape(1, -1)
    returns = returns_to_go(rewards, dones, discount=0.99)
    state_indices = selected_state_indices(observations, delta_z, max_state_factors=max_state_factors)
    add_base_nodes(world, action_z, observations, state_indices, rewards, dones, returns)
    add_action_state_edges(world, action_z, delta_z, returns, state_indices, min_effect=min_effect)
    add_action_event_edges(world, action_z, rewards, dones, returns, min_effect=min_effect)
    add_state_event_edges(world, observations, rewards, dones, returns, state_indices, min_effect=min_effect)
    return world


def action_sequence_graph_score(
    world: WorldMap,
    sequence: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> GraphPlanScore:
    actions = np.asarray(sequence, dtype=np.float32).reshape(sequence.shape[0], -1)
    action_z = normalize_actions(actions, action_low, action_high)
    utility = 0.0
    risk = 0.0
    support = 0.0
    edge_ids: list[str] = []
    for row in action_z:
        for action_idx, value in enumerate(row):
            if abs(float(value)) < 0.08:
                continue
            source = action_node_id(action_idx, float(value))
            magnitude = abs(float(value))
            edges = world.outgoing(source)
            support += min(1.0, len(edges) / 4.0) * magnitude
            for edge in edges[:4]:
                signed = float(edge.confidence) * float(edge.utility) * magnitude
                if edge.target == "event:terminal" and edge.relation == "predicts":
                    risk += abs(float(edge.effect_size)) * float(edge.confidence) * magnitude
                else:
                    utility += signed
                edge_ids.append(edge.edge_id)
    denom = max(1.0, float(action_z.shape[0] * action_z.shape[1]))
    trust = max(0.0, utility - risk) / denom
    return GraphPlanScore(
        trust=float(trust),
        utility=float(utility / denom),
        risk=float(risk / denom),
        support=float(support / denom),
        edge_ids=tuple(dict.fromkeys(edge_ids)),
    )


def graph_guided_sequences(
    world: WorldMap,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    horizon: int,
    count: int,
) -> list[np.ndarray]:
    """Return simple action priors from high-utility graph edges."""
    if int(count) <= 0:
        return []
    rows: list[np.ndarray] = []
    action_dim = int(np.asarray(action_low).reshape(-1).shape[0])
    for edge in world.ranked_edges(count=max(1, int(count) * 4)):
        parsed = parse_action_node(edge.source)
        if parsed is None or edge.target == "event:terminal":
            continue
        action_idx, sign = parsed
        action = np.zeros((action_dim,), dtype=np.float32)
        action[action_idx] = action_high[action_idx] if sign > 0 else action_low[action_idx]
        action = 0.65 * action
        rows.append(np.repeat(action.reshape(1, -1), max(1, int(horizon)), axis=0).astype(np.float32))
        if len(rows) >= int(count):
            break
    return rows


def add_base_nodes(
    world: WorldMap,
    action_z: np.ndarray,
    observations: np.ndarray,
    state_indices: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    returns: np.ndarray,
) -> None:
    support = float(observations.shape[0])
    for action_idx in range(action_z.shape[1]):
        world.add_node(WorldNode(action_node_id(action_idx, 1.0), "action_factor", support=support, confidence=1.0))
        world.add_node(WorldNode(action_node_id(action_idx, -1.0), "action_factor", support=support, confidence=1.0))
    for state_idx in state_indices:
        utility = abs(corr(observations[:, int(state_idx)], returns))
        world.add_node(
            WorldNode(
                state_node_id(int(state_idx)),
                "state_factor",
                support=support,
                confidence=confidence_from_effect(utility, support),
                utility=utility,
            )
        )
    event_values = {
        "event:terminal": float(np.mean(dones)) if dones.size else 0.0,
        "event:survival": float(np.mean(1.0 - dones)) if dones.size else 0.0,
        "event:high_reward": percentile_score(rewards, 80.0),
        "event:high_value": percentile_score(returns, 80.0),
    }
    for node_id, utility in event_values.items():
        world.add_node(WorldNode(node_id, "event", support=support, confidence=1.0, utility=utility))


def add_action_state_edges(
    world: WorldMap,
    action_z: np.ndarray,
    delta_z: np.ndarray,
    returns: np.ndarray,
    state_indices: np.ndarray,
    *,
    min_effect: float,
) -> None:
    support = float(action_z.shape[0])
    for action_idx in range(action_z.shape[1]):
        effects = [corr(action_z[:, action_idx], delta_z[:, int(state_idx)]) for state_idx in state_indices]
        keep = keep_indices(effects, min_effect=min_effect, top_count=3)
        for local_idx in keep:
            state_idx = int(state_indices[int(local_idx)])
            effect = float(effects[int(local_idx)])
            source = action_node_id(action_idx, effect)
            target = state_node_id(state_idx)
            state_value = abs(corr(delta_z[:, state_idx], returns))
            world.add_edge(
                source,
                "causes",
                target,
                effect_size=abs(effect),
                confidence=confidence_from_effect(effect, support),
                utility=abs(effect) + state_value,
                sample_cost=support,
                evidence=EvidenceRef("transition_batch", f"action:{action_idx}->state:{state_idx}", utility=state_value, cost=support),
            )


def add_action_event_edges(
    world: WorldMap,
    action_z: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    returns: np.ndarray,
    *,
    min_effect: float,
) -> None:
    support = float(action_z.shape[0])
    targets = (("event:terminal", dones), ("event:high_reward", rewards), ("event:high_value", returns))
    for action_idx in range(action_z.shape[1]):
        for target, values in targets:
            effect = corr(action_z[:, action_idx], values)
            if abs(effect) < float(min_effect):
                continue
            relation = "predicts"
            if target == "event:terminal" and effect < 0.0:
                relation = "prevents"
            source = action_node_id(action_idx, effect)
            utility = -abs(effect) if target == "event:terminal" and relation == "predicts" else abs(effect)
            world.add_edge(
                source,
                relation,
                target,
                effect_size=abs(effect),
                confidence=confidence_from_effect(effect, support),
                utility=utility,
                sample_cost=support,
                evidence=EvidenceRef("transition_batch", f"action:{action_idx}->{target}", utility=utility, cost=support),
            )


def add_state_event_edges(
    world: WorldMap,
    observations: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    returns: np.ndarray,
    state_indices: np.ndarray,
    *,
    min_effect: float,
) -> None:
    support = float(observations.shape[0])
    targets = (("event:terminal", dones), ("event:high_reward", rewards), ("event:high_value", returns))
    for state_idx in state_indices:
        source = state_node_id(int(state_idx))
        for target, values in targets:
            effect = corr(observations[:, int(state_idx)], values)
            if abs(effect) < float(min_effect):
                continue
            relation = "predicts"
            if target == "event:terminal" and effect < 0.0:
                relation = "prevents"
            utility = -abs(effect) if target == "event:terminal" and relation == "predicts" else abs(effect)
            world.add_edge(
                source,
                relation,
                target,
                effect_size=abs(effect),
                confidence=confidence_from_effect(effect, support),
                utility=utility,
                sample_cost=support,
                evidence=EvidenceRef("transition_batch", f"state:{int(state_idx)}->{target}", utility=utility, cost=support),
            )


def selected_state_indices(observations: np.ndarray, delta_z: np.ndarray, *, max_state_factors: int) -> np.ndarray:
    score = np.std(observations, axis=0) + np.std(delta_z, axis=0)
    order = np.argsort(score)[::-1]
    return np.sort(order[: max(1, min(int(max_state_factors), int(order.shape[0])))]).astype(np.int64)


def action_node_id(action_idx: int, value: float) -> str:
    direction = "positive" if float(value) >= 0.0 else "negative"
    return f"action:{int(action_idx)}:{direction}"


def parse_action_node(node_id: str) -> tuple[int, int] | None:
    parts = str(node_id).split(":")
    if len(parts) != 3 or parts[0] != "action":
        return None
    sign = 1 if parts[2] == "positive" else -1
    return int(parts[1]), sign


def state_node_id(state_idx: int) -> str:
    return f"state:{int(state_idx)}"


def keep_indices(values: list[float], *, min_effect: float, top_count: int) -> list[int]:
    if not values:
        return []
    ranked = sorted(range(len(values)), key=lambda idx: abs(values[idx]), reverse=True)
    return [idx for idx in ranked[: max(0, int(top_count))] if abs(values[idx]) >= float(min_effect)]


def confidence_from_effect(effect: float, support: float) -> float:
    support_gain = float(np.sqrt(max(0.0, float(support)) / max(1.0, float(support) + 20.0)))
    return float(np.clip(abs(float(effect)) * support_gain, 0.0, 1.0))


def percentile_score(values: np.ndarray, percentile: float) -> float:
    rows = np.asarray(values, dtype=np.float32).reshape(-1)
    if rows.size == 0:
        return 0.0
    threshold = float(np.percentile(rows, float(percentile)))
    return float(np.mean(rows >= threshold))


def corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size < 2 or right.size < 2 or float(np.std(left)) < 1e-6 or float(np.std(right)) < 1e-6:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def returns_to_go(rewards: np.ndarray, dones: np.ndarray, *, discount: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(rewards.size - 1, -1, -1):
        if idx < rewards.size - 1 and float(dones[idx]) > 0.5:
            running = 0.0
        running = float(rewards[idx]) + float(discount) * running
        out[idx] = running
    return out


__all__ = ["GraphPlanScore", "action_sequence_graph_score", "build_control_worldmap", "graph_guided_sequences"]
