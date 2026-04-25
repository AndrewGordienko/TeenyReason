"""Full-system controller, planner, and simulator-fanout entrypoints."""

from .affordance import (
    AffordanceSelection,
    BeliefAffordanceController,
    choose_affordance_action,
    generate_candidate_actions,
    mean_to_action,
)
from .affordance_eval import EvaluationEpisodeFixture, evaluate_belief_affordance_fixtures
from .affordance_train import train_belief_affordance_controller
from .planner import (
    BeliefDynamicsModel,
    BeliefPlannerConfig,
    PlanningBeliefState,
    build_planner_probe_dataset,
    fit_belief_dynamics_model,
    plan_cem_action,
    planner_trust_from_context,
    replay_batch_mean_error,
    update_belief_dynamics_from_replay,
    update_planner_prior,
)
from .planner_eval import evaluate_belief_planner, should_stop_belief_planner_plateau
from .planner_train import train_belief_planner
from .simulator_fanout import (
    ContinuousCartPoleSnapshot,
    FanoutLabel,
    PersistentFanoutLabelCache,
    SimulatorFanoutAdapter,
    candidate_score,
    cartpole_recoverability_from_state,
)

__all__ = [
    "AffordanceSelection",
    "BeliefAffordanceController",
    "BeliefDynamicsModel",
    "BeliefPlannerConfig",
    "ContinuousCartPoleSnapshot",
    "EvaluationEpisodeFixture",
    "FanoutLabel",
    "PersistentFanoutLabelCache",
    "PlanningBeliefState",
    "SimulatorFanoutAdapter",
    "build_planner_probe_dataset",
    "candidate_score",
    "cartpole_recoverability_from_state",
    "choose_affordance_action",
    "evaluate_belief_affordance_fixtures",
    "evaluate_belief_planner",
    "fit_belief_dynamics_model",
    "generate_candidate_actions",
    "mean_to_action",
    "plan_cem_action",
    "planner_trust_from_context",
    "replay_batch_mean_error",
    "should_stop_belief_planner_plateau",
    "train_belief_affordance_controller",
    "train_belief_planner",
    "update_belief_dynamics_from_replay",
    "update_planner_prior",
]
