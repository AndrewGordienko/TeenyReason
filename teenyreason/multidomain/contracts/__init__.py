"""Shared multidomain evidence and handoff contracts."""

from .decision_gate import DecisionGateInput, DecisionGateResult, decision_gate_payload, evaluate_decision_delta_gate
from .decision_local import DecisionLocalBelief, attach_decision_local_blocks, decision_local_row
from .decision_utility import DecisionUtilityContract, attach_decision_utility_blocks, decision_utility_row
from .evidence import DomainFeatureSpec, attach_standard_domain_blocks
from .handoff import BeliefHandoffContract, attach_belief_handoff_blocks, belief_handoff_row, rate_distortion_row
from .handoff_repair import attach_handoff_repair_blocks, handoff_repair_row

__all__ = [
    "BeliefHandoffContract",
    "DecisionGateInput",
    "DecisionGateResult",
    "DecisionLocalBelief",
    "DecisionUtilityContract",
    "DomainFeatureSpec",
    "attach_belief_handoff_blocks",
    "attach_decision_local_blocks",
    "attach_decision_utility_blocks",
    "attach_handoff_repair_blocks",
    "attach_standard_domain_blocks",
    "belief_handoff_row",
    "decision_gate_payload",
    "decision_local_row",
    "decision_utility_row",
    "evaluate_decision_delta_gate",
    "handoff_repair_row",
    "rate_distortion_row",
]
