"""Configuration for the cross-domain sample-efficiency suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...envs import CONTINUOUS_CARTPOLE_NAME
from .. import (
    AffordanceCrawlerSuiteConfig,
    BoardProbeBenchmarkConfig,
    CartPoleControllerBridgeConfig,
    CartPoleLatentMPCConfig,
    CartPoleMechanicsConfig,
    CartPolePlannerComparisonConfig,
    DecisionLocalCrawlerSuiteConfig,
    FamilyBridgeConfig,
    ImageProbeBenchmarkConfig,
    LanguageProbeBenchmarkConfig,
    LatentControlHandoffConfig,
    RealCausalAdapterConfig,
)


@dataclass(frozen=True)
class MultidomainSuiteConfig:
    """Configuration for the cross-domain sample-efficiency suite."""

    artifact_dir: Path = Path("artifacts")
    suite_name: str = "four_domain_belief_suite"
    rl_envs: tuple[str, ...] = (CONTINUOUS_CARTPOLE_NAME,)
    rl_seeds: tuple[int, ...] = (0, 1, 2)
    run_rl_benchmark: bool = True
    run_cartpole_mechanics_benchmark: bool = True
    run_cartpole_latent_mpc_benchmark: bool = True
    run_cartpole_planner_comparison: bool = True
    run_decision_local_crawler_benchmark: bool = True
    run_affordance_crawler_benchmark: bool = True
    run_family_bridge_benchmark: bool = True
    run_latent_handoff_benchmark: bool = True
    run_real_causal_benchmark: bool = True
    lunar_lander_randomize_physics: bool = True
    run_image_benchmark: bool = True
    run_language_benchmark: bool = True
    run_board_benchmark: bool = True
    image: ImageProbeBenchmarkConfig = field(default_factory=ImageProbeBenchmarkConfig)
    language: LanguageProbeBenchmarkConfig = field(default_factory=LanguageProbeBenchmarkConfig)
    board: BoardProbeBenchmarkConfig = field(default_factory=BoardProbeBenchmarkConfig)
    cartpole_mechanics: CartPoleMechanicsConfig = field(default_factory=CartPoleMechanicsConfig)
    cartpole_latent_mpc: CartPoleLatentMPCConfig = field(default_factory=CartPoleLatentMPCConfig)
    planner_comparison: CartPolePlannerComparisonConfig = field(default_factory=CartPolePlannerComparisonConfig)
    decision_local_crawler: DecisionLocalCrawlerSuiteConfig = field(default_factory=DecisionLocalCrawlerSuiteConfig)
    affordance_crawler: AffordanceCrawlerSuiteConfig = field(default_factory=AffordanceCrawlerSuiteConfig)
    cartpole_controller_bridge: CartPoleControllerBridgeConfig = field(default_factory=CartPoleControllerBridgeConfig)
    latent_handoff: LatentControlHandoffConfig = field(default_factory=LatentControlHandoffConfig)
    family_bridge: FamilyBridgeConfig = field(default_factory=FamilyBridgeConfig)
    real_causal: RealCausalAdapterConfig = field(default_factory=RealCausalAdapterConfig)
