"""PhotonShield V3 Adaptive Compute Module."""

from module_07_adaptive_compute.action_space import (
    ACTIONS,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    get_action_space,
    action_to_index,
    index_to_action,
)
from module_07_adaptive_compute.state_encoder import AdaptiveComputeStateEncoder, STATE_DIM, STATE_FEATURE_NAMES
from module_07_adaptive_compute.rule_scheduler import RuleBasedDiffusionScheduler
from module_07_adaptive_compute.supervised_scheduler import SupervisedDiffusionScheduler
from module_07_adaptive_compute.scheduler_diagnostics import compute_policy_metrics

from module_07_adaptive_compute.ppo_policy import ActorCriticNetwork
from module_07_adaptive_compute.ppo_reward import PPORewardCalculator
from module_07_adaptive_compute.ppo_environment import RadarAdaptiveComputeEnv
from module_07_adaptive_compute.ppo_agent import PPOAgent, PPORolloutBuffer
from module_07_adaptive_compute.ppo_diagnostics import compute_ppo_diagnostics

__all__ = [
    "ACTIONS",
    "ACTION_TO_IDX",
    "IDX_TO_ACTION",
    "get_action_space",
    "action_to_index",
    "index_to_action",
    "AdaptiveComputeStateEncoder",
    "STATE_DIM",
    "STATE_FEATURE_NAMES",
    "RuleBasedDiffusionScheduler",
    "SupervisedDiffusionScheduler",
    "compute_policy_metrics",
    "ActorCriticNetwork",
    "PPORewardCalculator",
    "RadarAdaptiveComputeEnv",
    "PPOAgent",
    "PPORolloutBuffer",
    "compute_ppo_diagnostics",
]
