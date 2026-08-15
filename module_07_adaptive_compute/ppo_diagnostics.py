"""Diagnostics, evaluation metrics, and ablation utilities for PhotonShield V3.3 PPO."""

from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

from module_07_adaptive_compute.action_space import ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION


def compute_ppo_diagnostics(
    ppo_actions: List[int],
    oracle_actions: List[int],
    rule_actions: List[int],
    ppo_objectives: List[float],
    oracle_objectives: List[float],
    rule_objectives: List[float],
    fixed_50_objectives: List[float],
) -> Dict[str, Any]:
    """Calculate RL-specific metrics: Action entropy, Regrets, Oracle Gap Closure, Confusion Matrix."""
    y_ppo = np.array(ppo_actions)
    y_orc = np.array(oracle_actions)
    y_rule = np.array(rule_actions)

    j_ppo = np.array(ppo_objectives)
    j_orc = np.array(oracle_objectives)
    j_rule = np.array(rule_objectives)
    j_50 = np.array(fixed_50_objectives)

    # 1. Action Distribution & Entropy
    action_counts = [int(np.sum(y_ppo == a)) for a in ACTIONS]
    total_samples = max(len(y_ppo), 1)
    action_probs = np.array([c / total_samples for c in action_counts], dtype=np.float32)
    # Shannon Entropy
    p_safe = np.where(action_probs > 0, action_probs, 1.0)
    entropy = -float(np.sum(action_probs * np.log(p_safe)))

    # 2. Step Statistics
    mean_steps = float(np.mean(y_ppo))
    median_steps = float(np.median(y_ppo))
    p95_steps = float(np.percentile(y_ppo, 95))
    compute_reduction = (1.0 - (mean_steps / 50.0)) * 100.0

    # 3. Regrets
    oracle_regret = float(np.mean(j_ppo - j_orc))
    rule_regret = float(np.mean(j_ppo - j_rule))

    # 4. Oracle Gap Closure: (J_50 - J_PPO) / (J_50 - J_Oracle)
    mean_j_50 = float(np.mean(j_50))
    mean_j_ppo = float(np.mean(j_ppo))
    mean_j_orc = float(np.mean(j_orc))
    denom = max(mean_j_50 - mean_j_orc, 1e-6)
    oracle_gap_closure_pct = float(np.clip((mean_j_50 - mean_j_ppo) / denom * 100.0, 0.0, 100.0))

    # 5. Agreement Accuracies
    oracle_agreement = float(accuracy_score(y_orc, y_ppo) * 100.0)
    rule_agreement = float(accuracy_score(y_rule, y_ppo) * 100.0)

    # 6. Confusion Matrix
    cm = confusion_matrix(y_orc, y_ppo, labels=ACTIONS)

    return {
        "mean_steps": mean_steps,
        "median_steps": median_steps,
        "p95_steps": p95_steps,
        "compute_reduction_pct": compute_reduction,
        "action_entropy": entropy,
        "action_distribution": {f"P_{a}": float(action_probs[i]) for i, a in enumerate(ACTIONS)},
        "oracle_regret": oracle_regret,
        "rule_regret": rule_regret,
        "oracle_gap_closure_pct": oracle_gap_closure_pct,
        "oracle_agreement_pct": oracle_agreement,
        "rule_agreement_pct": rule_agreement,
        "confusion_matrix": cm,
    }
