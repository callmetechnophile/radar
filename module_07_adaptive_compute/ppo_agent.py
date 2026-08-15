"""PPO Agent and Training Pipeline for PhotonShield V3.3.

Implements:
1. Rollout trajectory collection.
2. Generalized Advantage Estimation (GAE).
3. PPO clipped surrogate loss with entropy bonus and value clipping.
4. Mini-batch policy updates.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from module_07_adaptive_compute.ppo_policy import ActorCriticNetwork
from module_07_adaptive_compute.ppo_environment import RadarAdaptiveComputeEnv
from module_07_adaptive_compute.action_space import ACTIONS, IDX_TO_ACTION


class PPORolloutBuffer:
    """Stores experience tuples collected during rollout."""

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(float(reward))
        self.values.append(value.detach().cpu())
        self.dones.append(bool(done))

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    """Proximal Policy Optimization Agent."""

    def __init__(
        self,
        policy: ActorCriticNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.20,
        entropy_coef: float = 0.01,
        value_coef: float = 0.50,
        max_grad_norm: float = 0.50,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.policy = policy.to(device)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        self.buffer = PPORolloutBuffer()

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < T else 0.0)
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        if len(adv_tensor) > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return adv_tensor, ret_tensor

    def update(
        self,
        ppo_epochs: int = 4,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Perform PPO update over rollout buffer."""
        if len(self.buffer) == 0:
            return {}

        states = torch.cat(self.buffer.states, dim=0).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)
        values = [v.item() for v in self.buffer.values]

        advantages, returns = self.compute_gae(
            self.buffer.rewards, values, self.buffer.dones
        )

        n_samples = states.shape[0]
        indices = np.arange(n_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_updates = 0

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Evaluate actions with current policy
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    mb_states, action=mb_actions
                )

                # Ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped Policy Surrogate Loss
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(new_values, mb_returns)

                # Entropy Bonus
                entropy_loss = -entropy.mean()

                # Total Loss
                total_loss = policy_loss + (self.value_coef * value_loss) + (self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(total_updates, 1),
            "value_loss": total_value_loss / max(total_updates, 1),
            "entropy": total_entropy / max(total_updates, 1),
        }
