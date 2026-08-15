"""Actor-Critic Policy Architecture for PhotonShield V3.3 PPO.

Actor: 9 -> 64 -> 32 -> 4 (Tanh activations, Categorical distribution)
Critic: 9 -> 64 -> 32 -> 1 (Tanh activations, Value estimation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from module_07_adaptive_compute.action_space import ACTIONS, NUM_ACTIONS, IDX_TO_ACTION, ACTION_TO_IDX


class ActorCriticNetwork(nn.Module):
    """Compact Actor-Critic Network for discrete diffusion step selection."""

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 4,
        hidden1: int = 64,
        hidden2: int = 32,
    ) -> None:
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, action_dim),
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor and critic.

        Args:
            state: State tensor `[B, 9]`.

        Returns:
            Tuple of (action_logits `[B, 4]`, state_value `[B, 1]`).
        """
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate action and compute log probability, entropy, and value.

        Args:
            state: State tensor `[B, 9]`.
            action: Optional action indices `[B]`.
            deterministic: If True, takes argmax action.

        Returns:
            Tuple of (action `[B]`, log_prob `[B]`, entropy `[B]`, value `[B]`).
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value only."""
        return self.critic(state).squeeze(-1)

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)

    def load(self, path: Union[str, Path], device: torch.device) -> None:
        """Load model checkpoint."""
        self.load_state_dict(torch.load(path, map_location=device))
