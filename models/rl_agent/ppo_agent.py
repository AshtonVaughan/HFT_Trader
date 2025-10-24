"""
PPO Trading Agent

Learns optimal entry/exit timing using Proximal Policy Optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class Actor(nn.Module):
    """
    Policy network (actor).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_actions: int = 3):
        """
        Args:
            state_dim: State dimension
            hidden_dim: Hidden layer dimension
            num_actions: Number of actions (enter_long, enter_short, hold)
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)

        Returns:
            action_logits: (batch, num_actions)
        """
        return self.network(state)

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    """
    Value network (critic).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)

        Returns:
            value: (batch, 1) state value
        """
        return self.network(state)


class PPOAgent:
    """
    PPO trading agent.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_actions: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        device: str = 'cuda'
    ):
        """
        Args:
            state_dim: State dimension
            hidden_dim: Hidden layer dimension
            num_actions: Number of actions
            lr: Learning rate
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            device: Device (cuda/cpu)
        """
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # Create networks
        self.actor = Actor(state_dim, hidden_dim, num_actions).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        logger.info(f"PPOAgent initialized: {state_dim} state_dim, {num_actions} actions")

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Select action from policy.

        Args:
            state: (state_dim,) or (batch, state_dim)
            deterministic: If True, select argmax action

        Returns:
            action: int
            log_prob: tensor
        """
        with torch.no_grad():
            probs = self.actor.get_action_probs(state)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            log_prob = torch.log(probs[..., action] + 1e-8)

        return action.item() if action.dim() == 0 else action, log_prob

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[float, float]:
        """
        PPO update.

        Args:
            states: (batch, state_dim)
            actions: (batch,)
            old_log_probs: (batch,)
            returns: (batch,)
            advantages: (batch,)

        Returns:
            actor_loss, critic_loss
        """
        # Critic loss
        values = self.critic(states).squeeze()
        critic_loss = F.mse_loss(values, returns)

        # Actor loss (PPO clipped objective)
        action_probs = self.actor.get_action_probs(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save(self, path: str):
        """Save agent."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
        logger.info(f"Saved PPOAgent to {path}")

    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        logger.info(f"Loaded PPOAgent from {path}")


if __name__ == '__main__':
    # Test PPO agent
    state_dim = 100
    num_actions = 3

    agent = PPOAgent(state_dim=state_dim, hidden_dim=128, num_actions=num_actions, device='cpu')

    # Test action selection
    state = torch.randn(state_dim)
    action, log_prob = agent.select_action(state)
    print(f"\nSelected action: {action}")
    print(f"Log prob: {log_prob}")

    # Test update
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, num_actions, (batch_size,))
    old_log_probs = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    advantages = torch.randn(batch_size)

    actor_loss, critic_loss = agent.update(states, actions, old_log_probs, returns, advantages)
    print(f"\nActor loss: {actor_loss:.4f}")
    print(f"Critic loss: {critic_loss:.4f}")
