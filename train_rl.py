"""
Reinforcement Learning Training Pipeline

Trains PPO agent on top of ensemble predictions to optimize trading decisions.
The RL agent learns:
- When to enter/exit positions
- Position sizing
- Risk management
"""

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from dataclasses import dataclass

from models.rl_agent.ppo_agent import PPOAgent
from models.regime_detector.transformer_detector import RegimeDetector
from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
from models.meta_learner.attention_meta_learner import AttentionMetaLearner, EnsemblePredictor
from utils.gpu_utils import get_device
from utils.logger import setup_logger, logger


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class TradingEnvironment:
    """
    Simulated trading environment for RL training.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        ensemble: EnsemblePredictor,
        initial_capital: float = 10000,
        max_position: float = 1.0,
        spread_pips: float = 0.5,
        slippage_pips: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Args:
            data: Market data
            ensemble: Trained ensemble for predictions
            initial_capital: Starting capital
            max_position: Maximum position size (fraction of capital)
            spread_pips: Bid-ask spread in pips
            slippage_pips: Slippage in pips
            device: Device for ensemble
        """
        self.data = data
        self.ensemble = ensemble
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.device = device

        self.pip_value = 0.0001

        # State
        self.current_step = 0
        self.capital = initial_capital
        self.position = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0
        self.pnl = 0
        self.trade_count = 0

        self.ensemble.eval()

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.pnl = 0
        self.trade_count = 0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Get current state.

        State includes:
        - Ensemble predictions (direction, magnitude, confidence)
        - Current position (0, 1, -1)
        - Unrealized P&L
        - Recent price action
        """
        if self.current_step >= len(self.data):
            return np.zeros(20)  # Terminal state

        row = self.data.iloc[self.current_step]

        # Get ensemble prediction
        with torch.no_grad():
            # Prepare features (simplified - in practice you'd use full feature vector)
            features = torch.FloatTensor([
                row.get('close', 0),
                row.get('rsi_14', 0),
                row.get('atr_14', 0),
                # Add more features...
            ]).unsqueeze(0).unsqueeze(0).to(self.device)

            outputs = self.ensemble(features)

            direction = outputs['direction_logits'].argmax().item()  # 0 or 1
            magnitude = outputs['magnitude'].item()
            confidence = outputs['confidence'].item()

        # Current state features
        state = [
            # Ensemble outputs
            float(direction),
            magnitude,
            confidence,

            # Position info
            float(self.position),  # -1, 0, or 1
            self.pnl / self.initial_capital,  # Normalized P&L

            # Price features (recent returns)
            row.get('return_1', 0),
            row.get('return_5', 0),
            row.get('return_10', 0),

            # Volatility
            row.get('atr_14', 0) / row.get('close', 1),

            # Market regime
            row.get('regime_trending_up', 0),
            row.get('regime_trending_down', 0),
            row.get('regime_ranging', 0),
            row.get('regime_volatile', 0),

            # Technical indicators
            row.get('rsi_14', 50) / 100,
            row.get('macd', 0),
            row.get('bb_position', 0),

            # Volume
            row.get('volume_ratio_20', 1),

            # Session
            row.get('session_london', 0),
            row.get('session_ny', 0),

            # Account info
            self.capital / self.initial_capital,
        ]

        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.

        Actions:
        0: Hold/Close position
        1: Enter long
        2: Enter short

        Returns:
            next_state, reward, done, info
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']

        reward = 0
        info = {}

        # Transaction costs
        transaction_cost = (self.spread_pips + self.slippage_pips) * self.pip_value

        # Execute action
        if action == 1 and self.position != 1:  # Enter long
            # Close existing short if any
            if self.position == -1:
                pnl = (self.entry_price - current_price) * abs(self.position) * self.capital
                self.capital += pnl
                reward += pnl / self.initial_capital  # Normalized reward

            # Enter long
            self.position = 1
            self.entry_price = current_price + transaction_cost
            self.trade_count += 1

        elif action == 2 and self.position != -1:  # Enter short
            # Close existing long if any
            if self.position == 1:
                pnl = (current_price - self.entry_price) * abs(self.position) * self.capital
                self.capital += pnl
                reward += pnl / self.initial_capital

            # Enter short
            self.position = -1
            self.entry_price = current_price - transaction_cost
            self.trade_count += 1

        elif action == 0 and self.position != 0:  # Close position
            if self.position == 1:
                pnl = (current_price - self.entry_price) * abs(self.position) * self.capital
            else:
                pnl = (self.entry_price - current_price) * abs(self.position) * self.capital

            self.capital += pnl
            reward += pnl / self.initial_capital
            self.position = 0
            self.entry_price = 0

        # Update unrealized P&L if in position
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (next_price - self.entry_price) * abs(self.position) * self.capital
            else:
                unrealized_pnl = (self.entry_price - next_price) * abs(self.position) * self.capital

            # Add small reward for unrealized gains (encourages holding winners)
            reward += unrealized_pnl / self.initial_capital * 0.1

        # Penalty for excessive trading
        reward -= 0.001  # Small penalty per step to encourage selective trading

        # Penalty for large drawdown
        if self.capital < self.initial_capital * 0.9:
            reward -= 0.01

        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= len(self.data) - 1

        info = {
            'capital': self.capital,
            'position': self.position,
            'trade_count': self.trade_count
        }

        return next_state, reward, done, info


class RLTrainer:
    """
    PPO training for trading agent.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()

        logger.info(f"RL Trainer initialized on device: {self.device}")

    def train(self):
        """Train RL agent."""
        logger.info("="*80)
        logger.info("STARTING RL TRAINING PIPELINE")
        logger.info("="*80)

        # 1. Load trained ensemble
        logger.info("\n1. Loading trained ensemble...")
        ensemble = self._load_ensemble()

        # 2. Load data
        logger.info("\n2. Loading data...")
        train_df, val_df = self._load_data()

        # 3. Create PPO agent
        logger.info("\n3. Creating PPO agent...")
        state_dim = 20  # Based on _get_state()
        agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=256,
            num_actions=3,  # Hold, Long, Short
            lr=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            device=self.device
        )

        # 4. Train
        logger.info("\n4. Training PPO agent...")
        self._train_ppo(agent, ensemble, train_df, val_df)

        # 5. Evaluate
        logger.info("\n5. Evaluating RL agent...")
        metrics = self._evaluate_ppo(agent, ensemble, val_df)

        # 6. Save
        logger.info("\n6. Saving RL agent...")
        self._save_agent(agent)

        logger.info("\n" + "="*80)
        logger.info("RL TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nFinal metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    def _load_ensemble(self) -> EnsemblePredictor:
        """Load pretrained ensemble."""
        checkpoint_dir = Path('models/checkpoints')

        # Load config for model architecture
        feature_dim = 100  # This should match your actual feature dimension

        regime_detector = RegimeDetector(
            input_size=feature_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        ).to(self.device)

        lstm = LSTMPredictor(input_size=feature_dim, hidden_size=256, num_layers=3).to(self.device)
        gru = GRUPredictor(input_size=feature_dim, hidden_size=256, num_layers=3).to(self.device)
        cnn_lstm = CNNLSTMPredictor(input_size=feature_dim, cnn_channels=[64, 128, 256], lstm_hidden_size=256).to(self.device)

        specialized_models = [lstm, gru, cnn_lstm]

        meta_learner = AttentionMetaLearner(num_models=3, embedding_dim=128).to(self.device)

        ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner).to(self.device)

        # Load weights
        regime_detector.load_state_dict(torch.load(checkpoint_dir / 'regime_detector.pth'))
        lstm.load_state_dict(torch.load(checkpoint_dir / 'lstm.pth'))
        gru.load_state_dict(torch.load(checkpoint_dir / 'gru.pth'))
        cnn_lstm.load_state_dict(torch.load(checkpoint_dir / 'cnn_lstm.pth'))
        meta_learner.load_state_dict(torch.load(checkpoint_dir / 'meta_learner.pth'))

        logger.info("  Ensemble loaded successfully")

        return ensemble

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load preprocessed data."""
        data_dir = Path(self.config.get('data_collection', {}).get('processed_data_dir', 'processed_data'))

        train_df = pd.read_parquet(data_dir / 'train.parquet')
        val_df = pd.read_parquet(data_dir / 'val.parquet')

        logger.info(f"  Train: {len(train_df):,} samples")
        logger.info(f"  Val: {len(val_df):,} samples")

        return train_df, val_df

    def _train_ppo(
        self,
        agent: PPOAgent,
        ensemble: EnsemblePredictor,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_episodes: int = 100
    ):
        """Train PPO agent."""
        env = TradingEnvironment(
            data=train_df,
            ensemble=ensemble,
            initial_capital=10000,
            device=self.device
        )

        best_return = -float('inf')

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            experiences = []

            done = False

            while not done:
                # Get action from agent
                action, log_prob, value = agent.get_action(state)

                # Take action
                next_state, reward, done, info = env.step(action)

                # Store experience
                experiences.append(Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    value=value
                ))

                episode_reward += reward
                episode_steps += 1

                state = next_state

            # Update agent
            if len(experiences) > 0:
                agent.update(experiences)

            # Log progress
            total_return = (env.capital - env.initial_capital) / env.initial_capital
            logger.info(f"  Episode {episode+1}/{num_episodes}: "
                       f"Return={total_return:.2%}, "
                       f"Trades={env.trade_count}, "
                       f"Steps={episode_steps}")

            # Validate
            if (episode + 1) % 10 == 0:
                val_metrics = self._evaluate_ppo(agent, ensemble, val_df)
                logger.info(f"    Validation Return: {val_metrics['total_return']:.2%}")

                if val_metrics['total_return'] > best_return:
                    best_return = val_metrics['total_return']
                    # Save best model
                    checkpoint_dir = Path('models/checkpoints')
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    agent.save(str(checkpoint_dir / 'ppo_agent_best.pth'))
                    logger.info(f"    New best model saved! Return: {best_return:.2%}")

    def _evaluate_ppo(
        self,
        agent: PPOAgent,
        ensemble: EnsemblePredictor,
        data: pd.DataFrame
    ) -> Dict:
        """Evaluate PPO agent."""
        env = TradingEnvironment(
            data=data,
            ensemble=ensemble,
            initial_capital=10000,
            device=self.device
        )

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _, _ = agent.get_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

        total_return = (env.capital - env.initial_capital) / env.initial_capital

        metrics = {
            'total_return': total_return,
            'total_reward': total_reward,
            'final_capital': env.capital,
            'trade_count': env.trade_count
        }

        return metrics

    def _save_agent(self, agent: PPOAgent):
        """Save PPO agent."""
        checkpoint_dir = Path('models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        agent.save(str(checkpoint_dir / 'ppo_agent.pth'))
        logger.info(f"  PPO agent saved to {checkpoint_dir}/ppo_agent.pth")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for HFT trading")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    setup_logger(
        level='INFO',
        log_file='hft_rl_train.log'
    )

    # Train
    trainer = RLTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
