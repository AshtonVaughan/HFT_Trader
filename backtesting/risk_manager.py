"""
Advanced Risk Management

Implements sophisticated risk management:
- Kelly Criterion for optimal position sizing
- Value-at-Risk (VaR) and Conditional VaR
- Maximum drawdown limits
- Portfolio-level risk management
- Dynamic position sizing based on recent performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import logger


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_drawdown_pct: float = 0.15  # Maximum 15% drawdown
    max_position_size: float = 0.1  # Maximum 10% of capital per position
    max_leverage: float = 1.0  # No leverage by default
    max_correlation: float = 0.7  # Max correlation between positions
    var_confidence: float = 0.95  # VaR confidence level
    max_var_pct: float = 0.05  # Maximum 5% daily VaR


class RiskManager:
    """
    Advanced risk management system.
    """

    def __init__(
        self,
        initial_capital: float,
        risk_limits: Optional[RiskLimits] = None
    ):
        """
        Args:
            initial_capital: Starting capital
            risk_limits: Risk limit configuration
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()

        # Track performance
        self.equity_curve = [initial_capital]
        self.returns = []
        self.positions = {}  # Current open positions
        self.trade_history = []

        # Risk metrics
        self.peak_capital = initial_capital
        self.current_drawdown = 0

        logger.info(f"Risk Manager initialized with ${initial_capital:,.2f}")
        logger.info(f"  Max Drawdown: {self.risk_limits.max_drawdown_pct:.1%}")
        logger.info(f"  Max Position Size: {self.risk_limits.max_position_size:.1%}")

    def calculate_position_size(
        self,
        signal_strength: float,
        current_price: float,
        stop_loss_price: float,
        method: str = 'kelly'
    ) -> float:
        """
        Calculate optimal position size.

        Args:
            signal_strength: Confidence in signal (0-1)
            current_price: Current price
            stop_loss_price: Stop loss price
            method: 'kelly', 'fixed_pct', 'risk_parity'

        Returns:
            Position size in units
        """
        if method == 'kelly':
            return self._kelly_criterion_size(signal_strength, current_price, stop_loss_price)
        elif method == 'fixed_pct':
            return self._fixed_pct_size(current_price, stop_loss_price)
        elif method == 'risk_parity':
            return self._risk_parity_size(current_price, stop_loss_price)
        else:
            raise ValueError(f"Unknown sizing method: {method}")

    def _kelly_criterion_size(
        self,
        win_prob: float,
        current_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Kelly Criterion for optimal position sizing.

        Formula: f = (p * b - (1 - p)) / b
        where:
            f = fraction of capital to bet
            p = win probability
            b = win/loss ratio (take_profit / stop_loss)
        """
        # Calculate risk per unit
        risk_per_unit = abs(current_price - stop_loss_price)

        # Assume 2:1 reward/risk ratio
        reward_risk_ratio = 2.0

        # Kelly fraction
        kelly_fraction = (win_prob * reward_risk_ratio - (1 - win_prob)) / reward_risk_ratio

        # Half-Kelly for safety (less aggressive)
        kelly_fraction = kelly_fraction * 0.5

        # Limit to max position size
        kelly_fraction = min(kelly_fraction, self.risk_limits.max_position_size)
        kelly_fraction = max(kelly_fraction, 0)  # No negative positions from Kelly

        # Calculate position size
        capital_to_risk = self.current_capital * kelly_fraction
        position_size = capital_to_risk / risk_per_unit

        return position_size

    def _fixed_pct_size(self, current_price: float, stop_loss_price: float) -> float:
        """Fixed percentage risk per trade (e.g., 2%)."""
        risk_pct = 0.02  # Risk 2% per trade
        risk_amount = self.current_capital * risk_pct

        risk_per_unit = abs(current_price - stop_loss_price)
        position_size = risk_amount / risk_per_unit

        return position_size

    def _risk_parity_size(self, current_price: float, stop_loss_price: float) -> float:
        """Risk parity - equal risk contribution across positions."""
        # If we have multiple positions, equalize risk
        if len(self.positions) == 0:
            return self._fixed_pct_size(current_price, stop_loss_price)

        # Target: each position contributes equal risk
        target_risk_per_position = 0.02 / max(len(self.positions) + 1, 1)
        risk_amount = self.current_capital * target_risk_per_position

        risk_per_unit = abs(current_price - stop_loss_price)
        position_size = risk_amount / risk_per_unit

        return position_size

    def calculate_var(self, confidence: float = 0.95, time_horizon: int = 1) -> float:
        """
        Calculate Value-at-Risk.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            time_horizon: Time horizon in days

        Returns:
            VaR in dollars
        """
        if len(self.returns) < 30:
            # Not enough data, use conservative estimate
            return self.current_capital * 0.02

        # Historical VaR
        returns_array = np.array(self.returns)

        # Adjust for time horizon
        scaled_returns = returns_array * np.sqrt(time_horizon)

        # VaR is the negative of the confidence percentile
        var_pct = np.percentile(scaled_returns, (1 - confidence) * 100)
        var_dollars = abs(var_pct * self.current_capital)

        return var_dollars

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        CVaR is the expected loss given that we exceed VaR.
        """
        if len(self.returns) < 30:
            return self.current_capital * 0.03

        returns_array = np.array(self.returns)

        # Find VaR threshold
        var_threshold = np.percentile(returns_array, (1 - confidence) * 100)

        # CVaR is mean of returns below VaR
        tail_returns = returns_array[returns_array <= var_threshold]

        if len(tail_returns) == 0:
            return 0

        cvar_pct = tail_returns.mean()
        cvar_dollars = abs(cvar_pct * self.current_capital)

        return cvar_dollars

    def check_risk_limits(self, proposed_position_size: float, current_price: float) -> Tuple[bool, str]:
        """
        Check if proposed trade violates risk limits.

        Returns:
            (is_allowed, reason)
        """
        # Check drawdown limit
        if self.current_drawdown >= self.risk_limits.max_drawdown_pct:
            return False, f"Maximum drawdown reached: {self.current_drawdown:.1%}"

        # Check position size limit
        position_value = proposed_position_size * current_price
        position_pct = position_value / self.current_capital

        if position_pct > self.risk_limits.max_position_size:
            return False, f"Position size {position_pct:.1%} exceeds limit {self.risk_limits.max_position_size:.1%}"

        # Check VaR limit
        current_var = self.calculate_var()
        if current_var / self.current_capital > self.risk_limits.max_var_pct:
            return False, f"VaR {current_var/self.current_capital:.1%} exceeds limit {self.risk_limits.max_var_pct:.1%}"

        # Check total exposure
        total_exposure = sum([pos['size'] * pos['price'] for pos in self.positions.values()])
        total_exposure += position_value

        if total_exposure / self.current_capital > self.risk_limits.max_leverage:
            return False, f"Total leverage {total_exposure/self.current_capital:.1f}x exceeds limit {self.risk_limits.max_leverage:.1f}x"

        return True, "OK"

    def update_position(self, symbol: str, size: float, price: float, direction: int):
        """Update or open position."""
        self.positions[symbol] = {
            'size': size,
            'price': price,
            'direction': direction,  # 1 for long, -1 for short
            'timestamp': pd.Timestamp.now()
        }

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close position and calculate P&L."""
        if symbol not in self.positions:
            return 0

        pos = self.positions[symbol]

        # Calculate P&L
        if pos['direction'] == 1:  # Long
            pnl = (exit_price - pos['price']) * pos['size']
        else:  # Short
            pnl = (pos['price'] - exit_price) * pos['size']

        # Update capital
        self.current_capital += pnl

        # Update equity curve and returns
        self.equity_curve.append(self.current_capital)
        self.returns.append(pnl / self.current_capital)

        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'entry_price': pos['price'],
            'exit_price': exit_price,
            'size': pos['size'],
            'direction': pos['direction'],
            'pnl': pnl,
            'timestamp': pd.Timestamp.now()
        })

        # Remove position
        del self.positions[symbol]

        return pnl

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        metrics = {
            'current_capital': self.current_capital,
            'current_drawdown': self.current_drawdown,
            'peak_capital': self.peak_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'num_positions': len(self.positions),
            'total_exposure': sum([pos['size'] * pos['price'] for pos in self.positions.values()]),
        }

        if len(self.returns) > 0:
            metrics['var_95'] = self.calculate_var(0.95)
            metrics['cvar_95'] = self.calculate_cvar(0.95)
            metrics['sharpe_ratio'] = self._calculate_sharpe()
            metrics['sortino_ratio'] = self._calculate_sortino()
            metrics['calmar_ratio'] = self._calculate_calmar()

        return metrics

    def _calculate_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(self.returns) < 2:
            return 0

        returns_array = np.array(self.returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe

    def _calculate_sortino(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)."""
        if len(self.returns) < 2:
            return 0

        returns_array = np.array(self.returns)
        excess_returns = returns_array - (risk_free_rate / 252)

        downside_returns = returns_array[returns_array < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0

        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return sortino

    def _calculate_calmar(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        if self.current_drawdown == 0 or len(self.returns) == 0:
            return 0

        annual_return = np.mean(self.returns) * 252
        calmar = annual_return / self.current_drawdown

        return calmar

    def should_reduce_risk(self) -> bool:
        """Check if we should reduce risk based on recent performance."""
        # Reduce risk if drawdown is high
        if self.current_drawdown > self.risk_limits.max_drawdown_pct * 0.7:
            logger.warning(f"High drawdown {self.current_drawdown:.1%} - reducing risk")
            return True

        # Reduce risk if recent returns are poor
        if len(self.returns) >= 20:
            recent_returns = self.returns[-20:]
            if np.mean(recent_returns) < -0.001:  # Losing on average
                logger.warning("Recent poor performance - reducing risk")
                return True

        return False

    def get_dynamic_position_multiplier(self) -> float:
        """
        Get position size multiplier based on recent performance.

        Returns 0.5-1.5 multiplier.
        """
        if len(self.returns) < 20:
            return 1.0

        # Use recent performance to scale position sizes
        recent_returns = self.returns[-20:]
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)

        if volatility == 0:
            return 1.0

        # Sharpe-based multiplier
        recent_sharpe = (avg_return / volatility) * np.sqrt(252)

        # Scale between 0.5 and 1.5
        multiplier = np.clip(1.0 + recent_sharpe * 0.2, 0.5, 1.5)

        return multiplier


if __name__ == '__main__':
    # Test risk manager
    print("\n" + "="*80)
    print("Risk Manager Test")
    print("="*80)

    # Initialize
    risk_manager = RiskManager(initial_capital=10000)

    # Test position sizing
    print("\n1. Kelly Criterion Position Sizing:")
    size = risk_manager.calculate_position_size(
        signal_strength=0.6,  # 60% win probability
        current_price=1.1000,
        stop_loss_price=1.0950,
        method='kelly'
    )
    print(f"   Position size: {size:.2f} units")
    print(f"   Position value: ${size * 1.1000:,.2f}")

    # Simulate some trades
    print("\n2. Simulating trades...")

    # Open position
    risk_manager.update_position('EURUSD', size=1000, price=1.1000, direction=1)
    print(f"   Opened long position: 1000 units @ 1.1000")

    # Close with profit
    pnl = risk_manager.close_position('EURUSD', exit_price=1.1050)
    print(f"   Closed with P&L: ${pnl:.2f}")

    # Check risk metrics
    print("\n3. Risk Metrics:")
    metrics = risk_manager.get_risk_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'ratio' in key or 'return' in key or 'drawdown' in key:
                print(f"   {key}: {value:.2%}")
            else:
                print(f"   {key}: ${value:,.2f}")
        else:
            print(f"   {key}: {value}")

    # Test risk limits
    print("\n4. Testing Risk Limits:")
    allowed, reason = risk_manager.check_risk_limits(
        proposed_position_size=5000,
        current_price=1.1000
    )
    print(f"   Large position allowed: {allowed}")
    if not allowed:
        print(f"   Reason: {reason}")
