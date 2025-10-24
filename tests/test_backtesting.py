"""
Tests for backtesting and risk management.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_engine import BacktestEngine, Trade
from backtesting.risk_manager import RiskManager, RiskLimits


class TestBacktestEngine:
    """Test backtesting engine."""

    def setup_method(self):
        """Setup test data."""
        n = 1000
        dates = pd.date_range('2024-01-01', periods=n, freq='1H')

        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(n).cumsum() + 1.1000,
            'high': np.random.randn(n).cumsum() + 1.1010,
            'low': np.random.randn(n).cumsum() + 1.0990,
            'close': np.random.randn(n).cumsum() + 1.1000,
            'atr': np.random.uniform(0.0001, 0.001, n),
            'spread_mean': np.random.uniform(0.00005, 0.0001, n)
        }, index=dates)

        # Fix OHLC consistency
        self.data['high'] = self.data[['open', 'close']].max(axis=1) + abs(np.random.randn(n)) * 0.0001
        self.data['low'] = self.data[['open', 'close']].min(axis=1) - abs(np.random.randn(n)) * 0.0001

    def test_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine(initial_capital=10000)
        assert engine.initial_capital == 10000
        assert engine.pip_value == 0.0001

    def test_backtest_no_signals(self):
        """Test backtest with no trading signals."""
        engine = BacktestEngine(initial_capital=10000)

        signals = pd.Series(0, index=self.data.index)  # All hold

        results = engine.backtest(self.data, signals)

        assert results['metrics']['total_trades'] == 0
        assert results['metrics']['total_return'] == 0

    def test_backtest_with_signals(self):
        """Test backtest with trading signals."""
        engine = BacktestEngine(initial_capital=10000)

        # Random signals
        signals = pd.Series(np.random.choice([0, 0, 0, 1, -1], len(self.data)), index=self.data.index)

        results = engine.backtest(self.data, signals)

        # Check metrics exist
        assert 'total_return' in results['metrics']
        assert 'total_trades' in results['metrics']
        assert 'win_rate' in results['metrics']
        assert 'sharpe_ratio' in results['metrics']
        assert 'max_drawdown' in results['metrics']

        # Check trades were executed
        assert len(results['trades']) > 0

    def test_transaction_costs(self):
        """Test transaction costs are applied."""
        # Engine with high costs
        engine_high_cost = BacktestEngine(
            initial_capital=10000,
            spread_pips=2.0,
            slippage_pips=1.0
        )

        # Engine with low costs
        engine_low_cost = BacktestEngine(
            initial_capital=10000,
            spread_pips=0.1,
            slippage_pips=0.1
        )

        signals = pd.Series([1, 0, -1, 0] * 250, index=self.data.index)

        results_high = engine_high_cost.backtest(self.data, signals)
        results_low = engine_low_cost.backtest(self.data, signals)

        # High cost should have worse return
        assert results_high['metrics']['total_return'] < results_low['metrics']['total_return']


class TestRiskManager:
    """Test risk manager."""

    def test_initialization(self):
        """Test initialization."""
        rm = RiskManager(initial_capital=10000)
        assert rm.current_capital == 10000
        assert rm.initial_capital == 10000

    def test_kelly_criterion_sizing(self):
        """Test Kelly criterion position sizing."""
        rm = RiskManager(initial_capital=10000)

        size = rm.calculate_position_size(
            signal_strength=0.6,  # 60% win probability
            current_price=1.1000,
            stop_loss_price=1.0950,
            method='kelly'
        )

        assert size > 0
        assert isinstance(size, float)

    def test_fixed_pct_sizing(self):
        """Test fixed percentage sizing."""
        rm = RiskManager(initial_capital=10000)

        size = rm.calculate_position_size(
            signal_strength=0.6,
            current_price=1.1000,
            stop_loss_price=1.0950,
            method='fixed_pct'
        )

        assert size > 0

    def test_var_calculation(self):
        """Test VaR calculation."""
        rm = RiskManager(initial_capital=10000)

        # Simulate some returns
        for i in range(100):
            rm.returns.append(np.random.randn() * 0.001)

        var = rm.calculate_var(confidence=0.95)

        assert var > 0
        assert isinstance(var, float)

    def test_cvar_calculation(self):
        """Test CVaR calculation."""
        rm = RiskManager(initial_capital=10000)

        # Simulate returns
        for i in range(100):
            rm.returns.append(np.random.randn() * 0.001)

        cvar = rm.calculate_cvar(confidence=0.95)

        assert cvar > 0
        assert cvar >= rm.calculate_var(confidence=0.95)  # CVaR >= VaR

    def test_risk_limits(self):
        """Test risk limit checking."""
        limits = RiskLimits(
            max_drawdown_pct=0.10,
            max_position_size=0.05
        )

        rm = RiskManager(initial_capital=10000, risk_limits=limits)

        # Small position should be allowed
        allowed, reason = rm.check_risk_limits(
            proposed_position_size=100,
            current_price=1.1000
        )
        assert allowed

        # Very large position should be rejected
        allowed, reason = rm.check_risk_limits(
            proposed_position_size=10000,
            current_price=1.1000
        )
        assert not allowed

    def test_position_management(self):
        """Test position updates and closing."""
        rm = RiskManager(initial_capital=10000)

        # Open position
        rm.update_position('EURUSD', size=1000, price=1.1000, direction=1)
        assert 'EURUSD' in rm.positions

        # Close with profit
        pnl = rm.close_position('EURUSD', exit_price=1.1050)

        assert pnl > 0
        assert 'EURUSD' not in rm.positions
        assert rm.current_capital > rm.initial_capital

    def test_drawdown_tracking(self):
        """Test drawdown tracking."""
        rm = RiskManager(initial_capital=10000)

        # Simulate winning trade
        rm.update_position('EURUSD', 1000, 1.1000, 1)
        rm.close_position('EURUSD', 1.1050)

        assert rm.current_drawdown == 0  # At peak

        # Simulate losing trade
        rm.update_position('EURUSD', 1000, 1.1000, 1)
        rm.close_position('EURUSD', 1.0950)

        assert rm.current_drawdown > 0


class TestRiskMetrics:
    """Test risk metric calculations."""

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        rm = RiskManager(initial_capital=10000)

        # Simulate consistent positive returns
        for _ in range(100):
            rm.returns.append(0.001 + np.random.randn() * 0.0001)

        metrics = rm.get_risk_metrics()

        assert 'sharpe_ratio' in metrics
        assert metrics['sharpe_ratio'] > 0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        rm = RiskManager(initial_capital=10000)

        # Simulate returns
        for _ in range(100):
            rm.returns.append(np.random.randn() * 0.001)

        metrics = rm.get_risk_metrics()

        assert 'sortino_ratio' in metrics

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        rm = RiskManager(initial_capital=10000)

        # Simulate trades
        rm.update_position('EURUSD', 1000, 1.1000, 1)
        rm.close_position('EURUSD', 1.0900)  # Loss to create drawdown

        rm.update_position('EURUSD', 1000, 1.1000, 1)
        rm.close_position('EURUSD', 1.1050)  # Profit

        metrics = rm.get_risk_metrics()

        assert 'calmar_ratio' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
