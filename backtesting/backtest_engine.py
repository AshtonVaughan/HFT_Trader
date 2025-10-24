"""
Realistic Backtesting Engine

Simulates trading with:
- Transaction costs (spread + slippage + commission)
- Stop loss and take profit
- Position sizing
- Realistic execution
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import logger


@dataclass
class Trade:
    """Single trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    outcome: str  # 'win' or 'loss'
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal'


class BacktestEngine:
    """
    Backtesting engine with realistic execution.
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,
        spread_pips: float = 0.5,
        slippage_pips: float = 0.3,
        commission_per_lot: float = 0.0,  # Zero commission (update if broker charges)
        max_spread_pips: float = 1.5,
        stop_loss_atr_multiple: float = 1.5,
        take_profit_atr_multiple: float = 2.5
    ):
        """
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as fraction (0.02 = 2%)
            spread_pips: Typical spread in pips
            slippage_pips: Slippage per trade in pips
            commission_per_lot: Commission per lot
            max_spread_pips: Don't trade if spread > this
            stop_loss_atr_multiple: SL in ATR multiples
            take_profit_atr_multiple: TP in ATR multiples
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_per_lot = commission_per_lot
        self.max_spread_pips = max_spread_pips
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        self.take_profit_atr_multiple = take_profit_atr_multiple

        # One pip for EURUSD = 0.0001
        self.pip_value = 0.0001

        logger.info(f"BacktestEngine initialized: ${initial_capital} capital, {risk_per_trade:.1%} risk")

    def backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series  # 1 = buy, -1 = sell, 0 = hold
    ) -> Dict:
        """
        Run backtest.

        Args:
            data: DataFrame with OHLCV + ATR
            signals: Trading signals

        Returns:
            Dictionary with metrics and trades
        """
        logger.info("Running backtest...")

        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [capital]

        for i in range(len(data)):
            current_bar = data.iloc[i]

            # Check if we're in a position
            if position is not None:
                # Update position
                position, trade = self._update_position(position, current_bar, i)

                if trade is not None:
                    trades.append(trade)
                    capital += trade.pnl
                    position = None

            # Check for new entry signal
            if position is None and i < len(signals) - 1:
                signal = signals.iloc[i]

                if signal != 0:
                    # Check spread condition
                    if 'spread_mean' in data.columns:
                        current_spread = current_bar['spread_mean'] / self.pip_value
                        if current_spread > self.max_spread_pips:
                            continue  # Skip trade if spread too wide

                    # Enter position
                    position = self._enter_position(current_bar, signal, capital, i)

            equity_curve.append(capital)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        logger.info(f"Backtest complete: {len(trades)} trades, {metrics['total_return']:.2%} return")

        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve
        }

    def _enter_position(self, bar: pd.Series, signal: int, capital: float, bar_idx: int) -> Dict:
        """Enter a new position."""
        entry_price = bar['close']
        atr = bar.get('atr', 0.001)  # Default ATR if not available

        # Calculate position size based on risk
        risk_amount = capital * self.risk_per_trade
        stop_distance = atr * self.stop_loss_atr_multiple

        # Account for spread and slippage
        total_cost = (self.spread_pips + self.slippage_pips) * self.pip_value

        if signal == 1:  # Long
            actual_entry = entry_price + total_cost
            stop_loss = actual_entry - stop_distance
            take_profit = actual_entry + (stop_distance * self.take_profit_atr_multiple)
            direction = 'long'

        else:  # Short
            actual_entry = entry_price - total_cost
            stop_loss = actual_entry + stop_distance
            take_profit = actual_entry - (stop_distance * self.take_profit_atr_multiple)
            direction = 'short'

        # Position size (lots)
        position_size = risk_amount / stop_distance / 100000  # Standard lot size

        position = {
            'direction': direction,
            'entry_time': bar.name if hasattr(bar, 'name') else bar_idx,
            'entry_index': bar_idx,
            'entry_price': actual_entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_amount': risk_amount
        }

        return position

    def _update_position(self, position: Dict, bar: pd.Series, bar_idx: int) -> tuple:
        """Update position and check for exit."""
        # Calculate exit costs (spread + slippage on exit)
        exit_cost = (self.spread_pips + self.slippage_pips) * self.pip_value * position['position_size'] * 100000

        if position['direction'] == 'long':
            # Check stop loss
            if bar['low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl = -position['risk_amount'] - exit_cost  # Lose risk amount + exit costs
                outcome = 'loss'
                exit_reason = 'stop_loss'

                trade = Trade(
                    entry_time=position['entry_time'],
                    exit_time=bar.name if hasattr(bar, 'name') else bar_idx,
                    direction='long',
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    position_size=position['position_size'],
                    pnl=pnl,
                    pnl_pct=pnl / position['risk_amount'],
                    outcome=outcome,
                    exit_reason=exit_reason
                )
                return None, trade

            # Check take profit
            elif bar['high'] >= position['take_profit']:
                exit_price = position['take_profit']
                pnl = (position['risk_amount'] * self.take_profit_atr_multiple / self.stop_loss_atr_multiple) - exit_cost
                outcome = 'win'
                exit_reason = 'take_profit'

                trade = Trade(
                    entry_time=position['entry_time'],
                    exit_time=bar.name if hasattr(bar, 'name') else bar_idx,
                    direction='long',
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    position_size=position['position_size'],
                    pnl=pnl,
                    pnl_pct=pnl / position['risk_amount'],
                    outcome=outcome,
                    exit_reason=exit_reason
                )
                return None, trade

        else:  # Short
            # Check stop loss
            if bar['high'] >= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl = -position['risk_amount'] - exit_cost  # Lose risk amount + exit costs
                outcome = 'loss'
                exit_reason = 'stop_loss'

                trade = Trade(
                    entry_time=position['entry_time'],
                    exit_time=bar.name if hasattr(bar, 'name') else bar_idx,
                    direction='short',
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    position_size=position['position_size'],
                    pnl=pnl,
                    pnl_pct=pnl / position['risk_amount'],
                    outcome=outcome,
                    exit_reason=exit_reason
                )
                return None, trade

            # Check take profit
            elif bar['low'] <= position['take_profit']:
                exit_price = position['take_profit']
                pnl = (position['risk_amount'] * self.take_profit_atr_multiple / self.stop_loss_atr_multiple) - exit_cost
                outcome = 'win'
                exit_reason = 'take_profit'

                trade = Trade(
                    entry_time=position['entry_time'],
                    exit_time=bar.name if hasattr(bar, 'name') else bar_idx,
                    direction='short',
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    position_size=position['position_size'],
                    pnl=pnl,
                    pnl_pct=pnl / position['risk_amount'],
                    outcome=outcome,
                    exit_reason=exit_reason
                )
                return None, trade

        # Still in position
        return position, None

    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[float]) -> Dict:
        """Calculate performance metrics."""
        if len(trades) == 0:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_trade': 0,
                'final_capital': self.initial_capital
            }

        wins = [t for t in trades if t.outcome == 'win']
        losses = [t for t in trades if t.outcome == 'loss']

        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        win_rate = len(wins) / len(trades)

        gross_profit = sum([t.pnl for t in wins]) if wins else 0
        gross_loss = abs(sum([t.pnl for t in losses])) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe (simplified)
        returns = [t.pnl / self.initial_capital for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

        # Max drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max
        max_drawdown = np.max(drawdown)

        return {
            'total_return': total_return,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_trade': np.mean([t.pnl for t in trades]),
            'final_capital': equity_curve[-1]
        }


if __name__ == '__main__':
    # Test backtest engine
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 1.1000,
        'high': np.random.randn(1000).cumsum() + 1.1010,
        'low': np.random.randn(1000).cumsum() + 1.0990,
        'close': np.random.randn(1000).cumsum() + 1.1000,
        'atr': np.random.uniform(0.0001, 0.001, 1000),
        'spread_mean': np.random.uniform(0.00005, 0.0001, 1000)
    }, index=dates)

    # Random signals
    signals = pd.Series(np.random.choice([0, 0, 0, 1, -1], 1000), index=dates)

    engine = BacktestEngine(initial_capital=10000)
    results = engine.backtest(data, signals)

    print("\nBacktest Results:")
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key or 'drawdown' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nTotal trades: {len(results['trades'])}")
    if len(results['trades']) > 0:
        print(f"Sample trade: {results['trades'][0]}")
