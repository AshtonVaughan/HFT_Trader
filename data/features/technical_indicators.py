"""
Technical Indicators

RSI, ATR, Bollinger Bands, MACD, Stochastic, ADX, Support/Resistance, Fibonacci.
Uses pandas-ta for most indicators (fallback to manual calculation if not available).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class TechnicalIndicators:
    """
    Generate technical indicator features.
    """

    def __init__(self):
        logger.info("TechnicalIndicators initialized")

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with added indicators
        """
        logger.info("Adding technical indicators...")

        df = df.copy()

        # Trend indicators
        df = self.add_ema(df, periods=[9, 21, 50, 200])
        df = self.add_sma(df, periods=[20, 50, 200])

        # Momentum indicators
        df = self.add_rsi(df, periods=[14, 21])
        df = self.add_macd(df)
        df = self.add_stochastic(df)

        # Volatility indicators
        df = self.add_atr(df, period=14)
        df = self.add_bollinger_bands(df, period=20, std=2.0)

        # Trend strength
        df = self.add_adx(df, period=14)

        # Support/Resistance
        df = self.add_support_resistance(df, lookback=100, num_levels=3)

        # Fibonacci levels
        df = self.add_fibonacci_levels(df, lookback=50)

        logger.info(f"Added technical indicators")

        return df

    def add_ema(self, df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        df = df.copy()

        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        return df

    def add_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        df = df.copy()

        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

        return df

    def add_rsi(self, df: pd.DataFrame, periods: List[int] = [14, 21]) -> pd.DataFrame:
        """
        Add Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        df = df.copy()

        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return df

    def add_macd(self, df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).
        """
        df = df.copy()

        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        return df

    def add_stochastic(self, df: pd.DataFrame, k_period=14, d_period=3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.

        %K = 100 * (Close - Low_n) / (High_n - Low_n)
        %D = SMA(%K, d_period)
        """
        df = df.copy()

        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range.

        TR = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = SMA(TR, period)
        """
        df = df.copy()

        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()

        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']

        return df

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands.

        Middle Band = SMA(close, period)
        Upper Band = Middle + (std * StdDev)
        Lower Band = Middle - (std * StdDev)
        """
        df = df.copy()

        sma = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()

        df['bb_middle'] = sma
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_lower'] = sma - (std * std_dev)

        # Bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (trend strength).
        """
        df = df.copy()

        # Calculate directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        df['adx'] = dx.rolling(window=period).mean()

        df['pos_di'] = pos_di
        df['neg_di'] = neg_di

        return df

    def add_support_resistance(self, df: pd.DataFrame, lookback: int = 100, num_levels: int = 3) -> pd.DataFrame:
        """
        Add support and resistance levels based on recent swing highs/lows.
        """
        df = df.copy()

        for i in range(len(df)):
            if i < lookback:
                for level in range(num_levels):
                    df.loc[df.index[i], f'resistance_{level+1}'] = df['high'].iloc[:i+1].max()
                    df.loc[df.index[i], f'support_{level+1}'] = df['low'].iloc[:i+1].min()
            else:
                lookback_data = df.iloc[i-lookback:i]

                # Find swing highs (resistance)
                highs = lookback_data['high'].nlargest(num_levels).values
                for level, high in enumerate(highs):
                    df.loc[df.index[i], f'resistance_{level+1}'] = high

                # Find swing lows (support)
                lows = lookback_data['low'].nsmallest(num_levels).values
                for level, low in enumerate(lows):
                    df.loc[df.index[i], f'support_{level+1}'] = low

        # Distance to nearest support/resistance
        df['dist_to_resistance'] = (df['resistance_1'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['support_1']) / df['close']

        return df

    def add_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """
        Add Fibonacci retracement levels from recent swing high/low.
        """
        df = df.copy()

        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

        for i in range(lookback, len(df)):
            lookback_data = df.iloc[i-lookback:i]
            swing_high = lookback_data['high'].max()
            swing_low = lookback_data['low'].min()

            range_hl = swing_high - swing_low

            for ratio in fib_ratios:
                level = swing_high - (range_hl * ratio)
                df.loc[df.index[i], f'fib_{int(ratio*1000)}'] = level

        # Distance to key Fibonacci level (0.618)
        if 'fib_618' in df.columns:
            df['dist_to_fib_618'] = (df['close'] - df['fib_618']) / df['close']

        return df


if __name__ == '__main__':
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=500, freq='1T')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(500).cumsum() + 1.1000,
        'high': np.random.randn(500).cumsum() + 1.1010,
        'low': np.random.randn(500).cumsum() + 1.0990,
        'close': np.random.randn(500).cumsum() + 1.1000,
        'volume': np.random.randint(100, 1000, 500)
    })

    indicators = TechnicalIndicators()
    df_with_indicators = indicators.add_all_indicators(df)

    print(f"\nOriginal columns: {df.columns.tolist()}")
    print(f"New indicator columns: {[c for c in df_with_indicators.columns if c not in df.columns]}")
    print(f"\nTotal columns: {len(df_with_indicators.columns)}")
    print(f"\nSample RSI values:")
    print(df_with_indicators[['close', 'rsi_14', 'rsi_21']].tail(10))
