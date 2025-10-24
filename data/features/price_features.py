"""
Price and Volume Features

Implements VWAP, TWAP, returns, volume indicators, and other price-based features.
"""

import pandas as pd
import numpy as np
from typing import List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class PriceVolumeFeatures:
    """
    Generate price and volume-based features.
    """

    def __init__(self):
        logger.info("PriceVolumeFeatures initialized")

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all price and volume features to DataFrame.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with added features
        """
        logger.info("Adding price and volume features...")

        df = df.copy()

        # VWAP
        df = self.add_vwap(df, periods=[10, 20, 50, 100])

        # TWAP
        df = self.add_twap(df, periods=[10, 20, 50])

        # Returns
        df = self.add_returns(df, lags=[1, 5, 10, 30, 60])

        # Volume features
        df = self.add_volume_features(df, periods=[10, 20, 50])

        # Price range features
        df = self.add_price_range_features(df)

        # Tick velocity (if tick_count available)
        if 'tick_count' in df.columns:
            df = self.add_tick_velocity(df, periods=[10, 20])

        logger.info(f"Added {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} price/volume features")

        return df

    def add_vwap(self, df: pd.DataFrame, periods: List[int] = [10, 20, 50, 100]) -> pd.DataFrame:
        """
        Add Volume-Weighted Average Price.

        VWAP = Σ(Price * Volume) / Σ(Volume)
        """
        df = df.copy()
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0

        for period in periods:
            df[f'vwap_{period}'] = (
                (typical_price * df['volume']).rolling(window=period).sum() /
                df['volume'].rolling(window=period).sum()
            )

            # Distance from VWAP (normalized)
            df[f'vwap_{period}_dist'] = (df['close'] - df[f'vwap_{period}']) / df['close']

        return df

    def add_twap(self, df: pd.DataFrame, periods: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Add Time-Weighted Average Price (simple moving average of typical price).
        """
        df = df.copy()
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0

        for period in periods:
            df[f'twap_{period}'] = typical_price.rolling(window=period).mean()

        return df

    def add_returns(self, df: pd.DataFrame, lags: List[int] = [1, 5, 10, 30, 60]) -> pd.DataFrame:
        """
        Add price returns over different periods.
        """
        df = df.copy()

        for lag in lags:
            df[f'return_{lag}'] = df['close'].pct_change(periods=lag)
            df[f'log_return_{lag}'] = np.log(df['close'] / df['close'].shift(lag))

        return df

    def add_volume_features(self, df: pd.DataFrame, periods: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        Add volume-based features.
        """
        df = df.copy()

        # Volume SMA
        for period in periods:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()

        # Volume ratio (current / average)
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

        # Volume Rate of Change
        df['volume_roc_10'] = df['volume'].pct_change(periods=10)

        # Volume standard deviation
        df['volume_std_20'] = df['volume'].rolling(window=20).std()

        return df

    def add_price_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on price ranges.
        """
        df = df.copy()

        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = df['hl_range'] / df['close']

        # Body (Open-Close)
        df['body'] = abs(df['open'] - df['close'])
        df['body_pct'] = df['body'] / df['close']

        # Upper and lower shadows
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']

        # Shadow ratios
        df['shadow_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / df['hl_range']
        df['shadow_ratio'] = df['shadow_ratio'].fillna(0)

        return df

    def add_tick_velocity(self, df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        """
        Add tick velocity (ticks per minute).
        """
        df = df.copy()

        for period in periods:
            df[f'tick_velocity_{period}'] = df['tick_count'].rolling(window=period).mean()

        return df


if __name__ == '__main__':
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1T')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 1.1000,
        'high': np.random.randn(1000).cumsum() + 1.1010,
        'low': np.random.randn(1000).cumsum() + 1.0990,
        'close': np.random.randn(1000).cumsum() + 1.1000,
        'volume': np.random.randint(100, 1000, 1000)
    })

    features = PriceVolumeFeatures()
    df_with_features = features.add_all_features(df)

    print(f"\nOriginal columns: {df.columns.tolist()}")
    print(f"New columns: {[c for c in df_with_features.columns if c not in df.columns]}")
    print(f"\nTotal features: {len(df_with_features.columns)}")
    print(f"\nSample data:")
    print(df_with_features.head())
