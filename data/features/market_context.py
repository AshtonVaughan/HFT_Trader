"""
Market Context Features

Cross-pair correlations, DXY correlation, Gold correlation, relative strength.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class MarketContextFeatures:
    """
    Generate market context features from cross-pairs and related instruments.
    """

    def __init__(self):
        logger.info("MarketContextFeatures initialized")

    def add_context_features(
        self,
        primary_df: pd.DataFrame,
        cross_pair_data: Dict[str, pd.DataFrame] = None,
        context_data: Dict[str, pd.DataFrame] = None,
        correlation_windows: List[int] = [50, 100, 200]
    ) -> pd.DataFrame:
        """
        Add market context features.

        Args:
            primary_df: Primary pair DataFrame (EURUSD)
            cross_pair_data: Dict of cross-pair DataFrames (EURGBP, GBPUSD)
            context_data: Dict of context instrument DataFrames (DXY, GOLD)
            correlation_windows: Rolling correlation windows

        Returns:
            DataFrame with context features
        """
        logger.info("Adding market context features...")

        df = primary_df.copy()
        df = df.set_index('timestamp')

        # Add cross-pair correlations
        if cross_pair_data:
            for pair_name, pair_df in cross_pair_data.items():
                df = self._add_cross_pair_features(df, pair_df, pair_name, correlation_windows)

        # Add DXY correlation
        if context_data and 'DXY' in context_data:
            df = self._add_dxy_features(df, context_data['DXY'], correlation_windows)

        # Add Gold correlation
        if context_data and 'GOLD' in context_data:
            df = self._add_gold_features(df, context_data['GOLD'], correlation_windows)

        # Relative strength
        df = self._add_relative_strength(df, cross_pair_data)

        df = df.reset_index()

        return df

    def _add_cross_pair_features(
        self,
        df: pd.DataFrame,
        cross_df: pd.DataFrame,
        pair_name: str,
        windows: List[int]
    ) -> pd.DataFrame:
        """Add features from a cross-pair."""
        # Align timestamps
        if 'timestamp' in cross_df.columns:
            cross_df = cross_df.set_index('timestamp')
        # If timestamp not in columns, assume it's already the index

        # Join cross-pair close prices
        df = df.join(cross_df[['close']].rename(columns={'close': f'{pair_name}_close'}), how='left')

        # Forward-fill to handle missing data
        df[f'{pair_name}_close'] = df[f'{pair_name}_close'].ffill()

        # Calculate rolling correlations
        for window in windows:
            df[f'corr_{pair_name}_{window}'] = df['close'].rolling(window=window).corr(df[f'{pair_name}_close'])

        # Price ratio
        df[f'ratio_{pair_name}'] = df['close'] / df[f'{pair_name}_close']

        return df

    def _add_dxy_features(self, df: pd.DataFrame, dxy_df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add Dollar Index (DXY) features."""
        if 'timestamp' in dxy_df.columns:
            dxy_df = dxy_df.set_index('timestamp')

        # Join DXY close
        df = df.join(dxy_df[['close']].rename(columns={'close': 'dxy_close'}), how='left')
        df['dxy_close'] = df['dxy_close'].ffill()

        # Rolling correlations
        for window in windows:
            df[f'corr_dxy_{window}'] = df['close'].rolling(window=window).corr(df['dxy_close'])

        # DXY returns
        df['dxy_return_10'] = df['dxy_close'].pct_change(periods=10)

        return df

    def _add_gold_features(self, df: pd.DataFrame, gold_df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add Gold features (risk-on/risk-off indicator)."""
        if 'timestamp' in gold_df.columns:
            gold_df = gold_df.set_index('timestamp')

        # Join Gold close
        df = df.join(gold_df[['close']].rename(columns={'close': 'gold_close'}), how='left')
        df['gold_close'] = df['gold_close'].ffill()

        # Rolling correlations
        for window in windows:
            df[f'corr_gold_{window}'] = df['close'].rolling(window=window).corr(df['gold_close'])

        # Gold returns (risk indicator)
        df['gold_return_10'] = df['gold_close'].pct_change(periods=10)

        return df

    def _add_relative_strength(self, df: pd.DataFrame, cross_pair_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add relative strength index compared to cross-pairs.
        """
        if not cross_pair_data:
            return df

        # Calculate z-scores of returns
        df['return_zscore'] = (df['return_10'] - df['return_10'].rolling(100).mean()) / df['return_10'].rolling(100).std()

        return df


if __name__ == '__main__':
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')

    primary_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(500).cumsum() + 1.1000,
        'high': np.random.randn(500).cumsum() + 1.1010,
        'low': np.random.randn(500).cumsum() + 1.0990,
        'close': np.random.randn(500).cumsum() + 1.1000,
        'volume': np.random.randint(100, 1000, 500),
        'return_10': np.random.randn(500) * 0.001
    })

    eurgbp_df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(500).cumsum() + 0.8500
    })

    dxy_df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(500).cumsum() + 103.5
    })

    context = MarketContextFeatures()
    result = context.add_context_features(
        primary_df=primary_df,
        cross_pair_data={'EURGBP': eurgbp_df},
        context_data={'DXY': dxy_df},
        correlation_windows=[50, 100]
    )

    print(f"\nOriginal columns: {primary_df.columns.tolist()}")
    print(f"New context columns: {[c for c in result.columns if c not in primary_df.columns]}")
    print(f"\nTotal columns: {len(result.columns)}")
    print(f"\nSample correlations:")
    print(result[['close', 'EURGBP_close', 'corr_EURGBP_50', 'dxy_close', 'corr_dxy_50']].tail(10))
