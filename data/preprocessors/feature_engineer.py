"""
Master Feature Engineering Orchestrator

Combines all feature modules and creates the final feature matrix.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.features.price_features import PriceVolumeFeatures
from data.features.technical_indicators import TechnicalIndicators
from data.features.market_context import MarketContextFeatures
from data.features.time_features import TimeFeatures
from utils.logger import logger


class FeatureEngineer:
    """
    Master feature engineering pipeline.
    """

    def __init__(self):
        self.price_features = PriceVolumeFeatures()
        self.technical_indicators = TechnicalIndicators()
        self.market_context = MarketContextFeatures()
        self.time_features = TimeFeatures()

        logger.info("FeatureEngineer initialized")

    def engineer_features(
        self,
        primary_df: pd.DataFrame,
        cross_pair_data: Dict[str, pd.DataFrame] = None,
        context_data: Dict[str, pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate all features for the primary pair.

        Args:
            primary_df: Primary pair OHLCV data
            cross_pair_data: Cross-pair data (optional)
            context_data: Context instrument data (optional)

        Returns:
            DataFrame with all engineered features
        """
        logger.info("="*80)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("="*80)

        df = primary_df.copy()
        initial_cols = len(df.columns)

        # 1. Price & Volume Features
        logger.info("\n1. Adding price & volume features...")
        df = self.price_features.add_all_features(df)
        logger.info(f"   Added {len(df.columns) - initial_cols} features")

        # 2. Technical Indicators
        logger.info("\n2. Adding technical indicators...")
        prev_cols = len(df.columns)
        df = self.technical_indicators.add_all_indicators(df)
        logger.info(f"   Added {len(df.columns) - prev_cols} features")

        # 3. Time Features
        logger.info("\n3. Adding time features...")
        prev_cols = len(df.columns)
        df = self.time_features.add_all_time_features(df)
        logger.info(f"   Added {len(df.columns) - prev_cols} features")

        # 4. Market Context Features
        if cross_pair_data or context_data:
            logger.info("\n4. Adding market context features...")
            prev_cols = len(df.columns)
            df = self.market_context.add_context_features(df, cross_pair_data, context_data)
            logger.info(f"   Added {len(df.columns) - prev_cols} features")

        # 5. Clean data
        logger.info("\n5. Cleaning features...")
        df = self._clean_features(df)

        logger.info("\n" + "="*80)
        logger.info(f"FEATURE ENGINEERING COMPLETE")
        logger.info(f"Total features: {len(df.columns)} (from {initial_cols} original)")
        logger.info("="*80)

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean feature DataFrame.

        - Handle NaN/Inf values
        - Remove highly correlated features
        """
        df = df.copy()

        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill
        df = df.ffill().bfill()

        # Fill remaining NaNs with 0
        df = df.fillna(0)

        # Log cleaning stats
        logger.info(f"   Data shape: {df.shape}")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df


if __name__ == '__main__':
    # Test feature engineering
    import sys
    sys.path.append('..')

    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')

    primary_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 1.1000,
        'high': np.random.randn(1000).cumsum() + 1.1010,
        'low': np.random.randn(1000).cumsum() + 1.0990,
        'close': np.random.randn(1000).cumsum() + 1.1000,
        'volume': np.random.randint(100, 1000, 1000)
    })

    engineer = FeatureEngineer()
    result = engineer.engineer_features(primary_df)

    print(f"\nFinal feature count: {len(result.columns)}")
    print(f"Feature columns: {result.columns.tolist()}")
    print(f"\nSample data:")
    print(result.head())
    print(f"\nData info:")
    print(result.info())
