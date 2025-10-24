"""
Microstructure Features

Advanced order flow and microstructure features critical for HFT:
- Volume imbalance
- Trade intensity
- Spread dynamics
- Quote imbalance
- Price impact
- Trade flow toxicity (VPIN)
"""

import pandas as pd
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class MicrostructureFeatures:
    """
    Generate microstructure and order flow features.
    """

    def __init__(self):
        logger.info("MicrostructureFeatures initialized")

    def add_all_features(self, df: pd.DataFrame, has_bid_ask: bool = True) -> pd.DataFrame:
        """
        Add all microstructure features.

        Args:
            df: DataFrame with OHLCV or tick data
            has_bid_ask: Whether data includes bid/ask columns

        Returns:
            DataFrame with added features
        """
        logger.info("Adding microstructure features...")

        df = df.copy()

        if has_bid_ask:
            # Spread features
            df = self.add_spread_features(df)

            # Quote features
            df = self.add_quote_features(df)

            # Trade flow features
            df = self.add_trade_flow_features(df)

        # Volume features
        df = self.add_volume_features(df)

        # Trade intensity
        df = self.add_trade_intensity(df)

        # Price impact
        df = self.add_price_impact(df)

        # Liquidity features
        df = self.add_liquidity_features(df)

        logger.info(f"Added {len([c for c in df.columns if 'ms_' in c])} microstructure features")

        return df

    def add_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bid-ask spread features."""
        if 'bid' not in df.columns or 'ask' not in df.columns:
            logger.warning("No bid/ask columns found for spread features")
            return df

        # Raw spread
        df['ms_spread'] = df['ask'] - df['bid']

        # Relative spread (bps)
        mid = (df['bid'] + df['ask']) / 2
        df['ms_spread_bps'] = (df['ms_spread'] / mid) * 10000

        # Spread statistics
        windows = [10, 20, 50]
        for w in windows:
            df[f'ms_spread_mean_{w}'] = df['ms_spread'].rolling(w).mean()
            df[f'ms_spread_std_{w}'] = df['ms_spread'].rolling(w).std()
            df[f'ms_spread_min_{w}'] = df['ms_spread'].rolling(w).min()
            df[f'ms_spread_max_{w}'] = df['ms_spread'].rolling(w).max()

        # Spread momentum (is spread widening or tightening?)
        df['ms_spread_change'] = df['ms_spread'].diff()
        df['ms_spread_pct_change'] = df['ms_spread'].pct_change()

        # Spread percentile (is current spread wide or tight relative to recent history?)
        for w in [50, 100]:
            df[f'ms_spread_percentile_{w}'] = df['ms_spread'].rolling(w).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else 0.5
            )

        return df

    def add_quote_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quote-level features."""
        if 'bid' not in df.columns or 'ask' not in df.columns:
            return df

        # Quote mid price
        df['ms_mid'] = (df['bid'] + df['ask']) / 2

        # Quote changes (quote updates)
        df['ms_bid_change'] = df['bid'].diff()
        df['ms_ask_change'] = df['ask'].diff()
        df['ms_mid_change'] = df['ms_mid'].diff()

        # Quote arrival rate (how often quotes update)
        windows = [10, 20, 50]
        for w in windows:
            # Count number of quote changes in window
            df[f'ms_quote_updates_{w}'] = (
                (df['bid'].diff() != 0) | (df['ask'].diff() != 0)
            ).rolling(w).sum()

        # Microprice (weighted mid price based on depth)
        if 'bid_size' in df.columns and 'ask_size' in df.columns:
            total_size = df['bid_size'] + df['ask_size']
            df['ms_microprice'] = (
                df['bid'] * df['ask_size'] + df['ask'] * df['bid_size']
            ) / total_size

            # Quote imbalance
            df['ms_quote_imbalance'] = (df['bid_size'] - df['ask_size']) / total_size

            # Quote imbalance features
            for w in [10, 20, 50]:
                df[f'ms_quote_imbalance_mean_{w}'] = df['ms_quote_imbalance'].rolling(w).mean()
                df[f'ms_quote_imbalance_std_{w}'] = df['ms_quote_imbalance'].rolling(w).std()

        return df

    def add_trade_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trade flow features (buy vs sell volume)."""
        # Classify trades as buy or sell using tick rule
        # If price increased -> buy, if decreased -> sell
        df['ms_price_change'] = df['close'].diff()

        # Buy/sell classification (simplified)
        df['ms_trade_direction'] = np.where(df['ms_price_change'] > 0, 1,
                                           np.where(df['ms_price_change'] < 0, -1, 0))

        # Buy/sell volume
        if 'volume' in df.columns:
            df['ms_buy_volume'] = np.where(df['ms_trade_direction'] == 1, df['volume'], 0)
            df['ms_sell_volume'] = np.where(df['ms_trade_direction'] == -1, df['volume'], 0)

            # Volume imbalance (key feature for HFT)
            windows = [10, 20, 50, 100]
            for w in windows:
                buy_vol = df['ms_buy_volume'].rolling(w).sum()
                sell_vol = df['ms_sell_volume'].rolling(w).sum()
                total_vol = buy_vol + sell_vol

                df[f'ms_volume_imbalance_{w}'] = (buy_vol - sell_vol) / (total_vol + 1e-10)

            # Order flow toxicity (VPIN - Volume-Synchronized Probability of Informed Trading)
            for w in [50, 100]:
                df[f'ms_vpin_{w}'] = df[f'ms_volume_imbalance_{w}'].abs()

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'volume' not in df.columns:
            logger.warning("No volume column found")
            return df

        # Volume statistics
        windows = [10, 20, 50, 100]
        for w in windows:
            df[f'ms_volume_mean_{w}'] = df['volume'].rolling(w).mean()
            df[f'ms_volume_std_{w}'] = df['volume'].rolling(w).std()

        # Volume relative to recent average
        for w in [20, 50]:
            mean_vol = df[f'ms_volume_mean_{w}']
            df[f'ms_volume_ratio_{w}'] = df['volume'] / (mean_vol + 1e-10)

        # Volume percentile
        for w in [50, 100]:
            df[f'ms_volume_percentile_{w}'] = df['volume'].rolling(w).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else 0.5
            )

        # Volume momentum
        df['ms_volume_change'] = df['volume'].diff()
        df['ms_volume_pct_change'] = df['volume'].pct_change()

        return df

    def add_trade_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trade intensity features (ticks per unit time)."""
        # Number of trades in rolling window
        windows = [10, 20, 50]
        for w in windows:
            df[f'ms_trade_count_{w}'] = w  # In OHLC, each bar = 1 trade period

        # Trade arrival rate (trades per minute)
        if 'timestamp' in df.columns:
            # Calculate time between observations
            df['ms_time_delta'] = df['timestamp'].diff().dt.total_seconds()

            # Trade intensity = 1 / time_delta (inversely proportional)
            df['ms_trade_intensity'] = 1 / (df['ms_time_delta'] + 1e-10)

            # Smooth trade intensity
            for w in [10, 20, 50]:
                df[f'ms_trade_intensity_mean_{w}'] = df['ms_trade_intensity'].rolling(w).mean()

        return df

    def add_price_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price impact features (how much price moved per unit volume)."""
        if 'volume' not in df.columns:
            return df

        # Price change per unit volume
        price_change = df['close'].diff()
        df['ms_price_impact'] = price_change / (df['volume'] + 1e-10)

        # Smooth price impact
        windows = [10, 20, 50]
        for w in windows:
            df[f'ms_price_impact_mean_{w}'] = df['ms_price_impact'].rolling(w).mean()
            df[f'ms_price_impact_std_{w}'] = df['ms_price_impact'].rolling(w).std()

        # Kyle's lambda (price impact coefficient)
        for w in [20, 50]:
            # Covariance of price change and signed volume
            signed_volume = df['volume'] * df['ms_trade_direction']
            cov = df['close'].diff().rolling(w).cov(signed_volume)
            var = signed_volume.rolling(w).var()

            df[f'ms_kyle_lambda_{w}'] = cov / (var + 1e-10)

        return df

    def add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity features."""
        # Effective spread (actual cost of trading)
        if 'bid' in df.columns and 'ask' in df.columns:
            mid = (df['bid'] + df['ask']) / 2

            # For a buy trade (executed at ask)
            df['ms_effective_spread_buy'] = 2 * (df['ask'] - mid) / mid

            # For a sell trade (executed at bid)
            df['ms_effective_spread_sell'] = 2 * (mid - df['bid']) / mid

            # Average effective spread
            df['ms_effective_spread'] = (df['ms_effective_spread_buy'] + df['ms_effective_spread_sell']) / 2

        # Price range as liquidity proxy (wider range = less liquid)
        if 'high' in df.columns and 'low' in df.columns:
            df['ms_price_range'] = df['high'] - df['low']

            for w in [10, 20, 50]:
                df[f'ms_price_range_mean_{w}'] = df['ms_price_range'].rolling(w).mean()

        # Amihud illiquidity measure (price impact per dollar volume)
        if 'volume' in df.columns:
            returns = df['close'].pct_change().abs()
            dollar_volume = df['close'] * df['volume']

            df['ms_amihud_illiquidity'] = returns / (dollar_volume + 1e-10)

            for w in [20, 50]:
                df[f'ms_amihud_illiquidity_mean_{w}'] = df['ms_amihud_illiquidity'].rolling(w).mean()

        return df

    def add_advanced_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced microstructure features."""
        # Roll measure (bid-ask bounce estimate)
        if 'close' in df.columns:
            price_changes = df['close'].diff()
            df['ms_roll_measure'] = -price_changes.rolling(50).cov(price_changes.shift(1))

        # Realized spread (post-trade price reversion)
        if 'close' in df.columns and 'ms_mid' in df.columns:
            # How much price reverted after 5 bars
            future_mid = df['ms_mid'].shift(-5)
            df['ms_realized_spread'] = 2 * (df['close'] - future_mid)

        # Order flow imbalance (comprehensive)
        if 'ms_buy_volume' in df.columns and 'ms_sell_volume' in df.columns:
            for w in [10, 20, 50]:
                buy_vol = df['ms_buy_volume'].rolling(w).sum()
                sell_vol = df['ms_sell_volume'].rolling(w).sum()

                # Percentage imbalance
                df[f'ms_ofi_pct_{w}'] = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-10)

                # Absolute imbalance
                df[f'ms_ofi_abs_{w}'] = buy_vol - sell_vol

        return df


if __name__ == '__main__':
    # Test microstructure features
    print("\n" + "="*80)
    print("Microstructure Features Test")
    print("="*80)

    # Create sample tick data
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='1s')

    df = pd.DataFrame({
        'timestamp': dates,
        'bid': np.random.randn(n).cumsum() / 100 + 1.1000,
        'ask': np.random.randn(n).cumsum() / 100 + 1.1005,
        'close': np.random.randn(n).cumsum() / 100 + 1.1002,
        'volume': np.random.randint(100, 10000, n),
        'bid_size': np.random.randint(1000, 50000, n),
        'ask_size': np.random.randint(1000, 50000, n),
    })

    # Add high/low for OHLC
    df['high'] = df['close'] + abs(np.random.randn(n)) * 0.0001
    df['low'] = df['close'] - abs(np.random.randn(n)) * 0.0001

    # Add features
    ms = MicrostructureFeatures()
    df_features = ms.add_all_features(df, has_bid_ask=True)
    df_features = ms.add_advanced_microstructure(df_features)

    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"With features: {len(df_features.columns)}")

    ms_cols = [c for c in df_features.columns if 'ms_' in c]
    print(f"\nMicrostructure features added: {len(ms_cols)}")
    print("\nSample features:")
    for col in ms_cols[:10]:
        print(f"  {col}")

    print(f"\nSample data:")
    print(df_features[['timestamp', 'close', 'ms_spread', 'ms_volume_imbalance_20', 'ms_vpin_50']].head())
