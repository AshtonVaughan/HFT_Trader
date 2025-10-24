"""
OHLC Builder - Aggregate tick data into OHLC bars.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class OHLCBuilder:
    """
    Build OHLC bars from tick data at multiple timeframes.
    """

    # Timeframe mapping (internal name -> pandas resample rule)
    TIMEFRAME_RULES = {
        "1m": "1T",    # 1 minute
        "5m": "5T",
        "15m": "15T",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D"
    }

    def __init__(self):
        """Initialize OHLC builder."""
        logger.info("OHLC builder initialized")

    def build_ohlc_from_ticks(
        self,
        tick_data: pd.DataFrame,
        timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h"],
        price_column: str = "bid"  # Use bid prices (or 'ask', or mid=(bid+ask)/2)
    ) -> Dict[str, pd.DataFrame]:
        """
        Build OHLC bars from tick data at multiple timeframes.

        Args:
            tick_data: DataFrame with columns: timestamp, bid, ask, volume
            timeframes: List of timeframes to build
            price_column: Which price to use ('bid', 'ask', or 'mid')

        Returns:
            Dictionary of {timeframe: ohlc_df}
        """
        if 'timestamp' not in tick_data.columns:
            logger.error("Tick data must have 'timestamp' column")
            return {}

        # Set timestamp as index
        df = tick_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Calculate mid price if requested
        if price_column == 'mid':
            df['mid'] = (df['bid'] + df['ask']) / 2.0
            price_col = 'mid'
        else:
            price_col = price_column

        if price_col not in df.columns:
            logger.error(f"Price column '{price_col}' not found in data")
            return {}

        # Build OHLC for each timeframe
        ohlc_data = {}

        for timeframe in timeframes:
            if timeframe not in self.TIMEFRAME_RULES:
                logger.warning(f"Unknown timeframe: {timeframe}, skipping")
                continue

            rule = self.TIMEFRAME_RULES[timeframe]
            logger.info(f"Building {timeframe} OHLC bars...")

            # Resample
            ohlc = df.resample(rule).agg({
                price_col: ['first', 'max', 'min', 'last', 'count']
            })

            # Flatten column names
            ohlc.columns = ['_'.join(col).strip() if col[1] else col[0] for col in ohlc.columns]

            # Rename to standard OHLC
            ohlc = ohlc.rename(columns={
                f'{price_col}_first': 'open',
                f'{price_col}_max': 'high',
                f'{price_col}_min': 'low',
                f'{price_col}_last': 'close',
                f'{price_col}_count': 'tick_count'
            })

            # Add volume if available
            if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
                volume = df.resample(rule).agg({
                    'bid_volume': 'sum',
                    'ask_volume': 'sum'
                })
                ohlc['volume'] = volume['bid_volume'] + volume['ask_volume']
            else:
                ohlc['volume'] = 0.0

            # Add spread statistics
            if 'spread' in df.columns:
                spread_stats = df.resample(rule)['spread'].agg(['mean', 'max', 'min'])
                ohlc['spread_mean'] = spread_stats['mean']
                ohlc['spread_max'] = spread_stats['max']
                ohlc['spread_min'] = spread_stats['min']

            # Drop bars with no ticks
            ohlc = ohlc[ohlc['tick_count'] > 0].copy()

            # Reset index to have timestamp as column
            ohlc = ohlc.reset_index()
            ohlc = ohlc.rename(columns={'timestamp': 'timestamp'})

            logger.info(f"  Created {len(ohlc):,} {timeframe} bars")

            ohlc_data[timeframe] = ohlc

        return ohlc_data

    def save_ohlc(
        self,
        ohlc_data: Dict[str, pd.DataFrame],
        output_dir: str,
        pair: str = "EURUSD",
        start_date: str = None,
        end_date: str = None
    ):
        """
        Save OHLC data to Parquet files.

        Args:
            ohlc_data: Dictionary of {timeframe: DataFrame}
            output_dir: Output directory
            pair: Pair name
            start_date: Start date for filename
            end_date: End date for filename
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for timeframe, df in ohlc_data.items():
            if len(df) == 0:
                continue

            # Build filename
            if start_date and end_date:
                filename = f"{pair}_{timeframe}_ohlc_{start_date}_to_{end_date}.parquet"
            else:
                filename = f"{pair}_{timeframe}_ohlc.parquet"

            output_file = output_path / filename
            df.to_parquet(output_file, compression='snappy', index=False)
            logger.info(f"  Saved {timeframe} to {output_file}")

    def merge_timeframes(
        self,
        ohlc_data: Dict[str, pd.DataFrame],
        base_timeframe: str = "1m"
    ) -> pd.DataFrame:
        """
        Merge multiple timeframes into a single DataFrame with prefixed columns.

        Args:
            ohlc_data: Dictionary of {timeframe: DataFrame}
            base_timeframe: The base timeframe to use as index

        Returns:
            Merged DataFrame with columns like: 1m_open, 5m_open, 1h_close, etc.
        """
        if base_timeframe not in ohlc_data:
            logger.error(f"Base timeframe '{base_timeframe}' not found")
            return pd.DataFrame()

        # Start with base timeframe
        merged = ohlc_data[base_timeframe].copy()
        merged = merged.set_index('timestamp')

        # Add other timeframes
        for tf, df in ohlc_data.items():
            if tf == base_timeframe:
                continue

            df_indexed = df.set_index('timestamp')

            # Prefix columns
            df_indexed.columns = [f"{tf}_{col}" for col in df_indexed.columns]

            # Forward-fill to align with base timeframe
            merged = merged.join(df_indexed, how='left')
            merged[[col for col in merged.columns if col.startswith(f"{tf}_")]] = \
                merged[[col for col in merged.columns if col.startswith(f"{tf}_")]].ffill()

        merged = merged.reset_index()
        logger.info(f"Merged {len(ohlc_data)} timeframes into {len(merged)} rows")

        return merged


if __name__ == '__main__':
    # Test with sample tick data
    from data.collectors.dukascopy_downloader import DukascopyDownloader

    # Download 1 day of ticks
    downloader = DukascopyDownloader(output_dir="../../../raw_data")
    ticks = downloader.download_tick_data(
        pair="EURUSD",
        start_date="2024-01-02",
        end_date="2024-01-03",
        max_workers=4
    )

    if len(ticks) > 0:
        # Build OHLC
        builder = OHLCBuilder()
        ohlc_data = builder.build_ohlc_from_ticks(
            tick_data=ticks,
            timeframes=["1m", "5m", "15m", "1h"],
            price_column="mid"
        )

        # Print samples
        for tf, df in ohlc_data.items():
            print(f"\n{tf} OHLC sample:")
            print(df.head())

        # Save
        builder.save_ohlc(
            ohlc_data=ohlc_data,
            output_dir="../../../processed_data",
            pair="EURUSD",
            start_date="2024-01-02",
            end_date="2024-01-03"
        )

        # Test merge
        merged = builder.merge_timeframes(ohlc_data, base_timeframe="1m")
        print(f"\nMerged shape: {merged.shape}")
        print(merged.head())
