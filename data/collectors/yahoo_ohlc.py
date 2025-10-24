"""
Yahoo Finance OHLC Data Downloader

Backup data source for OHLC bars and cross-pair data.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class YahooOHLCDownloader:
    """
    Download OHLC data from Yahoo Finance.
    """

    # Pair mapping (internal name -> Yahoo symbol)
    PAIR_MAPPING = {
        "EURUSD": "EURUSD=X",
        "EURGBP": "EURGBP=X",
        "GBPUSD": "GBPUSD=X",
        "DXY": "DX-Y.NYB",  # Dollar Index
        "GOLD": "GC=F"      # Gold Futures
    }

    def __init__(self, output_dir: str = "raw_data"):
        """
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Yahoo OHLC downloader initialized. Output dir: {self.output_dir}")

    def download_ohlc(
        self,
        pair: str = "EURUSD",
        start_date: str = "2022-01-01",
        end_date: str = "2025-01-01",
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Download OHLC data.

        Args:
            pair: Pair name (EURUSD, EURGBP, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 1h, 4h, 1d)

        Returns:
            DataFrame with OHLCV data
        """
        # Get Yahoo symbol
        yahoo_symbol = self.PAIR_MAPPING.get(pair, f"{pair}=X")

        logger.info(f"Downloading {pair} ({yahoo_symbol}) {interval} data from {start_date} to {end_date}")

        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if len(df) == 0:
                logger.error(f"No data downloaded for {pair}")
                return pd.DataFrame()

            # Clean up
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['open', 'high', 'low', 'close', 'volume']

            # Remove timezone for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df.index.name = 'timestamp'

            logger.info(f"Downloaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

            # Save to Parquet
            output_file = self.output_dir / f"{pair}_{interval}_ohlc_{start_date}_to_{end_date}.parquet"
            df.to_parquet(output_file, compression='snappy')
            logger.info(f"Saved to {output_file}")

            return df

        except Exception as e:
            logger.error(f"Error downloading {pair}: {e}")
            return pd.DataFrame()

    def download_multiple_pairs(
        self,
        pairs: list,
        start_date: str,
        end_date: str,
        interval: str = "1m"
    ) -> dict:
        """
        Download multiple pairs.

        Args:
            pairs: List of pair names
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Dictionary of {pair: DataFrame}
        """
        data = {}

        for pair in pairs:
            logger.info(f"\nDownloading {pair}...")
            df = self.download_ohlc(pair, start_date, end_date, interval)
            if len(df) > 0:
                data[pair] = df
            else:
                logger.warning(f"Skipping {pair} - no data")

        logger.info(f"\nSuccessfully downloaded {len(data)}/{len(pairs)} pairs")
        return data


if __name__ == '__main__':
    # Test downloader
    downloader = YahooOHLCDownloader(output_dir="../../../raw_data")

    # Download 1 month of 1-hour data for testing
    test_start = "2024-01-01"
    test_end = "2024-02-01"

    # Test single pair
    df_eurusd = downloader.download_ohlc(
        pair="EURUSD",
        start_date=test_start,
        end_date=test_end,
        interval="1h"
    )

    if len(df_eurusd) > 0:
        print(f"\nEURUSD sample:")
        print(df_eurusd.head(10))

    # Test multiple pairs
    print("\n" + "="*60)
    all_data = downloader.download_multiple_pairs(
        pairs=["EURUSD", "EURGBP", "GBPUSD", "DXY", "GOLD"],
        start_date=test_start,
        end_date=test_end,
        interval="1h"
    )

    print(f"\nDownloaded data summary:")
    for pair, df in all_data.items():
        print(f"  {pair}: {len(df)} bars")
