"""
Alpha Vantage Forex Data Downloader

Downloads intraday forex data (1min, 5min, 15min, 30min, 60min).
Alpha Vantage provides better intraday coverage than Yahoo Finance.

Get free API key: https://www.alphavantage.co/support/#api-key
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class AlphaVantageDownloader:
    """
    Download forex data from Alpha Vantage.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # Interval mapping
    INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min"
    }

    def __init__(self, api_key: str, output_dir: str = "processed_data"):
        """
        Args:
            api_key: Alpha Vantage API key
            output_dir: Directory to save data
        """
        if api_key == "YOUR_API_KEY_HERE" or not api_key:
            raise ValueError(
                "Please set your Alpha Vantage API key in config.yaml!\n"
                "Get a free key at: https://www.alphavantage.co/support/#api-key"
            )

        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Alpha Vantage downloader initialized. Output dir: {self.output_dir}")

    def download_forex_intraday(
        self,
        from_symbol: str = "EUR",
        to_symbol: str = "USD",
        interval: str = "1m",
        outputsize: str = "full"
    ) -> pd.DataFrame:
        """
        Download intraday forex data.

        Args:
            from_symbol: Base currency (e.g., EUR)
            to_symbol: Quote currency (e.g., USD)
            interval: 1m, 5m, 15m, 30m, 1h
            outputsize: 'compact' (last 100 data points) or 'full' (full history)

        Returns:
            DataFrame with OHLCV data
        """
        if interval not in self.INTERVAL_MAP:
            logger.error(f"Invalid interval: {interval}. Must be one of {list(self.INTERVAL_MAP.keys())}")
            return pd.DataFrame()

        av_interval = self.INTERVAL_MAP[interval]

        logger.info(f"Downloading {from_symbol}/{to_symbol} {interval} data from Alpha Vantage...")

        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'interval': av_interval,
            'outputsize': outputsize,
            'apikey': self.api_key,
            'datatype': 'json'
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            data = response.json()

            # Check for errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()

            if 'Note' in data:
                logger.warning(f"API limit reached: {data['Note']}")
                logger.warning("Free tier: 25 requests/day. Wait 60 seconds or upgrade to premium.")
                return pd.DataFrame()

            if 'Information' in data:
                logger.warning(f"Alpha Vantage info: {data['Information']}")
                return pd.DataFrame()

            # Parse time series data
            time_series_key = f'Time Series FX ({av_interval})'
            if time_series_key not in data:
                logger.error(f"No data found. Response keys: {list(data.keys())}")
                return pd.DataFrame()

            time_series = data[time_series_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns
            df.columns = ['open', 'high', 'low', 'close']
            df = df.astype(float)

            # Add volume (not available for forex, use placeholder)
            df['volume'] = 0

            # Reset index
            df = df.reset_index()
            df.rename(columns={'index': 'timestamp'}, inplace=True)

            logger.info(f"Downloaded {len(df):,} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Save to Parquet
            pair_name = f"{from_symbol}{to_symbol}"
            output_file = self.output_dir / f"{pair_name}_{interval}_ohlc_alphavantage.parquet"
            df.to_parquet(output_file, compression='snappy', index=False)
            logger.info(f"Saved to {output_file}")

            return df

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return pd.DataFrame()

    def download_multiple_intervals(
        self,
        from_symbol: str = "EUR",
        to_symbol: str = "USD",
        intervals: list = ["1m", "5m", "15m", "1h"]
    ) -> dict:
        """
        Download multiple intervals.

        Args:
            from_symbol: Base currency
            to_symbol: Quote currency
            intervals: List of intervals

        Returns:
            Dictionary of {interval: DataFrame}
        """
        data = {}

        for i, interval in enumerate(intervals):
            logger.info(f"\nDownloading {interval} data ({i+1}/{len(intervals)})...")

            df = self.download_forex_intraday(from_symbol, to_symbol, interval)

            if len(df) > 0:
                data[interval] = df

            # Rate limiting: free tier allows 25 requests/day, 5 requests/minute
            if i < len(intervals) - 1:
                logger.info("Waiting 12 seconds (API rate limit)...")
                time.sleep(12)  # Wait 12 seconds between requests

        logger.info(f"\nSuccessfully downloaded {len(data)}/{len(intervals)} intervals")
        return data


if __name__ == '__main__':
    # Test downloader
    import sys

    print("\n" + "="*80)
    print("Alpha Vantage Forex Downloader Test")
    print("="*80)

    api_key = input("\nEnter your Alpha Vantage API key (or press Enter to skip): ").strip()

    if not api_key:
        print("\nTo get a free API key:")
        print("1. Go to: https://www.alphavantage.co/support/#api-key")
        print("2. Enter your email and click 'GET FREE API KEY'")
        print("3. Copy the key and add it to config/config.yaml")
        sys.exit(0)

    downloader = AlphaVantageDownloader(api_key=api_key, output_dir="../../../processed_data")

    # Download 1-minute data
    df = downloader.download_forex_intraday(
        from_symbol="EUR",
        to_symbol="USD",
        interval="1m",
        outputsize="full"
    )

    if len(df) > 0:
        print(f"\nSample data:")
        print(df.head(10))
        print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total bars: {len(df):,}")
