"""
Dukascopy Historical Tick Data Downloader

Downloads EUR/USD tick data from Dukascopy historical data feed.
Handles bi5 compressed format and saves to Parquet for fast loading.

Dukascopy URL structure:
http://datafeed.dukascopy.com/datafeed/{PAIR}/{YEAR}/{MONTH}/{DAY}/{HOUR}h_ticks.bi5
"""

import struct
import lzma
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class DukascopyDownloader:
    """
    Download historical tick data from Dukascopy.
    """

    BASE_URL = "https://datafeed.dukascopy.com/datafeed"

    def __init__(self, output_dir: str = "raw_data"):
        """
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dukascopy downloader initialized. Output dir: {self.output_dir}")

    def download_tick_data(
        self,
        pair: str = "EURUSD",
        start_date: str = "2022-01-01",
        end_date: str = "2025-01-01",
        max_workers: int = 8
    ) -> pd.DataFrame:
        """
        Download tick data for specified date range.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_workers: Number of parallel download threads

        Returns:
            DataFrame with tick data
        """
        logger.info(f"Downloading {pair} tick data from {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate list of hours to download
        hour_list = self._generate_hour_list(start_dt, end_dt)
        logger.info(f"Total hours to download: {len(hour_list)}")

        # Download in parallel
        all_ticks = []
        failed_downloads = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_hour, pair, dt): dt
                for dt in hour_list
            }

            with tqdm(total=len(hour_list), desc="Downloading ticks") as pbar:
                for future in as_completed(futures):
                    dt = futures[future]
                    try:
                        ticks = future.result()
                        if ticks is not None and len(ticks) > 0:
                            all_ticks.append(ticks)
                    except Exception as e:
                        logger.warning(f"Failed to download {dt}: {e}")
                        failed_downloads.append(dt)

                    pbar.update(1)

        if len(failed_downloads) > 0:
            logger.warning(f"Failed to download {len(failed_downloads)} hours")

        if len(all_ticks) == 0:
            logger.error("No data downloaded!")
            return pd.DataFrame()

        # Combine all ticks
        logger.info("Combining tick data...")
        df = pd.concat(all_ticks, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Downloaded {len(df):,} ticks from {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Save to Parquet
        output_file = self.output_dir / f"{pair}_ticks_{start_date}_to_{end_date}.parquet"
        df.to_parquet(output_file, compression='snappy')
        logger.info(f"Saved to {output_file}")

        return df

    def _generate_hour_list(self, start_dt: datetime, end_dt: datetime) -> List[datetime]:
        """Generate list of hour timestamps to download."""
        hour_list = []
        current = start_dt.replace(minute=0, second=0, microsecond=0)

        while current <= end_dt:
            hour_list.append(current)
            current += timedelta(hours=1)

        return hour_list

    def _download_hour(self, pair: str, dt: datetime) -> Optional[pd.DataFrame]:
        """
        Download tick data for a single hour.

        Args:
            pair: Currency pair
            dt: Datetime for the hour

        Returns:
            DataFrame with tick data for that hour
        """
        # Convert pair format (EURUSD -> EUR/USD)
        pair_formatted = f"{pair[:3]}/{pair[3:]}"

        # Build URL
        url = f"{self.BASE_URL}/{pair_formatted}/{dt.year}/{dt.month-1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"

        try:
            # Download
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                # Data might not exist for this hour (weekend, holiday)
                return None

            # Decompress
            decompressed = lzma.decompress(response.content)

            # Parse bi5 format
            ticks = self._parse_bi5(decompressed, dt)

            return ticks

        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error for {url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error processing {url}: {e}")
            return None

    def _parse_bi5(self, data: bytes, base_dt: datetime) -> pd.DataFrame:
        """
        Parse Dukascopy bi5 binary format.

        Format (20 bytes per tick):
        - 4 bytes: time offset (milliseconds from hour start)
        - 4 bytes: ask price (in point format)
        - 4 bytes: bid price (in point format)
        - 4 bytes: ask volume
        - 4 bytes: bid volume

        Args:
            data: Decompressed binary data
            base_dt: Base datetime (hour start)

        Returns:
            DataFrame with columns: timestamp, bid, ask, bid_volume, ask_volume
        """
        if len(data) == 0:
            return pd.DataFrame()

        num_ticks = len(data) // 20

        if num_ticks == 0:
            return pd.DataFrame()

        ticks = []

        for i in range(num_ticks):
            offset = i * 20

            # Unpack binary data (big-endian)
            time_ms, ask_points, bid_points, ask_vol, bid_vol = struct.unpack(
                '>IIIII',
                data[offset:offset+20]
            )

            # Convert to timestamp
            timestamp = base_dt + timedelta(milliseconds=time_ms)

            # Convert points to price (divide by 100000 for 5-decimal pairs)
            ask_price = ask_points / 100000.0
            bid_price = bid_points / 100000.0

            ticks.append({
                'timestamp': timestamp,
                'bid': bid_price,
                'ask': ask_price,
                'bid_volume': bid_vol / 1000000.0,  # Convert to lots
                'ask_volume': ask_vol / 1000000.0,
                'spread': ask_price - bid_price
            })

        df = pd.DataFrame(ticks)
        return df


if __name__ == '__main__':
    # Test downloader
    downloader = DukascopyDownloader(output_dir="../../../raw_data")

    # Download 1 week of data for testing
    test_start = "2024-01-01"
    test_end = "2024-01-07"

    df = downloader.download_tick_data(
        pair="EURUSD",
        start_date=test_start,
        end_date=test_end,
        max_workers=4
    )

    if len(df) > 0:
        print(f"\nSample data:")
        print(df.head(10))
        print(f"\nData info:")
        print(df.info())
        print(f"\nSpread statistics:")
        print(df['spread'].describe())
