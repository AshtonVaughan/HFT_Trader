"""
MetaTrader 5 Data Downloader

Downloads real-time and historical forex data from MT5.
Provides tick data, OHLC bars, and bid/ask spreads.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class MT5Downloader:
    """
    Download forex data from MetaTrader 5.
    """

    def __init__(self, output_dir: str = "raw_data", login: int = None, password: str = None, server: str = None):
        """
        Args:
            output_dir: Directory to save data
            login: MT5 account login (optional for demo)
            password: MT5 account password
            server: MT5 server name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MT5
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            raise RuntimeError("Failed to initialize MT5")

        # Login if credentials provided
        if login and password and server:
            if not mt5.login(login, password, server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                raise RuntimeError("Failed to login to MT5")
            logger.info(f"Logged in to MT5: {server}")

        logger.info("MT5 downloader initialized")
        logger.info(f"MT5 version: {mt5.version()}")
        logger.info(f"Output dir: {self.output_dir}")

    def __del__(self):
        """Cleanup MT5 connection."""
        mt5.shutdown()

    def download_tick_data(
        self,
        symbol: str = "EURUSD",
        start_date: str = "2024-01-01",
        end_date: str = "2025-01-01"
    ) -> pd.DataFrame:
        """
        Download tick data.

        Args:
            symbol: Trading symbol (e.g., EURUSD)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: [timestamp, bid, ask, last, volume, flags]
        """
        logger.info(f"Downloading {symbol} tick data from {start_date} to {end_date}")

        # Convert dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Get ticks
        ticks = mt5.copy_ticks_range(symbol, start_dt, end_dt, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            logger.error(f"Failed to download ticks: {mt5.last_error()}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(ticks)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df[['timestamp', 'bid', 'ask', 'last', 'volume', 'flags']]

        # Calculate mid price and spread
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']

        logger.info(f"Downloaded {len(df):,} ticks from {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Save to parquet
        output_file = self.output_dir / f"{symbol}_ticks_{start_date}_to_{end_date}.parquet"
        df.to_parquet(output_file, compression='snappy', index=False)
        logger.info(f"Saved to {output_file}")

        return df

    def download_ohlc_data(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "1m",
        start_date: str = "2024-01-01",
        end_date: str = "2025-01-01"
    ) -> pd.DataFrame:
        """
        Download OHLC data.

        Args:
            symbol: Trading symbol
            timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        # Map timeframes
        tf_map = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }

        if timeframe not in tf_map:
            logger.error(f"Invalid timeframe: {timeframe}")
            return pd.DataFrame()

        mt5_tf = tf_map[timeframe]

        logger.info(f"Downloading {symbol} {timeframe} OHLC from {start_date} to {end_date}")

        # Convert dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Get rates
        rates = mt5.copy_rates_range(symbol, mt5_tf, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to download OHLC: {mt5.last_error()}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'tick_volume': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

        logger.info(f"Downloaded {len(df):,} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Save to parquet
        output_file = self.output_dir / f"{symbol}_{timeframe}_ohlc_{start_date}_to_{end_date}.parquet"
        df.to_parquet(output_file, compression='snappy', index=False)
        logger.info(f"Saved to {output_file}")

        return df

    def get_symbol_info(self, symbol: str = "EURUSD") -> dict:
        """Get symbol information."""
        info = mt5.symbol_info(symbol)

        if info is None:
            logger.error(f"Failed to get symbol info: {mt5.last_error()}")
            return {}

        return {
            'symbol': symbol,
            'description': info.description,
            'currency_base': info.currency_base,
            'currency_profit': info.currency_profit,
            'currency_margin': info.currency_margin,
            'digits': info.digits,
            'point': info.point,
            'trade_contract_size': info.trade_contract_size,
            'trade_tick_value': info.trade_tick_value,
            'trade_tick_size': info.trade_tick_size,
            'bid': info.bid,
            'ask': info.ask,
            'spread': info.spread,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step
        }

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        symbols = mt5.symbols_get()
        if symbols is None:
            return []

        return [s.name for s in symbols]


if __name__ == '__main__':
    # Test downloader
    print("\n" + "="*80)
    print("MT5 Downloader Test")
    print("="*80)

    try:
        downloader = MT5Downloader(output_dir="../../../raw_data")

        # Get symbol info
        info = downloader.get_symbol_info("EURUSD")
        print(f"\nSymbol Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Download 1 day of 1-minute data for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        df = downloader.download_ohlc_data(
            symbol="EURUSD",
            timeframe="1m",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        if len(df) > 0:
            print(f"\nSample data:")
            print(df.head(10))
            print(f"\nData range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Total bars: {len(df):,}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: MT5 must be installed and running for this to work.")
        print("If you don't have MT5, use Dukascopy or Alpha Vantage instead.")
