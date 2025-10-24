"""
Master Data Collection Script

Orchestrates the entire data collection process:
1. Download EUR/USD tick data from Dukascopy
2. Aggregate ticks into multiple OHLC timeframes
3. Download cross-pair and context data (EUR/GBP, GBP/USD, DXY, Gold)
4. Merge all data for feature engineering
"""

import yaml
import argparse
from pathlib import Path
from datetime import datetime

from data.collectors.dukascopy_downloader import DukascopyDownloader
from data.collectors.yahoo_ohlc import YahooOHLCDownloader
from data.collectors.alphavantage_downloader import AlphaVantageDownloader
from data.preprocessors.ohlc_builder import OHLCBuilder
from utils.logger import logger


class DataCollector:
    """
    Master data collection orchestrator.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.raw_data_dir = Path(self.config['data_collection']['raw_data_dir'])
        self.processed_data_dir = Path(self.config['data_collection']['processed_data_dir'])

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Data Collector initialized")
        logger.info(f"Raw data dir: {self.raw_data_dir}")
        logger.info(f"Processed data dir: {self.processed_data_dir}")

    def collect_all_data(self, use_tick_data: bool = True):
        """
        Run the complete data collection pipeline.

        Args:
            use_tick_data: If True, download tick data and aggregate to OHLC.
                          If False, download OHLC directly from Yahoo (faster but less granular)
        """
        logger.info("="*80)
        logger.info("STARTING DATA COLLECTION PIPELINE")
        logger.info("="*80)

        start_date = self.config['data_collection']['start_date']
        end_date = self.config['data_collection']['end_date']
        primary_pair = self.config['data_collection']['primary_pair']

        # Step 1: Collect primary pair data
        if use_tick_data:
            logger.info("\n" + "="*80)
            logger.info("STEP 1: Download EUR/USD tick data")
            logger.info("="*80)
            primary_data = self._collect_tick_data(primary_pair, start_date, end_date)
        else:
            logger.info("\n" + "="*80)
            logger.info("STEP 1: Download EUR/USD OHLC data")
            logger.info("="*80)
            primary_data = self._collect_ohlc_data(primary_pair, start_date, end_date)

        # Step 2: Collect cross-pair data
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Download cross-pair data")
        logger.info("="*80)
        cross_pair_data = self._collect_cross_pairs(start_date, end_date)

        # Step 3: Collect context data (DXY, Gold)
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Download context data (DXY, Gold)")
        logger.info("="*80)
        context_data = self._collect_context_data(start_date, end_date)

        logger.info("\n" + "="*80)
        logger.info("DATA COLLECTION COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nPrimary pair ({primary_pair}): {len(primary_data.get('1m', [])):,} 1-minute bars")
        logger.info(f"Cross pairs: {len(cross_pair_data)} pairs")
        logger.info(f"Context instruments: {len(context_data)} instruments")
        logger.info(f"\nData saved to: {self.processed_data_dir}")

    def _collect_tick_data(self, pair: str, start_date: str, end_date: str) -> dict:
        """Download tick data and aggregate to OHLC."""
        # Download ticks
        downloader = DukascopyDownloader(output_dir=str(self.raw_data_dir))
        ticks = downloader.download_tick_data(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            max_workers=8  # Parallel downloads
        )

        if len(ticks) == 0:
            logger.error("Failed to download tick data! Falling back to Yahoo OHLC...")
            return self._collect_ohlc_data(pair, start_date, end_date)

        # Aggregate to OHLC
        logger.info("\nAggregating ticks to OHLC bars...")
        builder = OHLCBuilder()
        ohlc_data = builder.build_ohlc_from_ticks(
            tick_data=ticks,
            timeframes=self.config['timeframes']['additional'],
            price_column="mid"  # Use mid price (bid+ask)/2
        )

        # Save
        builder.save_ohlc(
            ohlc_data=ohlc_data,
            output_dir=str(self.processed_data_dir),
            pair=pair,
            start_date=start_date,
            end_date=end_date
        )

        return ohlc_data

    def _collect_ohlc_data(self, pair: str, start_date: str, end_date: str) -> dict:
        """Download OHLC data from Alpha Vantage (or Yahoo as fallback)."""
        # Try Alpha Vantage first (better intraday coverage)
        api_key = self.config.get('data_collection', {}).get('alphavantage_api_key', '')

        if api_key and api_key != "YOUR_API_KEY_HERE":
            logger.info("Using Alpha Vantage for OHLC data...")
            downloader = AlphaVantageDownloader(
                api_key=api_key,
                output_dir=str(self.processed_data_dir)
            )

            # Alpha Vantage uses EUR/USD format
            from_symbol = pair[:3] if len(pair) >= 6 else "EUR"
            to_symbol = pair[3:] if len(pair) >= 6 else "USD"

            ohlc_data = downloader.download_multiple_intervals(
                from_symbol=from_symbol,
                to_symbol=to_symbol,
                intervals=self.config['timeframes']['additional']
            )

            if len(ohlc_data) > 0:
                return ohlc_data

            logger.warning("Alpha Vantage failed, falling back to Yahoo Finance...")

        # Fallback to Yahoo Finance
        logger.info("Using Yahoo Finance for OHLC data...")
        downloader = YahooOHLCDownloader(output_dir=str(self.processed_data_dir))

        ohlc_data = {}

        # Download each timeframe
        for timeframe in self.config['timeframes']['additional']:
            logger.info(f"\nDownloading {pair} {timeframe} data...")
            df = downloader.download_ohlc(
                pair=pair,
                start_date=start_date,
                end_date=end_date,
                interval=timeframe
            )

            if len(df) > 0:
                ohlc_data[timeframe] = df

        return ohlc_data

    def _collect_cross_pairs(self, start_date: str, end_date: str) -> dict:
        """Download cross-pair data for correlations."""
        downloader = YahooOHLCDownloader(output_dir=str(self.processed_data_dir))

        cross_pairs = self.config['data_collection'].get('cross_pairs', [])
        if len(cross_pairs) == 0:
            logger.info("No cross pairs configured, skipping")
            return {}

        # Download all cross pairs (use 1-hour data for efficiency)
        cross_data = downloader.download_multiple_pairs(
            pairs=cross_pairs,
            start_date=start_date,
            end_date=end_date,
            interval="1h"
        )

        return cross_data

    def _collect_context_data(self, start_date: str, end_date: str) -> dict:
        """Download context instruments (DXY, Gold) for market regime detection."""
        downloader = YahooOHLCDownloader(output_dir=str(self.processed_data_dir))

        context_instruments = self.config['data_collection'].get('context_instruments', [])
        if len(context_instruments) == 0:
            logger.info("No context instruments configured, skipping")
            return {}

        # Map to internal names
        instrument_map = {
            "DX-Y.NYB": "DXY",
            "GC=F": "GOLD"
        }

        context_data = {}

        for instrument in context_instruments:
            internal_name = instrument_map.get(instrument, instrument)
            logger.info(f"\nDownloading {internal_name} ({instrument})...")

            df = downloader.download_ohlc(
                pair=internal_name,
                start_date=start_date,
                end_date=end_date,
                interval="1h"
            )

            if len(df) > 0:
                context_data[internal_name] = df

        return context_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect HFT Trader data")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--use-tick-data', action='store_true', help='Download tick data (slower but more granular)')
    parser.add_argument('--use-ohlc-only', action='store_true', help='Download OHLC only (faster)')

    args = parser.parse_args()

    # Default to OHLC if neither specified
    use_tick = args.use_tick_data
    if args.use_ohlc_only:
        use_tick = False

    collector = DataCollector(config_path=args.config)
    collector.collect_all_data(use_tick_data=use_tick)


if __name__ == '__main__':
    main()
