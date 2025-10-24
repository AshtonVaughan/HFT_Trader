"""
Master Preprocessing Pipeline

Orchestrates the entire data preprocessing workflow:
1. Load raw OHLC data
2. Feature engineering
3. Train/val/test split with regime labeling
4. Save processed datasets
"""

import yaml
import pandas as pd
from pathlib import Path
import argparse

from data.preprocessors.feature_engineer import FeatureEngineer
from data.preprocessors.data_splitter import DataSplitter
from utils.logger import setup_logger, logger


def main():
    parser = argparse.ArgumentParser(description="Preprocess all HFT data")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--input-dir', type=str, default='processed_data', help='Input directory with OHLC data')
    parser.add_argument('--output-dir', type=str, default='processed_data', help='Output directory for features')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    setup_logger(
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('log_file', 'hft_preprocess.log')
    )

    logger.info("="*80)
    logger.info("HFT_TRADER PREPROCESSING PIPELINE")
    logger.info("="*80)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load primary pair data (try 1m, 5m, 15m, 1h in order)
    logger.info("\n1. Loading primary pair data...")
    primary_files = []
    for timeframe in ['1m', '5m', '15m', '1h']:
        primary_files = list(input_dir.glob(f"EURUSD_{timeframe}_*.parquet"))
        if len(primary_files) > 0:
            logger.info(f"   Found {timeframe} data")
            break

    if len(primary_files) == 0:
        logger.error(f"No EURUSD data found in {input_dir}")
        logger.error("Please run: python collect_data.py --use-ohlc-only")
        return

    primary_file = primary_files[0]
    logger.info(f"   Loading: {primary_file}")
    primary_df = pd.read_parquet(primary_file)
    logger.info(f"   Loaded {len(primary_df):,} bars")

    # 2. Load cross-pair data (optional)
    logger.info("\n2. Loading cross-pair data...")
    cross_pair_data = {}

    for pair in config.get('data_collection', {}).get('cross_pairs', []):
        pair_files = list(input_dir.glob(f"{pair}_*.parquet"))
        if len(pair_files) > 0:
            cross_df = pd.read_parquet(pair_files[0])
            cross_pair_data[pair] = cross_df
            logger.info(f"   {pair}: {len(cross_df):,} bars")

    # 3. Load context data (DXY, Gold)
    logger.info("\n3. Loading context data...")
    context_data = {}

    for instrument in config.get('data_collection', {}).get('context_instruments', []):
        internal_name = 'DXY' if 'DX-Y' in instrument else ('GOLD' if 'GC=' in instrument else instrument)
        context_files = list(input_dir.glob(f"{internal_name}_*.parquet"))
        if len(context_files) > 0:
            context_df = pd.read_parquet(context_files[0])
            context_data[internal_name] = context_df
            logger.info(f"   {internal_name}: {len(context_df):,} bars")

    # 4. Feature Engineering
    logger.info("\n4. Running feature engineering...")
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(
        primary_df=primary_df,
        cross_pair_data=cross_pair_data if len(cross_pair_data) > 0 else None,
        context_data=context_data if len(context_data) > 0 else None
    )

    # 5. Split and label
    logger.info("\n5. Splitting data and labeling regimes...")
    splitter = DataSplitter(
        train_pct=config.get('preprocessing', {}).get('train_pct', 0.70),
        val_pct=config.get('preprocessing', {}).get('val_pct', 0.15),
        test_pct=config.get('preprocessing', {}).get('test_pct', 0.15)
    )

    splits = splitter.split_and_label(features_df)

    # 6. Save
    logger.info("\n6. Saving processed datasets...")
    splitter.save_splits(splits, output_dir=output_dir)

    # 7. Summary
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nOriginal bars: {len(primary_df):,}")
    logger.info(f"Features generated: {len(features_df.columns)}")
    logger.info(f"Train samples: {len(splits['train']):,}")
    logger.info(f"Val samples: {len(splits['val']):,}")
    logger.info(f"Test samples: {len(splits['test']):,}")
    logger.info(f"\nOutputs saved to: {output_dir}/")
    logger.info(f"  - train.parquet")
    logger.info(f"  - val.parquet")
    logger.info(f"  - test.parquet")
    logger.info(f"  - scaler.pkl")

    logger.info("\nNext step: python train.py")


if __name__ == '__main__':
    main()
