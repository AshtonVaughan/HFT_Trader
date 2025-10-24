"""
Data Quality Validation

Validates data for:
- Missing values and gaps
- Outliers and anomalies
- Data consistency
- Statistical properties
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


@dataclass
class ValidationReport:
    """Data validation report."""
    passed: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, any]


class DataValidator:
    """
    Comprehensive data quality validator.
    """

    def __init__(
        self,
        max_gap_minutes: int = 5,
        outlier_std_threshold: float = 5.0,
        min_data_points: int = 1000
    ):
        """
        Args:
            max_gap_minutes: Maximum acceptable gap in minutes
            outlier_std_threshold: Number of standard deviations for outlier detection
            min_data_points: Minimum required data points
        """
        self.max_gap_minutes = max_gap_minutes
        self.outlier_std_threshold = outlier_std_threshold
        self.min_data_points = min_data_points

    def validate(self, df: pd.DataFrame, data_type: str = "ohlc") -> ValidationReport:
        """
        Run full validation suite.

        Args:
            df: DataFrame to validate
            data_type: 'ohlc', 'tick', or 'features'

        Returns:
            ValidationReport with results
        """
        logger.info(f"Validating {data_type} data ({len(df):,} rows)...")

        errors = []
        warnings_list = []
        stats = {}

        # 1. Check minimum data points
        if len(df) < self.min_data_points:
            errors.append(f"Insufficient data: {len(df)} rows (minimum: {self.min_data_points})")

        # 2. Check for missing values
        missing_report = self._check_missing_values(df)
        if missing_report['has_missing']:
            warnings_list.append(f"Missing values found: {missing_report}")
        stats['missing_values'] = missing_report

        # 3. Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            warnings_list.append(f"Found {duplicates} duplicate rows")
        stats['duplicates'] = duplicates

        # 4. Check temporal consistency
        if 'timestamp' in df.columns:
            time_report = self._check_temporal_consistency(df)
            if len(time_report['errors']) > 0:
                errors.extend(time_report['errors'])
            if len(time_report['warnings']) > 0:
                warnings_list.extend(time_report['warnings'])
            stats['temporal'] = time_report['stats']

        # 5. Check for outliers
        if data_type == 'ohlc':
            outlier_report = self._check_ohlc_outliers(df)
        else:
            outlier_report = self._check_general_outliers(df)

        if outlier_report['count'] > 0:
            warnings_list.append(f"Found {outlier_report['count']} outliers")
        stats['outliers'] = outlier_report

        # 6. Check data consistency (OHLC specific)
        if data_type == 'ohlc':
            consistency_report = self._check_ohlc_consistency(df)
            if len(consistency_report['errors']) > 0:
                errors.extend(consistency_report['errors'])
            stats['ohlc_consistency'] = consistency_report

        # 7. Statistical validation
        stat_report = self._check_statistical_properties(df, data_type)
        if len(stat_report['warnings']) > 0:
            warnings_list.extend(stat_report['warnings'])
        stats['statistical'] = stat_report['stats']

        # 8. Check for data drift (if we have historical stats)
        # drift_report = self._check_data_drift(df)
        # stats['drift'] = drift_report

        passed = len(errors) == 0

        report = ValidationReport(
            passed=passed,
            errors=errors,
            warnings=warnings_list,
            stats=stats
        )

        self._log_report(report)

        return report

    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values."""
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100

        missing_cols = {col: {'count': int(missing_count[col]), 'pct': float(missing_pct[col])}
                        for col in df.columns if missing_count[col] > 0}

        return {
            'has_missing': len(missing_cols) > 0,
            'total_missing': int(missing_count.sum()),
            'columns': missing_cols
        }

    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """Check temporal consistency."""
        errors = []
        warnings = []
        stats = {}

        if 'timestamp' not in df.columns:
            errors.append("No timestamp column found")
            return {'errors': errors, 'warnings': warnings, 'stats': stats}

        # Ensure sorted
        if not df['timestamp'].is_monotonic_increasing:
            warnings.append("Timestamps not sorted")

        # Check for gaps
        time_diffs = df['timestamp'].diff()
        max_gap = time_diffs.max()
        mean_gap = time_diffs.mean()

        gaps = time_diffs[time_diffs > pd.Timedelta(minutes=self.max_gap_minutes)]
        if len(gaps) > 0:
            warnings.append(f"Found {len(gaps)} time gaps > {self.max_gap_minutes} minutes")

        # Check for negative time deltas (time going backwards)
        negative_deltas = time_diffs[time_diffs < pd.Timedelta(0)]
        if len(negative_deltas) > 0:
            errors.append(f"Found {len(negative_deltas)} negative time deltas")

        stats = {
            'max_gap_minutes': float(max_gap.total_seconds() / 60) if pd.notna(max_gap) else None,
            'mean_gap_seconds': float(mean_gap.total_seconds()) if pd.notna(mean_gap) else None,
            'num_gaps': len(gaps),
            'num_negative_deltas': len(negative_deltas),
            'start_time': str(df['timestamp'].min()),
            'end_time': str(df['timestamp'].max()),
            'duration_days': float((df['timestamp'].max() - df['timestamp'].min()).days)
        }

        return {'errors': errors, 'warnings': warnings, 'stats': stats}

    def _check_ohlc_outliers(self, df: pd.DataFrame) -> Dict:
        """Check for OHLC outliers."""
        outliers = []
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]

        if len(price_cols) == 0:
            return {'count': 0, 'indices': [], 'details': {}}

        for col in price_cols:
            mean = df[col].mean()
            std = df[col].std()

            # Z-score method
            z_scores = np.abs((df[col] - mean) / std)
            col_outliers = df[z_scores > self.outlier_std_threshold].index.tolist()

            if len(col_outliers) > 0:
                outliers.extend(col_outliers)

        outliers = list(set(outliers))  # Remove duplicates

        return {
            'count': len(outliers),
            'indices': outliers[:100],  # Limit to first 100
            'pct': float(len(outliers) / len(df) * 100)
        }

    def _check_general_outliers(self, df: pd.DataFrame) -> Dict:
        """Check for outliers in any numeric column."""
        outliers = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['timestamp']:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            # IQR method
            lower_bound = q1 - (3 * iqr)
            upper_bound = q3 + (3 * iqr)

            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()

            if len(col_outliers) > 0:
                outliers.extend(col_outliers)

        outliers = list(set(outliers))

        return {
            'count': len(outliers),
            'indices': outliers[:100],
            'pct': float(len(outliers) / len(df) * 100) if len(df) > 0 else 0
        }

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> Dict:
        """Check OHLC consistency (high >= low, etc.)."""
        errors = []

        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            errors.append("Missing OHLC columns")
            return {'errors': errors}

        # High should be >= Low
        invalid_high_low = df[df['high'] < df['low']]
        if len(invalid_high_low) > 0:
            errors.append(f"High < Low in {len(invalid_high_low)} rows")

        # High should be >= Open and Close
        invalid_high = df[(df['high'] < df['open']) | (df['high'] < df['close'])]
        if len(invalid_high) > 0:
            errors.append(f"High < Open/Close in {len(invalid_high)} rows")

        # Low should be <= Open and Close
        invalid_low = df[(df['low'] > df['open']) | (df['low'] > df['close'])]
        if len(invalid_low) > 0:
            errors.append(f"Low > Open/Close in {len(invalid_low)} rows")

        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            invalid_prices = df[df[col] <= 0]
            if len(invalid_prices) > 0:
                errors.append(f"{col} has {len(invalid_prices)} zero/negative values")

        return {'errors': errors, 'invalid_count': len(invalid_high_low) + len(invalid_high) + len(invalid_low)}

    def _check_statistical_properties(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Check statistical properties."""
        warnings = []
        stats = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['timestamp']:
                continue

            col_mean = df[col].mean()
            col_std = df[col].std()
            col_skew = df[col].skew()
            col_kurt = df[col].kurtosis()

            # Check for constant columns (no variance)
            if col_std == 0 or np.isnan(col_std):
                warnings.append(f"Column '{col}' has zero variance (constant)")

            # Check for extreme skewness
            if abs(col_skew) > 10:
                warnings.append(f"Column '{col}' has extreme skewness: {col_skew:.2f}")

            # Check for inf values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                warnings.append(f"Column '{col}' has {inf_count} infinite values")

            stats[col] = {
                'mean': float(col_mean) if not np.isnan(col_mean) else None,
                'std': float(col_std) if not np.isnan(col_std) else None,
                'min': float(df[col].min()) if not np.isnan(df[col].min()) else None,
                'max': float(df[col].max()) if not np.isnan(df[col].max()) else None,
                'skew': float(col_skew) if not np.isnan(col_skew) else None,
                'kurtosis': float(col_kurt) if not np.isnan(col_kurt) else None
            }

        return {'warnings': warnings, 'stats': stats}

    def _log_report(self, report: ValidationReport):
        """Log validation report."""
        if report.passed:
            logger.info("✓ Validation PASSED")
        else:
            logger.error(f"✗ Validation FAILED with {len(report.errors)} errors")

        if len(report.errors) > 0:
            logger.error("Errors:")
            for error in report.errors:
                logger.error(f"  - {error}")

        if len(report.warnings) > 0:
            logger.warning(f"Warnings ({len(report.warnings)}):")
            for warning in report.warnings[:10]:  # Show first 10
                logger.warning(f"  - {warning}")

    def fix_common_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically fix common data issues."""
        df_fixed = df.copy()

        logger.info("Fixing common data issues...")

        # 1. Remove duplicates
        duplicates_before = df_fixed.duplicated().sum()
        if duplicates_before > 0:
            df_fixed = df_fixed.drop_duplicates()
            logger.info(f"  Removed {duplicates_before} duplicate rows")

        # 2. Sort by timestamp
        if 'timestamp' in df_fixed.columns:
            df_fixed = df_fixed.sort_values('timestamp').reset_index(drop=True)
            logger.info("  Sorted by timestamp")

        # 3. Forward fill small gaps
        missing_before = df_fixed.isnull().sum().sum()
        if missing_before > 0:
            df_fixed = df_fixed.fillna(method='ffill').fillna(method='bfill')
            logger.info(f"  Filled {missing_before} missing values")

        # 4. Remove infinite values
        numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_mask = np.isinf(df_fixed[col])
            if inf_mask.any():
                df_fixed.loc[inf_mask, col] = np.nan
                df_fixed[col] = df_fixed[col].fillna(method='ffill')
                logger.info(f"  Fixed infinite values in {col}")

        # 5. Cap extreme outliers (optional)
        for col in numeric_cols:
            if col in ['timestamp']:
                continue

            q01 = df_fixed[col].quantile(0.01)
            q99 = df_fixed[col].quantile(0.99)

            outliers = ((df_fixed[col] < q01) | (df_fixed[col] > q99)).sum()
            if outliers > 0:
                df_fixed[col] = df_fixed[col].clip(lower=q01, upper=q99)
                logger.info(f"  Capped {outliers} outliers in {col}")

        logger.info(f"Data cleaning complete: {len(df)} → {len(df_fixed)} rows")

        return df_fixed


if __name__ == '__main__':
    # Test validator
    print("\n" + "="*80)
    print("Data Validator Test")
    print("="*80)

    # Create sample data with issues
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })

    # Introduce some issues
    df.loc[100, 'high'] = df.loc[100, 'low'] - 1  # Invalid: high < low
    df.loc[200:205, 'close'] = np.nan  # Missing values
    df.loc[300, 'close'] = 1000  # Outlier
    df = pd.concat([df, df.iloc[0:1]])  # Duplicate

    # Validate
    validator = DataValidator()
    report = validator.validate(df, data_type='ohlc')

    print(f"\nValidation Result: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")

    # Fix issues
    df_fixed = validator.fix_common_issues(df)

    # Re-validate
    report2 = validator.validate(df_fixed, data_type='ohlc')
    print(f"\nAfter fixing: {'PASSED' if report2.passed else 'FAILED'}")
