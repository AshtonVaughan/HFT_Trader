"""
Model Monitoring and Drift Detection

Monitors model performance in production:
- Performance degradation detection
- Data drift detection (PSI, KL divergence)
- Feature drift monitoring
- Alert system for anomalies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from scipy import stats
from scipy.spatial.distance import jensenshannon

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import logger


@dataclass
class DriftAlert:
    """Drift detection alert."""
    timestamp: datetime
    alert_type: str  # 'performance', 'data_drift', 'feature_drift'
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric: str
    current_value: float
    baseline_value: float
    threshold: float
    message: str


class ModelMonitor:
    """
    Monitor model performance and detect drift.
    """

    def __init__(
        self,
        baseline_data: Optional[pd.DataFrame] = None,
        performance_window: int = 100,
        drift_window: int = 1000
    ):
        """
        Args:
            baseline_data: Historical data for drift comparison
            performance_window: Window for performance monitoring
            drift_window: Window for drift detection
        """
        self.baseline_data = baseline_data
        self.performance_window = performance_window
        self.drift_window = drift_window

        # Tracking
        self.predictions = []
        self.actual_returns = []
        self.confidences = []
        self.timestamps = []
        self.features_history = []

        # Baseline statistics
        self.baseline_stats = {}
        if baseline_data is not None:
            self._calculate_baseline_stats()

        # Alerts
        self.alerts = []

        logger.info("Model Monitor initialized")

    def _calculate_baseline_stats(self):
        """Calculate baseline statistics from historical data."""
        numeric_cols = self.baseline_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            self.baseline_stats[col] = {
                'mean': self.baseline_data[col].mean(),
                'std': self.baseline_data[col].std(),
                'min': self.baseline_data[col].min(),
                'max': self.baseline_data[col].max(),
                'q25': self.baseline_data[col].quantile(0.25),
                'q50': self.baseline_data[col].quantile(0.50),
                'q75': self.baseline_data[col].quantile(0.75),
                'distribution': self.baseline_data[col].values
            }

        logger.info(f"Calculated baseline stats for {len(self.baseline_stats)} features")

    def log_prediction(
        self,
        prediction: int,
        confidence: float,
        actual_return: Optional[float] = None,
        features: Optional[Dict[str, float]] = None
    ):
        """Log a prediction."""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.timestamps.append(datetime.now())

        if actual_return is not None:
            self.actual_returns.append(actual_return)

        if features is not None:
            self.features_history.append(features)

    def check_performance_drift(self) -> List[DriftAlert]:
        """Check for performance degradation."""
        alerts = []

        if len(self.predictions) < self.performance_window:
            return alerts

        # Get recent performance
        recent_preds = self.predictions[-self.performance_window:]
        recent_actuals = self.actual_returns[-self.performance_window:]
        recent_confidences = self.confidences[-self.performance_window:]

        if len(recent_actuals) < self.performance_window:
            return alerts

        # Calculate metrics
        accuracy = np.mean([1 if p == (1 if a > 0 else 0) else 0
                           for p, a in zip(recent_preds, recent_actuals)])

        avg_confidence = np.mean(recent_confidences)

        # Average return
        avg_return = np.mean(recent_actuals)

        # Sharpe ratio
        if np.std(recent_actuals) > 0:
            sharpe = (avg_return / np.std(recent_actuals)) * np.sqrt(252)
        else:
            sharpe = 0

        # Check against thresholds
        if accuracy < 0.52:  # Below random
            alerts.append(DriftAlert(
                timestamp=datetime.now(),
                alert_type='performance',
                severity='critical' if accuracy < 0.48 else 'high',
                metric='accuracy',
                current_value=accuracy,
                baseline_value=0.55,
                threshold=0.52,
                message=f"Accuracy degraded to {accuracy:.2%}"
            ))

        if avg_confidence < 0.5:  # Model is uncertain
            alerts.append(DriftAlert(
                timestamp=datetime.now(),
                alert_type='performance',
                severity='medium',
                metric='confidence',
                current_value=avg_confidence,
                baseline_value=0.65,
                threshold=0.5,
                message=f"Low average confidence: {avg_confidence:.2%}"
            ))

        if sharpe < 0:  # Negative Sharpe
            alerts.append(DriftAlert(
                timestamp=datetime.now(),
                alert_type='performance',
                severity='high',
                metric='sharpe_ratio',
                current_value=sharpe,
                baseline_value=1.0,
                threshold=0,
                message=f"Negative Sharpe ratio: {sharpe:.2f}"
            ))

        return alerts

    def check_data_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """
        Check for data drift using PSI (Population Stability Index).

        PSI measures how much a distribution has shifted.
        PSI < 0.1: No significant change
        0.1 < PSI < 0.2: Moderate change
        PSI > 0.2: Significant change
        """
        alerts = []

        if self.baseline_data is None:
            logger.warning("No baseline data for drift detection")
            return alerts

        numeric_cols = [col for col in current_data.columns
                       if col in self.baseline_stats and pd.api.types.is_numeric_dtype(current_data[col])]

        for col in numeric_cols:
            psi = self._calculate_psi(
                baseline=self.baseline_stats[col]['distribution'],
                current=current_data[col].values
            )

            if psi > 0.2:
                severity = 'critical' if psi > 0.5 else 'high'
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    alert_type='data_drift',
                    severity=severity,
                    metric=f'psi_{col}',
                    current_value=psi,
                    baseline_value=0,
                    threshold=0.2,
                    message=f"Significant drift in {col}: PSI={psi:.3f}"
                ))

        return alerts

    def check_feature_drift(self) -> List[DriftAlert]:
        """Check for drift in individual features."""
        alerts = []

        if len(self.features_history) < self.drift_window or not self.baseline_stats:
            return alerts

        # Get recent features
        recent_features = self.features_history[-self.drift_window:]

        # Convert to DataFrame
        df_recent = pd.DataFrame(recent_features)

        for col in df_recent.columns:
            if col not in self.baseline_stats:
                continue

            # Calculate KL divergence
            kl_div = self._calculate_kl_divergence(
                baseline=self.baseline_stats[col]['distribution'],
                current=df_recent[col].values
            )

            if kl_div > 0.5:  # Significant divergence
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    alert_type='feature_drift',
                    severity='high' if kl_div > 1.0 else 'medium',
                    metric=f'kl_div_{col}',
                    current_value=kl_div,
                    baseline_value=0,
                    threshold=0.5,
                    message=f"Feature drift in {col}: KL={kl_div:.3f}"
                ))

            # Check for mean shift
            current_mean = df_recent[col].mean()
            baseline_mean = self.baseline_stats[col]['mean']
            baseline_std = self.baseline_stats[col]['std']

            if baseline_std > 0:
                z_score = abs((current_mean - baseline_mean) / baseline_std)

                if z_score > 3:  # 3-sigma event
                    alerts.append(DriftAlert(
                        timestamp=datetime.now(),
                        alert_type='feature_drift',
                        severity='medium',
                        metric=f'mean_shift_{col}',
                        current_value=current_mean,
                        baseline_value=baseline_mean,
                        threshold=baseline_mean + 3 * baseline_std,
                        message=f"Mean shift in {col}: {z_score:.2f} sigma"
                    ))

        return alerts

    def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index.

        PSI = sum((current_% - baseline_%) * ln(current_% / baseline_%))
        """
        # Create bins based on baseline
        bin_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate distributions
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to percentages
        baseline_pct = baseline_counts / len(baseline)
        current_pct = current_counts / len(current)

        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return psi

    def _calculate_kl_divergence(self, baseline: np.ndarray, current: np.ndarray, bins: int = 50) -> float:
        """Calculate KL divergence using histograms."""
        # Create bins
        all_data = np.concatenate([baseline, current])
        bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)

        # Calculate distributions
        baseline_hist, _ = np.histogram(baseline, bins=bin_edges, density=True)
        current_hist, _ = np.histogram(current, bins=bin_edges, density=True)

        # Normalize
        baseline_hist = baseline_hist / baseline_hist.sum()
        current_hist = current_hist / current_hist.sum()

        # Avoid zeros
        baseline_hist = np.where(baseline_hist == 0, 1e-10, baseline_hist)
        current_hist = np.where(current_hist == 0, 1e-10, current_hist)

        # Jensen-Shannon divergence (symmetric version of KL)
        js_div = jensenshannon(baseline_hist, current_hist) ** 2

        return js_div

    def run_full_check(self, current_data: Optional[pd.DataFrame] = None) -> List[DriftAlert]:
        """Run all checks and return alerts."""
        all_alerts = []

        # Performance drift
        perf_alerts = self.check_performance_drift()
        all_alerts.extend(perf_alerts)

        # Data drift
        if current_data is not None:
            data_alerts = self.check_data_drift(current_data)
            all_alerts.extend(data_alerts)

        # Feature drift
        feature_alerts = self.check_feature_drift()
        all_alerts.extend(feature_alerts)

        # Store alerts
        self.alerts.extend(all_alerts)

        # Log alerts
        if len(all_alerts) > 0:
            logger.warning(f"Detected {len(all_alerts)} alerts:")
            for alert in all_alerts:
                logger.warning(f"  [{alert.severity.upper()}] {alert.message}")

        return all_alerts

    def get_monitoring_report(self) -> Dict:
        """Get comprehensive monitoring report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(self.predictions),
            'alerts': {
                'total': len(self.alerts),
                'critical': len([a for a in self.alerts if a.severity == 'critical']),
                'high': len([a for a in self.alerts if a.severity == 'high']),
                'medium': len([a for a in self.alerts if a.severity == 'medium']),
                'low': len([a for a in self.alerts if a.severity == 'low'])
            }
        }

        # Recent performance
        if len(self.predictions) >= self.performance_window:
            recent_preds = self.predictions[-self.performance_window:]
            recent_actuals = self.actual_returns[-self.performance_window:]

            if len(recent_actuals) >= self.performance_window:
                accuracy = np.mean([1 if p == (1 if a > 0 else 0) else 0
                                   for p, a in zip(recent_preds, recent_actuals)])

                report['performance'] = {
                    'accuracy': accuracy,
                    'avg_confidence': np.mean(self.confidences[-self.performance_window:]),
                    'avg_return': np.mean(recent_actuals),
                    'sharpe': (np.mean(recent_actuals) / np.std(recent_actuals)) * np.sqrt(252) if np.std(recent_actuals) > 0 else 0
                }

        return report

    def save_report(self, output_path: str = "monitoring/reports"):
        """Save monitoring report."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self.get_monitoring_report()

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"monitoring_report_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Monitoring report saved to {report_file}")

        # Save alerts
        if len(self.alerts) > 0:
            alerts_df = pd.DataFrame([
                {
                    'timestamp': a.timestamp,
                    'type': a.alert_type,
                    'severity': a.severity,
                    'metric': a.metric,
                    'current_value': a.current_value,
                    'threshold': a.threshold,
                    'message': a.message
                }
                for a in self.alerts
            ])

            alerts_file = output_dir / f"alerts_{timestamp}.csv"
            alerts_df.to_csv(alerts_file, index=False)

            logger.info(f"Alerts saved to {alerts_file}")


if __name__ == '__main__':
    # Test monitor
    print("\n" + "="*80)
    print("Model Monitor Test")
    print("="*80)

    # Create baseline data
    baseline = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000) * 2 + 1,
        'feature3': np.random.exponential(1, 1000)
    })

    # Initialize monitor
    monitor = ModelMonitor(baseline_data=baseline)

    # Simulate predictions
    print("\n1. Simulating predictions...")
    for i in range(200):
        pred = np.random.choice([0, 1])
        conf = np.random.uniform(0.5, 0.9)
        actual = np.random.randn() * 0.001

        monitor.log_prediction(pred, conf, actual)

    # Check performance
    print("\n2. Checking performance drift...")
    perf_alerts = monitor.check_performance_drift()
    print(f"   Performance alerts: {len(perf_alerts)}")

    # Create drifted data
    print("\n3. Checking data drift...")
    drifted_data = pd.DataFrame({
        'feature1': np.random.randn(500) + 0.5,  # Mean shift
        'feature2': np.random.randn(500) * 3 + 1,  # Variance increase
        'feature3': np.random.exponential(1.5, 500)  # Distribution change
    })

    data_alerts = monitor.check_data_drift(drifted_data)
    print(f"   Data drift alerts: {len(data_alerts)}")

    # Full check
    print("\n4. Running full check...")
    all_alerts = monitor.run_full_check(drifted_data)
    print(f"   Total alerts: {len(all_alerts)}")

    # Get report
    print("\n5. Generating report...")
    report = monitor.get_monitoring_report()
    print(f"   Report:")
    for key, value in report.items():
        print(f"   {key}: {value}")

    # Save report
    monitor.save_report()
