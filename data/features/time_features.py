"""
Time and Session Features

Trading session encoding, hour/day cyclical features, seasonality.
"""

import pandas as pd
import numpy as np
from datetime import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class TimeFeatures:
    """
    Generate time-based features.
    """

    # Trading sessions (UTC times)
    SESSIONS = {
        'asian': (time(0, 0), time(9, 0)),
        'london': (time(8, 0), time(17, 0)),
        'new_york': (time(13, 0), time(22, 0)),
        'overlap': (time(13, 0), time(17, 0))  # London/NY overlap (most liquid)
    }

    def __init__(self):
        logger.info("TimeFeatures initialized")

    def add_all_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all time-based features.

        Args:
            df: DataFrame with 'timestamp' column

        Returns:
            DataFrame with time features
        """
        logger.info("Adding time features...")

        df = df.copy()

        # Ensure timestamp exists - if not in columns, use index
        if 'timestamp' not in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df['timestamp'] = df.index
            else:
                df['timestamp'] = pd.to_datetime(df.index)
        elif not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Session encoding
        df = self.add_session_features(df)

        # Cyclical time encoding
        df = self.add_cyclical_features(df)

        # Day of week
        df = self.add_day_features(df)

        # Intraday patterns
        df = self.add_intraday_patterns(df)

        logger.info(f"Added time features")

        return df

    def add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trading session one-hot encoding.
        """
        df = df.copy()

        # Extract time component
        df['time_utc'] = df['timestamp'].dt.time

        # One-hot encode sessions
        df['session_asian'] = df['time_utc'].apply(lambda t: self._in_session(t, 'asian')).astype(int)
        df['session_london'] = df['time_utc'].apply(lambda t: self._in_session(t, 'london')).astype(int)
        df['session_new_york'] = df['time_utc'].apply(lambda t: self._in_session(t, 'new_york')).astype(int)
        df['session_overlap'] = df['time_utc'].apply(lambda t: self._in_session(t, 'overlap')).astype(int)

        # Session ID (0=asian, 1=london, 2=ny, 3=overlap)
        def get_session_id(t):
            if self._in_session(t, 'overlap'):
                return 3
            elif self._in_session(t, 'new_york'):
                return 2
            elif self._in_session(t, 'london'):
                return 1
            else:
                return 0

        df['session_id'] = df['time_utc'].apply(get_session_id)

        df = df.drop('time_utc', axis=1)

        return df

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical encoding for hour and day.

        Uses sine/cosine transformation to preserve cyclical nature.
        """
        df = df.copy()

        # Hour of day (0-23)
        hour = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Day of week (0-6, Monday=0)
        day = df['timestamp'].dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * day / 7)
        df['day_cos'] = np.cos(2 * np.pi * day / 7)

        # Minute of hour (for intraday patterns)
        minute = df['timestamp'].dt.minute
        df['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * minute / 60)

        return df

    def add_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add day-of-week features.
        """
        df = df.copy()

        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday

        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # One-hot encode weekdays
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        return df

    def add_intraday_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on intraday volatility patterns.

        Calculate historical volatility for each hour of day.
        """
        df = df.copy()

        df['hour'] = df['timestamp'].dt.hour

        # Calculate historical hourly volatility
        if 'return_1' in df.columns:
            hourly_vol = df.groupby('hour')['return_1'].transform(lambda x: x.rolling(100, min_periods=10).std())
            df['hourly_vol_pattern'] = hourly_vol

        # High liquidity hours (London/NY overlap: 13:00-17:00 UTC)
        df['is_high_liquidity'] = ((df['hour'] >= 13) & (df['hour'] < 17)).astype(int)

        df = df.drop('hour', axis=1)

        return df

    def _in_session(self, t: time, session_name: str) -> bool:
        """Check if time is within a session."""
        if session_name not in self.SESSIONS:
            return False

        start, end = self.SESSIONS[session_name]

        if start < end:
            return start <= t < end
        else:
            # Session crosses midnight
            return t >= start or t < end


if __name__ == '__main__':
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(1000).cumsum() + 1.1000,
        'return_1': np.random.randn(1000) * 0.001
    })

    time_features = TimeFeatures()
    df_with_time = time_features.add_all_time_features(df)

    print(f"\nOriginal columns: {df.columns.tolist()}")
    print(f"New time columns: {[c for c in df_with_time.columns if c not in df.columns]}")
    print(f"\nTotal columns: {len(df_with_time.columns)}")
    print(f"\nSample time features:")
    print(df_with_time[['timestamp', 'session_id', 'session_overlap', 'hour_sin', 'hour_cos', 'is_high_liquidity']].head(30))
