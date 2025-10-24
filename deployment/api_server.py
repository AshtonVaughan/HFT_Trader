"""
FastAPI Production Server for Real-time Trading Inference

Provides REST API for:
- Real-time price predictions
- Trading signal generation
- Model health checks
- Performance metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import uvicorn

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.regime_detector.transformer_detector import RegimeDetector
from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
from models.meta_learner.attention_meta_learner import AttentionMetaLearner, EnsemblePredictor
from backtesting.risk_manager import RiskManager, RiskLimits
from utils.gpu_utils import get_device
from utils.logger import setup_logger, logger


# Request/Response Models
class MarketData(BaseModel):
    """Market data for prediction."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    features: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    """Prediction request."""
    market_data: List[MarketData]
    symbol: str = "EURUSD"
    sequence_length: int = 100


class PredictionResponse(BaseModel):
    """Prediction response."""
    symbol: str
    timestamp: str
    direction: int = Field(..., description="0=down, 1=up")
    direction_prob: float = Field(..., ge=0, le=1)
    magnitude: float
    confidence: float = Field(..., ge=0, le=1)
    regime: str
    regime_prob: float
    should_trade: bool
    position_size: Optional[float] = None


class SignalRequest(BaseModel):
    """Trading signal request."""
    market_data: List[MarketData]
    symbol: str = "EURUSD"
    capital: float = 10000
    risk_limits: Optional[Dict] = None


class SignalResponse(BaseModel):
    """Trading signal response."""
    symbol: str
    timestamp: str
    action: str = Field(..., description="BUY, SELL, or HOLD")
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reasoning: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    total_predictions: int
    avg_inference_time_ms: float


# Initialize FastAPI app
app = FastAPI(
    title="HFT Trader API",
    description="Real-time trading predictions and signals",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradingModel:
    """Trading model singleton."""

    def __init__(self):
        self.device = get_device()
        self.ensemble = None
        self.config = None
        self.risk_manager = None

        self.start_time = datetime.now()
        self.total_predictions = 0
        self.inference_times = []

        self.load_models()

    def load_models(self):
        """Load trained models."""
        logger.info("Loading models...")

        # Load config
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Model architecture (match training)
        feature_dim = 100  # Should match your actual feature dimension

        regime_detector = RegimeDetector(
            input_size=feature_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        ).to(self.device)

        lstm = LSTMPredictor(input_size=feature_dim, hidden_size=256, num_layers=3).to(self.device)
        gru = GRUPredictor(input_size=feature_dim, hidden_size=256, num_layers=3).to(self.device)
        cnn_lstm = CNNLSTMPredictor(input_size=feature_dim, cnn_channels=[64, 128, 256], lstm_hidden_size=256).to(self.device)

        specialized_models = [lstm, gru, cnn_lstm]

        meta_learner = AttentionMetaLearner(num_models=3, embedding_dim=128).to(self.device)

        self.ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner).to(self.device)

        # Load weights
        checkpoint_dir = Path(__file__).parent.parent / "models" / "checkpoints"

        regime_detector.load_state_dict(torch.load(checkpoint_dir / 'regime_detector.pth', map_location=self.device))
        lstm.load_state_dict(torch.load(checkpoint_dir / 'lstm.pth', map_location=self.device))
        gru.load_state_dict(torch.load(checkpoint_dir / 'gru.pth', map_location=self.device))
        cnn_lstm.load_state_dict(torch.load(checkpoint_dir / 'cnn_lstm.pth', map_location=self.device))
        meta_learner.load_state_dict(torch.load(checkpoint_dir / 'meta_learner.pth', map_location=self.device))

        self.ensemble.eval()

        # Initialize risk manager
        self.risk_manager = RiskManager(initial_capital=10000)

        logger.info(f"Models loaded successfully on {self.device}")

    def predict(self, market_data: List[MarketData], symbol: str) -> PredictionResponse:
        """Generate prediction."""
        start_time = datetime.now()

        # Convert to features (simplified - in production, use full feature engineering)
        features = self._prepare_features(market_data)

        # Run inference
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.ensemble(features_tensor)

            direction_logits = outputs['direction_logits']
            direction = torch.argmax(direction_logits, dim=1).item()
            direction_prob = torch.softmax(direction_logits, dim=1)[0, direction].item()

            magnitude = outputs['magnitude'].item()
            confidence = outputs['confidence'].item()

            regime_logits = outputs.get('regime_logits')
            if regime_logits is not None:
                regime_idx = torch.argmax(regime_logits, dim=1).item()
                regime_prob = torch.softmax(regime_logits, dim=1)[0, regime_idx].item()
                regime_names = ['trending_up', 'trending_down', 'ranging', 'volatile']
                regime = regime_names[regime_idx] if regime_idx < len(regime_names) else 'unknown'
            else:
                regime = 'unknown'
                regime_prob = 0.0

        # Determine if should trade
        should_trade = confidence > 0.65 and direction_prob > 0.6

        # Calculate position size if should trade
        position_size = None
        if should_trade:
            current_price = market_data[-1].close
            stop_loss_price = current_price * (0.995 if direction == 1 else 1.005)

            position_size = self.risk_manager.calculate_position_size(
                signal_strength=confidence,
                current_price=current_price,
                stop_loss_price=stop_loss_price,
                method='kelly'
            )

        # Track inference time
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]

        self.total_predictions += 1

        return PredictionResponse(
            symbol=symbol,
            timestamp=market_data[-1].timestamp,
            direction=direction,
            direction_prob=direction_prob,
            magnitude=magnitude,
            confidence=confidence,
            regime=regime,
            regime_prob=regime_prob,
            should_trade=should_trade,
            position_size=position_size
        )

    def generate_signal(
        self,
        market_data: List[MarketData],
        symbol: str,
        capital: float,
        risk_limits: Optional[Dict]
    ) -> SignalResponse:
        """Generate trading signal with full risk management."""
        # Get prediction
        prediction = self.predict(market_data, symbol)

        current_price = market_data[-1].close

        # Determine action
        if not prediction.should_trade:
            action = "HOLD"
            position_size = 0
            stop_loss = current_price
            take_profit = current_price
            reasoning = f"Low confidence ({prediction.confidence:.2%}) or probability ({prediction.direction_prob:.2%})"

        else:
            # Calculate position size with risk limits
            if risk_limits:
                limits = RiskLimits(**risk_limits)
                temp_risk_manager = RiskManager(capital, limits)
            else:
                temp_risk_manager = self.risk_manager

            # Calculate stop loss and take profit
            atr = self._estimate_atr(market_data)

            if prediction.direction == 1:  # Long
                action = "BUY"
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 2.5)
            else:  # Short
                action = "SELL"
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 2.5)

            # Calculate position size
            position_size = temp_risk_manager.calculate_position_size(
                signal_strength=prediction.confidence,
                current_price=current_price,
                stop_loss_price=stop_loss,
                method='kelly'
            )

            # Check risk limits
            allowed, reason = temp_risk_manager.check_risk_limits(position_size, current_price)

            if not allowed:
                action = "HOLD"
                position_size = 0
                reasoning = f"Risk limit violation: {reason}"
            else:
                reasoning = f"High confidence ({prediction.confidence:.2%}), regime: {prediction.regime}"

        return SignalResponse(
            symbol=symbol,
            timestamp=market_data[-1].timestamp,
            action=action,
            position_size=position_size,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=prediction.confidence,
            reasoning=reasoning
        )

    def _prepare_features(self, market_data: List[MarketData]) -> np.ndarray:
        """Convert market data to feature vector."""
        # Simplified feature extraction
        # In production, use full feature engineering pipeline
        df = pd.DataFrame([m.dict() for m in market_data])

        # Basic features
        df['return'] = df['close'].pct_change()
        df['high_low'] = (df['high'] - df['low']) / df['close']
        df['close_open'] = (df['close'] - df['open']) / df['open']

        # Fill NaN
        df = df.fillna(0)

        # Select last row features
        feature_cols = ['close', 'volume', 'return', 'high_low', 'close_open']
        features = df[feature_cols].iloc[-1].values

        # Pad to expected dimension
        expected_dim = 100
        if len(features) < expected_dim:
            features = np.pad(features, (0, expected_dim - len(features)))
        else:
            features = features[:expected_dim]

        return features

    def _estimate_atr(self, market_data: List[MarketData], period: int = 14) -> float:
        """Estimate Average True Range."""
        if len(market_data) < period:
            # Fallback to simple range
            return (market_data[-1].high - market_data[-1].low) * 2

        df = pd.DataFrame([m.dict() for m in market_data])

        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else (df['high'].iloc[-1] - df['low'].iloc[-1])


# Initialize model
model = TradingModel()


# API Endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "service": "HFT Trader API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - model.start_time).total_seconds()

    return HealthResponse(
        status="healthy" if model.ensemble is not None else "unhealthy",
        model_loaded=model.ensemble is not None,
        device=str(model.device),
        uptime_seconds=uptime,
        total_predictions=model.total_predictions,
        avg_inference_time_ms=np.mean(model.inference_times) if model.inference_times else 0
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate prediction."""
    try:
        if len(request.market_data) < request.sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data points. Required: {request.sequence_length}, Got: {len(request.market_data)}"
            )

        # Use last sequence_length points
        market_data = request.market_data[-request.sequence_length:]

        prediction = model.predict(market_data, request.symbol)

        return prediction

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signal", response_model=SignalResponse)
async def generate_signal(request: SignalRequest):
    """Generate trading signal."""
    try:
        if len(request.market_data) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data points. Required: 100, Got: {len(request.market_data)}"
            )

        signal = model.generate_signal(
            market_data=request.market_data,
            symbol=request.symbol,
            capital=request.capital,
            risk_limits=request.risk_limits
        )

        return signal

    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_models")
async def reload_models(background_tasks: BackgroundTasks):
    """Reload models (for updates)."""
    background_tasks.add_task(model.load_models)
    return {"message": "Model reload initiated"}


if __name__ == "__main__":
    # Setup logger
    setup_logger(level="INFO", log_file="api_server.log")

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
