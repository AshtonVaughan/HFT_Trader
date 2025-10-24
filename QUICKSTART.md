# HFT_Trader - Quick Start Guide

Get up and running in 3 simple steps!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (H100, RTX 5090, or similar) - recommended
- 16GB+ RAM
- 10GB+ free disk space

## Step 1: Install Dependencies

```bash
cd HFT_Trader

# Install Python packages
pip install -r requirements.txt

# Note: If TA-Lib installation fails, install via:
# - Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# - Linux: sudo apt-get install ta-lib
# - Mac: brew install ta-lib
```

## Step 2: Collect Data

### Option A: OHLC Data Only (Recommended for Quick Start)

Download 3 years of EURUSD data in ~5-10 minutes:

```bash
python collect_data.py --use-ohlc-only
```

This downloads:
- EURUSD: 1m, 5m, 15m, 1h, 4h bars
- EUR/GBP, GBP/USD: 1h bars (cross-pairs)
- DXY, Gold: 1h bars (market context)

**Total size**: ~500 MB
**Download time**: 5-10 minutes

### Option B: Tick Data (For Production)

Download 3 years of tick-level data (more granular):

```bash
python collect_data.py --use-tick-data
```

**Total size**: ~50-100 GB
**Download time**: 2-4 hours

## Step 3: Preprocess Data

Generate features, split data, and create training sets:

```bash
python preprocess_all.py
```

This will:
1. Load OHLC data
2. Generate 200+ technical features
3. Add market context (correlations, regimes)
4. Split into train (70%) / val (15%) / test (15%)
5. Save preprocessed datasets

**Output files**:
- `processed_data/train.parquet` - Training set
- `processed_data/val.parquet` - Validation set
- `processed_data/test.parquet` - Test set
- `processed_data/scaler.pkl` - Feature scaler

**Time**: 5-15 minutes (depending on data size)

## Step 4: Train Models

Train the full ensemble (Regime Detector + LSTM/GRU/CNN-LSTM + Meta-Learner):

```bash
python train.py
```

This will:
1. Load preprocessed data
2. Train Regime Detector (Transformer) - 10 epochs
3. Train LSTM Predictor - 10 epochs
4. Train GRU Predictor - 10 epochs
5. Train CNN-LSTM Predictor - 10 epochs
6. Train Meta-Learner (ensemble) - 5 epochs
7. Evaluate on test set
8. Run backtest on test period

**Time on H100**: ~2-4 hours (full training)
**Time on CPU**: ~12-24 hours

**Outputs**:
- `models/checkpoints/regime_detector.pth`
- `models/checkpoints/lstm.pth`
- `models/checkpoints/gru.pth`
- `models/checkpoints/cnn_lstm.pth`
- `models/checkpoints/meta_learner.pth`

## What to Expect

### Training Metrics

After training, you should see:

```
Test Metrics:
  accuracy: 0.5200-0.5800  # Direction prediction accuracy (52-58%)

Backtest Results:
  total_return: -0.05 to +0.20  # -5% to +20% on test set
  total_trades: 50-200  # Number of trades
  win_rate: 0.45-0.60  # 45-60% win rate
  profit_factor: 0.8-2.5  # Gross profit / gross loss
  sharpe_ratio: -0.5 to 2.0  # Risk-adjusted return
  max_drawdown: 0.05-0.25  # Maximum drawdown (5-25%)
```

### Realistic Expectations

**IMPORTANT**: This is an EXTREMELY difficult problem. Most results will be:

- **Accuracy**: 50-58% (anything >55% is excellent)
- **Profit Factor**: <1.5 initially (needs optimization)
- **Sharpe Ratio**: Often negative initially
- **Overfitting**: Very common - validate rigorously

Even with powerful models, **most configurations will lose money** after transaction costs.

## Next Steps

### 1. Hyperparameter Tuning

Edit `config/config.yaml` to adjust:
- Batch size, learning rates
- Model architectures (hidden sizes, layers)
- Risk parameters (stop loss, take profit)
- Confidence thresholds

### 2. Extended Training

Train for more epochs or with different regimes:

```bash
# Train for 50 epochs
python train.py --epochs 50

# Train only on London/NY overlap (high liquidity)
# Edit config.yaml: sessions.focus_session
```

### 3. Walk-Forward Validation

Test on multiple time periods to check for overfitting:

```python
# Use different date ranges in config.yaml
data_collection:
  start_date: "2023-01-01"
  end_date: "2024-01-01"
```

### 4. Paper Trading

**DO NOT trade live money yet!**

Set up paper trading to monitor live performance vs backtest:

```bash
# (Implementation coming in future update)
python paper_trade.py
```

Monitor for minimum 3 months before considering real capital.

### 5. Live Trading (Advanced)

**ONLY after**:
- ✅ Positive results on test set
- ✅ 3+ months successful paper trading
- ✅ Understanding all risks
- ✅ Starting with $500-1000 max

```bash
# (Implementation coming in future update)
python live_trade.py --capital 500
```

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size in `config/config.yaml`:

```yaml
dataloader:
  batch_size: 128  # Reduce from 256
  sequence_length: 500  # Reduce from 1000
```

### Slow Training (CPU)

Use smaller models:

```python
# Edit train.py, reduce model sizes:
regime_detector = RegimeDetector(d_model=128, num_layers=2)  # Was 256, 4
lstm = LSTMPredictor(hidden_size=128, num_layers=2)  # Was 256, 3
```

### NaN Loss

Check for:
- Inf/NaN values in data (should be cleaned automatically)
- Learning rate too high (reduce to 1e-5)
- Exploding gradients (add gradient clipping)

### Poor Backtest Results

Normal! Try:
- Adjust confidence threshold (0.65 → 0.70 or 0.75)
- Filter trades by regime (only trade in favorable regimes)
- Adjust risk parameters (wider stops, smaller position size)
- More training epochs

## Configuration Reference

Key settings in `config/config.yaml`:

```yaml
# Data
data_collection:
  start_date: "2022-01-01"
  end_date: "2025-01-01"

# Training
dataloader:
  batch_size: 256
  sequence_length: 1000

# Risk Management
risk:
  max_spread_pips: 1.5
  stop_loss_atr_multiple: 1.5
  take_profit_atr_multiple: 2.5
  min_confidence: 0.65

# Hardware
hardware:
  device: "cuda"  # or "cpu"
  mixed_precision: true
  num_workers: 4
```

## Support

Found a bug? Have questions?

- Check `hft_train.log` for detailed logs
- Review `README.md` for architecture details
- Check GitHub issues (if applicable)

## ⚠️ Risk Warning

**TRADING IS RISKY**

- You can lose all your capital
- Past performance ≠ future results
- Transaction costs often exceed profits
- Models degrade over time
- News events cause unpredictable moves

**NEVER**:
- Trade with money you can't afford to lose
- Skip paper trading phase
- Ignore risk management
- Trade during major news events
- Use high leverage

**Start small, validate thoroughly, and scale gradually.**
