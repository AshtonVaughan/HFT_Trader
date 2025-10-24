# HFT Trader - High-Frequency Trading System

Advanced machine learning-based forex trading system with regime detection and ensemble predictions.

## Architecture

### Models
- **Regime Detector**: Transformer-based market regime classifier
- **Specialized Predictors**: LSTM, GRU, CNN-LSTM, Transformer-XL
- **Meta-Learner**: Attention-based ensemble

### Features (117 total)
- Price/Volume: 35 features
- Technical Indicators: 38 features
- Time Features: 18 features
- Market Context: 21 features

## Quick Start

### Setup
```bash
pip install torch pandas numpy pyyaml tqdm scikit-learn yfinance
```

### Collect Data
```bash
python collect_data.py --use-ohlc-only
```

### Preprocess
```bash
python preprocess_all.py
```

### Train (requires GPU)
```bash
python train_enhanced.py --config config/config.yaml
```

## Cloud GPU Training

1. Upload project to cloud instance
2. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
3. Train: `python train_enhanced.py`

## Model Specs
- Total Parameters: 10.9M
- Training Data: 7,856 samples
- Test Data: 1,684 samples
- Sequence Length: 1000 bars

## Data
- EURUSD: 11,223 bars (1h, Jan 2024 - Oct 2025)
- Cross-pairs: EURGBP, GBPUSD
- Context: DXY, GOLD

## Testing
```bash
pytest tests/ -v
```

## License
MIT
