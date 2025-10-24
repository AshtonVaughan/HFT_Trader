"""
Dual GPU Training Script for HFT_Trader

Hybrid approach:
- Parallel model training: Train LSTM + GRU simultaneously on separate GPUs
- Data parallelism: Use both GPUs for CNN-LSTM and Transformer-XL
"""

import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import argparse
from tqdm import tqdm
import threading
from queue import Queue

from data.loaders.dataset import create_dataloaders
from models.regime_detector.transformer_detector import RegimeDetector
from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
from models.predictors.transformer_xl import TransformerXLPredictor
from models.meta_learner.attention_meta_learner import AttentionMetaLearner, EnsemblePredictor
from utils.logger import setup_logger, logger


class DualGPUTrainer:
    """
    Hybrid dual-GPU training:
    - Small models (LSTM, GRU): Train in parallel on separate GPUs
    - Large models (CNN-LSTM, Transformer-XL): Data parallelism across both GPUs
    """

    def __init__(self, config: dict):
        self.config = config
        self.device0 = torch.device('cuda:0')
        self.device1 = torch.device('cuda:1')

        # Check both GPUs available
        if torch.cuda.device_count() < 2:
            raise RuntimeError(f"Need 2 GPUs, found {torch.cuda.device_count()}")

        logger.info(f"GPU 0: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 1: {torch.cuda.get_device_name(1)}")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB x 2")

        self.use_amp = config.get('training', {}).get('use_amp', True)
        self.accumulation_steps = config.get('training', {}).get('gradient_accumulation_steps', 4)

    def train(self):
        """Main training loop with hybrid dual-GPU strategy."""

        logger.info("\n" + "="*80)
        logger.info("DUAL GPU TRAINING - HYBRID MODE")
        logger.info("="*80)

        # 1. Load data
        logger.info("\n1. Loading preprocessed data...")
        train_df = pd.read_parquet('processed_data/train.parquet')
        val_df = pd.read_parquet('processed_data/val.parquet')
        test_df = pd.read_parquet('processed_data/test.parquet')

        logger.info(f"   Train: {len(train_df):,} samples")
        logger.info(f"   Val: {len(val_df):,} samples")
        logger.info(f"   Test: {len(test_df):,} samples")

        # 2. Create dataloaders with larger batch size (dual GPU can handle it)
        logger.info("\n2. Creating dataloaders...")
        batch_size = self.config.get('dataloader', {}).get('batch_size', 256)

        train_loader, val_loader, test_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=batch_size,
            sequence_length=self.config.get('dataloader', {}).get('sequence_length', 1000),
            num_workers=8,  # More workers for dual GPU
            pin_memory=True
        )

        feature_dim = train_loader.dataset.get_feature_dim()

        # 3. Build models
        logger.info("\n3. Building models...")

        # Don't move to device yet - will be assigned to specific GPUs later
        regime_detector = RegimeDetector(
            input_size=feature_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        )

        lstm = LSTMPredictor(input_size=feature_dim, hidden_size=256, num_layers=3)
        gru = GRUPredictor(input_size=feature_dim, hidden_size=256, num_layers=3)
        cnn_lstm = CNNLSTMPredictor(input_size=feature_dim, cnn_channels=[64, 128, 256], lstm_hidden_size=256)
        transformer_xl = TransformerXLPredictor(input_size=feature_dim, d_model=256, nhead=8, num_layers=4)

        specialized_models = [lstm, gru, cnn_lstm, transformer_xl]

        meta_learner = AttentionMetaLearner(num_models=4, embedding_dim=128)
        ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner)

        total_params = sum(p.numel() for p in ensemble.parameters())
        logger.info(f"  Total parameters: {total_params:,}")

        # 4. Train Regime Detector on GPU 0
        logger.info("\n4. Training Regime Detector on GPU 0...")
        regime_detector = regime_detector.to(self.device0)
        self._train_single_gpu(
            model=regime_detector,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name='regime_detector',
            device=self.device0,
            epochs=20,
            lr=1e-4
        )

        # 5. Train LSTM + GRU in parallel (separate GPUs)
        logger.info("\n5. Training LSTM (GPU 0) and GRU (GPU 1) in parallel...")

        lstm = lstm.to(self.device0)
        gru = gru.to(self.device1)

        # Create separate data loaders for each GPU (avoid data contention)
        train_loader_0, val_loader_0, _ = create_dataloaders(
            train_df=train_df, val_df=val_df, test_df=test_df,
            batch_size=batch_size, sequence_length=1000, num_workers=4, pin_memory=True
        )
        train_loader_1, val_loader_1, _ = create_dataloaders(
            train_df=train_df, val_df=val_df, test_df=test_df,
            batch_size=batch_size, sequence_length=1000, num_workers=4, pin_memory=True
        )

        self._train_parallel(
            model1=lstm,
            model2=gru,
            train_loader1=train_loader_0,
            val_loader1=val_loader_0,
            train_loader2=train_loader_1,
            val_loader2=val_loader_1,
            model_name1='lstm',
            model_name2='gru',
            device1=self.device0,
            device2=self.device1,
            epochs=10,
            lr=1e-4
        )

        # Move LSTM and GRU to CPU to free GPU memory
        logger.info("\n   Moving trained models to CPU to free GPU memory...")
        lstm = lstm.cpu()
        gru = gru.cpu()

        # Clear GPU cache on both GPUs
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info(f"   GPU 0 free memory: {torch.cuda.mem_get_info(0)[0] / 1024**3:.2f} GB")
        logger.info(f"   GPU 1 free memory: {torch.cuda.mem_get_info(1)[0] / 1024**3:.2f} GB")

        # 6. Train CNN-LSTM with data parallelism (both GPUs)
        logger.info("\n6. Training CNN-LSTM with data parallelism (GPU 0+1)...")
        cnn_lstm = nn.DataParallel(cnn_lstm, device_ids=[0, 1]).to(self.device0)
        self._train_single_gpu(
            model=cnn_lstm,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name='cnn_lstm',
            device=self.device0,
            epochs=10,
            lr=1e-4
        )

        # Move CNN-LSTM to CPU and clear GPU memory
        logger.info("\n   Moving CNN-LSTM to CPU to free GPU memory...")
        cnn_lstm = cnn_lstm.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info(f"   GPU 0 free memory: {torch.cuda.mem_get_info(0)[0] / 1024**3:.2f} GB")
        logger.info(f"   GPU 1 free memory: {torch.cuda.mem_get_info(1)[0] / 1024**3:.2f} GB")

        # 7. Train Transformer-XL with data parallelism (both GPUs)
        logger.info("\n7. Training Transformer-XL with data parallelism (GPU 0+1)...")
        transformer_xl = nn.DataParallel(transformer_xl, device_ids=[0, 1]).to(self.device0)
        self._train_single_gpu(
            model=transformer_xl,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name='transformer_xl',
            device=self.device0,
            epochs=10,
            lr=1e-4
        )

        # Move Transformer-XL to CPU and clear GPU memory
        logger.info("\n   Moving Transformer-XL to CPU to free GPU memory...")
        transformer_xl = transformer_xl.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info(f"   GPU 0 free memory: {torch.cuda.mem_get_info(0)[0] / 1024**3:.2f} GB")
        logger.info(f"   GPU 1 free memory: {torch.cuda.mem_get_info(1)[0] / 1024**3:.2f} GB")

        # 8. Train Ensemble Meta-Learner on GPU 0
        logger.info("\n8. Training Ensemble Meta-Learner on GPU 0...")

        # Load trained specialist models from checkpoints
        lstm.load_state_dict(torch.load('checkpoints/lstm_best.pth'))
        gru.load_state_dict(torch.load('checkpoints/gru_best.pth'))

        # Unwrap DataParallel models
        if isinstance(cnn_lstm, nn.DataParallel):
            cnn_lstm_single = cnn_lstm.module
        else:
            cnn_lstm_single = cnn_lstm
        cnn_lstm_single.load_state_dict(torch.load('checkpoints/cnn_lstm_best.pth'))

        if isinstance(transformer_xl, nn.DataParallel):
            transformer_xl_single = transformer_xl.module
        else:
            transformer_xl_single = transformer_xl
        transformer_xl_single.load_state_dict(torch.load('checkpoints/transformer_xl_best.pth'))

        # Move all to GPU 0 for ensemble
        lstm = lstm.to(self.device0)
        gru = gru.to(self.device0)
        cnn_lstm_single = cnn_lstm_single.to(self.device0)
        transformer_xl_single = transformer_xl_single.to(self.device0)
        ensemble = ensemble.to(self.device0)

        # Freeze specialist models
        for model in [lstm, gru, cnn_lstm_single, transformer_xl_single]:
            for param in model.parameters():
                param.requires_grad = False

        # Update ensemble's specialist models
        ensemble.specialized_models = nn.ModuleList([lstm, gru, cnn_lstm_single, transformer_xl_single])

        self._train_single_gpu(
            model=ensemble,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name='ensemble',
            device=self.device0,
            epochs=5,
            lr=1e-4
        )

        # 9. Final evaluation
        logger.info("\n9. Final evaluation on test set...")
        self._evaluate_test(ensemble, test_loader, self.device0)

        logger.info("\n" + "="*80)
        logger.info("DUAL GPU TRAINING COMPLETE!")
        logger.info("="*80)

    def _train_parallel(
        self,
        model1, model2,
        train_loader1, val_loader1,
        train_loader2, val_loader2,
        model_name1, model_name2,
        device1, device2,
        epochs, lr
    ):
        """Train two models in parallel on separate GPUs using threading."""

        results_queue = Queue()

        def train_worker(model, train_loader, val_loader, model_name, device):
            """Worker function to train a single model."""
            try:
                self._train_single_gpu(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_name=model_name,
                    device=device,
                    epochs=epochs,
                    lr=lr
                )
                results_queue.put((model_name, 'success'))
            except Exception as e:
                results_queue.put((model_name, f'error: {str(e)}'))

        # Start both training threads
        thread1 = threading.Thread(
            target=train_worker,
            args=(model1, train_loader1, val_loader1, model_name1, device1)
        )
        thread2 = threading.Thread(
            target=train_worker,
            args=(model2, train_loader2, val_loader2, model_name2, device2)
        )

        thread1.start()
        thread2.start()

        # Wait for both to complete
        thread1.join()
        thread2.join()

        # Check results
        for _ in range(2):
            model_name, status = results_queue.get()
            if 'error' in status:
                logger.error(f"{model_name} training failed: {status}")
            else:
                logger.info(f"{model_name} training completed successfully")

    def _train_single_gpu(
        self,
        model,
        train_loader,
        val_loader,
        model_name,
        device,
        epochs,
        lr
    ):
        """Train a single model on specified GPU (same as original implementation)."""

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=self.config.get('training', {}).get('weight_decay', 1e-5)
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader) // self.accumulation_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        scaler = GradScaler(enabled=self.use_amp)
        criterion_direction = nn.CrossEntropyLoss()
        criterion_magnitude = nn.MSELoss()

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}")

            for batch_idx, (features, targets, regimes, target_signs) in enumerate(pbar):
                features = features.to(device)
                targets = targets.to(device)
                target_signs = target_signs.to(device)
                regimes = regimes.to(device)

                with autocast(enabled=self.use_amp):
                    outputs = model(features)

                    if isinstance(outputs, tuple):
                        regime_logits, volatility = outputs
                        loss_regime = criterion_direction(regime_logits, regimes)
                        loss_vol = criterion_magnitude(volatility.squeeze(), targets.abs())
                        loss = (loss_regime + 0.3 * loss_vol) / self.accumulation_steps
                    else:
                        loss_dir = criterion_direction(outputs['direction_logits'], target_signs)
                        loss_mag = criterion_magnitude(outputs['magnitude'].squeeze(), targets)
                        loss = (loss_dir + 0.5 * loss_mag) / self.accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * self.accumulation_steps
                pbar.set_postfix({'loss': loss.item() * self.accumulation_steps, 'lr': scheduler.get_last_lr()[0]})

            # Validation
            val_loss, val_acc = self._validate(
                model, val_loader, criterion_direction, criterion_magnitude, device
            )

            logger.info(
                f"   Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"   Early stopping triggered for {model_name}")
                    break

    def _validate(self, model, val_loader, criterion_direction, criterion_magnitude, device):
        """Validation loop."""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, targets, regimes, target_signs in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                target_signs = target_signs.to(device)
                regimes = regimes.to(device)

                outputs = model(features)

                if isinstance(outputs, tuple):
                    regime_logits, volatility = outputs
                    loss_regime = criterion_direction(regime_logits, regimes)
                    loss_vol = criterion_magnitude(volatility.squeeze(), targets.abs())
                    loss = loss_regime + 0.3 * loss_vol
                    preds = torch.argmax(regime_logits, dim=1)
                    correct += (preds == regimes).sum().item()
                else:
                    loss_dir = criterion_direction(outputs['direction_logits'], target_signs)
                    loss_mag = criterion_magnitude(outputs['magnitude'].squeeze(), targets)
                    loss = loss_dir + 0.5 * loss_mag
                    preds = torch.argmax(outputs['direction_logits'], dim=1)
                    correct += (preds == target_signs).sum().item()

                val_loss += loss.item()
                total += target_signs.size(0)

        return val_loss / len(val_loader), 100.0 * correct / total

    def _evaluate_test(self, model, test_loader, device):
        """Final test evaluation."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, targets, regimes, target_signs in test_loader:
                features = features.to(device)
                target_signs = target_signs.to(device)

                outputs = model(features)
                preds = torch.argmax(outputs['direction_logits'], dim=1)
                correct += (preds == target_signs).sum().item()
                total += target_signs.size(0)

        test_acc = 100.0 * correct / total
        logger.info(f"   Test Accuracy: {test_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train HFT models on dual GPUs")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    setup_logger(
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('log_file', 'hft_training.log')
    )

    # Create checkpoints directory
    Path('checkpoints').mkdir(exist_ok=True)

    trainer = DualGPUTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
