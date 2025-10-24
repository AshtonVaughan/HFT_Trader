"""
Master Pipeline Script

Runs the complete HFT_Trader pipeline from start to finish:
1. Data collection
2. Data validation
3. Feature engineering
4. Preprocessing
5. Model training (enhanced)
6. Hyperparameter optimization (optional)
7. RL training (optional)
8. Walk-forward validation
9. Model deployment
10. Monitoring setup

Usage:
    python run_full_pipeline.py --full                    # Complete pipeline
    python run_full_pipeline.py --quick                   # Skip optimization
    python run_full_pipeline.py --deploy-only             # Only deploy
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
from datetime import datetime

from utils.logger import setup_logger, logger


class PipelineRunner:
    """Master pipeline orchestrator."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        setup_logger(level="INFO", log_file="pipeline.log")

    def run_command(self, command: list, description: str):
        """Run a command and log output."""
        logger.info("="*80)
        logger.info(f"STEP: {description}")
        logger.info("="*80)
        logger.info(f"Command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )

            if result.stdout:
                logger.info(f"Output: {result.stdout}")

            logger.info(f"✓ {description} completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description} failed!")
            logger.error(f"Error: {e.stderr}")
            return False

    def step_1_data_collection(self, use_tick_data: bool = False):
        """Step 1: Collect market data."""
        if use_tick_data:
            cmd = [sys.executable, "collect_data.py", "--use-tick-data"]
        else:
            cmd = [sys.executable, "collect_data.py", "--use-ohlc-only"]

        return self.run_command(cmd, "Data Collection")

    def step_2_data_validation(self):
        """Step 2: Validate data quality."""
        cmd = [sys.executable, "-m", "data.preprocessors.data_validator"]
        return self.run_command(cmd, "Data Validation")

    def step_3_preprocessing(self):
        """Step 3: Feature engineering and preprocessing."""
        cmd = [sys.executable, "preprocess_all.py"]
        return self.run_command(cmd, "Data Preprocessing & Feature Engineering")

    def step_4_hyperparameter_optimization(self, trials: int = 50):
        """Step 4: Hyperparameter optimization (optional)."""
        cmd = [sys.executable, "optimize_hyperparams.py", "--trials", str(trials)]
        return self.run_command(cmd, f"Hyperparameter Optimization ({trials} trials)")

    def step_5_model_training(self, use_optimized: bool = False):
        """Step 5: Train models."""
        if use_optimized and Path("config/optimized_config.yaml").exists():
            cmd = [sys.executable, "train_enhanced.py", "--config", "config/optimized_config.yaml"]
            desc = "Model Training (with optimized hyperparameters)"
        else:
            cmd = [sys.executable, "train_enhanced.py"]
            desc = "Model Training (enhanced)"

        return self.run_command(cmd, desc)

    def step_6_rl_training(self):
        """Step 6: RL agent training (optional)."""
        cmd = [sys.executable, "train_rl.py"]
        return self.run_command(cmd, "RL Agent Training (PPO)")

    def step_7_walk_forward(self):
        """Step 7: Walk-forward validation."""
        cmd = [
            sys.executable,
            "walk_forward_backtest.py",
            "--train-months", "6",
            "--test-months", "1",
            "--step-months", "1"
        ]
        return self.run_command(cmd, "Walk-Forward Validation")

    def step_8_test_models(self):
        """Step 8: Run test suite."""
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=html"]
        return self.run_command(cmd, "Running Test Suite")

    def step_9_deploy(self):
        """Step 9: Deploy models."""
        logger.info("="*80)
        logger.info("DEPLOYMENT OPTIONS")
        logger.info("="*80)
        logger.info("\n1. Local API Server:")
        logger.info("   python deployment/api_server.py")
        logger.info("\n2. Docker Deployment:")
        logger.info("   docker-compose up -d")
        logger.info("\n3. View API docs:")
        logger.info("   http://localhost:8000/docs")
        logger.info("\n4. View monitoring:")
        logger.info("   Grafana: http://localhost:3000")
        logger.info("   Prometheus: http://localhost:9090")

        return True

    def run_full_pipeline(
        self,
        use_tick_data: bool = False,
        optimize: bool = False,
        train_rl: bool = False,
        run_tests: bool = True
    ):
        """Run complete pipeline."""
        start_time = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("HFT_TRADER FULL PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {start_time}")
        logger.info(f"Configuration: {self.config_path}")
        logger.info(f"Tick data: {use_tick_data}")
        logger.info(f"Optimization: {optimize}")
        logger.info(f"RL training: {train_rl}")
        logger.info("="*80 + "\n")

        steps = []

        # Step 1: Data collection
        if not self.step_1_data_collection(use_tick_data):
            logger.error("Pipeline failed at data collection")
            return False
        steps.append("Data Collection")

        # Step 2: Data validation
        if not self.step_2_data_validation():
            logger.warning("Data validation had warnings (proceeding anyway)")
        steps.append("Data Validation")

        # Step 3: Preprocessing
        if not self.step_3_preprocessing():
            logger.error("Pipeline failed at preprocessing")
            return False
        steps.append("Preprocessing")

        # Step 4: Hyperparameter optimization (optional)
        if optimize:
            if not self.step_4_hyperparameter_optimization(trials=50):
                logger.warning("Optimization failed (using default config)")
            else:
                steps.append("Hyperparameter Optimization")

        # Step 5: Model training
        if not self.step_5_model_training(use_optimized=optimize):
            logger.error("Pipeline failed at model training")
            return False
        steps.append("Model Training")

        # Step 6: RL training (optional)
        if train_rl:
            if not self.step_6_rl_training():
                logger.warning("RL training failed (models still usable)")
            else:
                steps.append("RL Training")

        # Step 7: Walk-forward validation
        if not self.step_7_walk_forward():
            logger.warning("Walk-forward validation failed")
        else:
            steps.append("Walk-Forward Validation")

        # Step 8: Tests
        if run_tests:
            if not self.step_8_test_models():
                logger.warning("Some tests failed")
            else:
                steps.append("Testing")

        # Step 9: Deployment info
        self.step_9_deploy()
        steps.append("Deployment Info")

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"End time: {end_time}")
        logger.info(f"Total duration: {duration}")
        logger.info(f"\nCompleted steps:")
        for i, step in enumerate(steps, 1):
            logger.info(f"  {i}. {step} ✓")

        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS:")
        logger.info("="*80)
        logger.info("1. Review results in results/ directory")
        logger.info("2. Check TensorBoard: tensorboard --logdir runs/")
        logger.info("3. Deploy API: python deployment/api_server.py")
        logger.info("4. Or use Docker: docker-compose up -d")
        logger.info("="*80 + "\n")

        return True

    def run_quick_pipeline(self):
        """Run quick pipeline (no optimization, no RL)."""
        return self.run_full_pipeline(
            use_tick_data=False,
            optimize=False,
            train_rl=False,
            run_tests=True
        )


def main():
    parser = argparse.ArgumentParser(
        description="HFT_Trader Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline with optimization and RL')
    parser.add_argument('--quick', action='store_true',
                       help='Quick pipeline (OHLC data, no optimization)')
    parser.add_argument('--optimize', action='store_true',
                       help='Include hyperparameter optimization')
    parser.add_argument('--rl', action='store_true',
                       help='Include RL training')
    parser.add_argument('--tick-data', action='store_true',
                       help='Use tick data (slow, 2-4 hours)')
    parser.add_argument('--deploy-only', action='store_true',
                       help='Only show deployment instructions')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file')

    args = parser.parse_args()

    runner = PipelineRunner(config_path=args.config)

    if args.deploy_only:
        runner.step_9_deploy()
        return

    if args.full:
        success = runner.run_full_pipeline(
            use_tick_data=args.tick_data,
            optimize=True,
            train_rl=True,
            run_tests=True
        )
    elif args.quick:
        success = runner.run_quick_pipeline()
    else:
        success = runner.run_full_pipeline(
            use_tick_data=args.tick_data,
            optimize=args.optimize,
            train_rl=args.rl,
            run_tests=True
        )

    if success:
        logger.info("✓ Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Pipeline failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
