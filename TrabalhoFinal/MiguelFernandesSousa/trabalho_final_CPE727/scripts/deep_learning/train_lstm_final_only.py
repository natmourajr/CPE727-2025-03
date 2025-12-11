"""
Train LSTM on full AG News dataset with best hyperparameters from Stage 2
Skips Stage 1 and Stage 2 grid search.
"""
import mlflow
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader_agnews_tokenized import AGNewsTokenizedLoader
from src.models_deep import AGNewsLSTM, PyTorchClassifier
from src.config import MLRUNS_DIR
from src.utils import setup_logger
import numpy as np

logger = setup_logger(__name__)

# Best hyperparameters from Stage 2
BEST_PARAMS = {
    'learning_rate': 0.005,
    'embedding_dim': 100,
    'hidden_dim': 128,
    'dropout': 0.4,
    'bidirectional': True,
    'batch_size': 32,
    'epochs': 20,  # More epochs for final training
    'vocab_size': 10000
}

def main():
    logger.info("=" * 80)
    logger.info("LSTM (AG News) - FINAL TRAINING ONLY")
    logger.info("=" * 80)
    logger.info(f"\nBest hyperparameters from Stage 2:")
    for k, v in BEST_PARAMS.items():
        logger.info(f"  {k}: {v}")

    # Load full dataset
    logger.info("\nLoading full AG News dataset...")
    loader = AGNewsTokenizedLoader(
        max_vocab_size=BEST_PARAMS['vocab_size'],
        max_seq_length=200
    )
    X_train, y_train, X_val, y_val, X_test, y_test = loader.get_numpy_arrays()

    # Combine train + val
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])

    logger.info(f"Training set: {X_full.shape}")
    logger.info(f"Test set: {X_test.shape}")

    # Create model
    logger.info("\nCreating LSTM model...")
    model = AGNewsLSTM(
        vocab_size=BEST_PARAMS['vocab_size'],
        embedding_dim=BEST_PARAMS['embedding_dim'],
        hidden_dim=BEST_PARAMS['hidden_dim'],
        dropout=BEST_PARAMS['dropout'],
        bidirectional=BEST_PARAMS['bidirectional']
    )

    classifier = PyTorchClassifier(
        model=model,
        learning_rate=BEST_PARAMS['learning_rate'],
        batch_size=BEST_PARAMS['batch_size'],
        epochs=BEST_PARAMS['epochs'],
        verbose=True,
        use_long_tensor=True,
        log_to_mlflow=True
    )

    # Setup MLflow
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("ag_news_lstm_final")

    # Train
    logger.info(f"\nTraining for {BEST_PARAMS['epochs']} epochs...")
    logger.info("This will take approximately 60-90 minutes...")

    import time
    start_time = time.time()

    with mlflow.start_run(run_name="lstm_final_full_dataset"):
        mlflow.log_params(BEST_PARAMS)

        classifier.fit(X_full, y_full)

        # Evaluate
        y_pred = classifier.predict(X_test)
        test_accuracy = (y_pred == y_test).mean()

        train_time = time.time() - start_time

        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('train_time', train_time)

        # Save model
        try:
            mlflow.pytorch.log_model(classifier.model, "model")
            logger.info("\n✓ Model saved to MLflow")
        except Exception as e:
            logger.warning(f"Could not save model: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nTest Accuracy: {test_accuracy:.4f}")
    logger.info(f"Training time: {train_time/60:.1f} minutes")
    logger.info("\nView results:")
    logger.info("  mlflow ui")
    logger.info("  Open: http://localhost:5000")

if __name__ == "__main__":
    main()
