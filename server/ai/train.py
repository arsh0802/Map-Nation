import os
import sys
import logging
import time
from datetime import datetime
import json
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_service import AIService
from config.config import AIConfig
from utils.helpers import (
    setup_logging,
    save_training_data,
    load_training_data,
    split_data,
    create_experiment_dir,
    save_experiment_results,
    format_training_time,
    calculate_model_size,
    format_model_size
)

def train_model(
    data_path: str,
    config_path: str = None,
    experiment_name: str = None
):
    """Train the AI model"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = AIConfig.from_json(config_path)
    else:
        config = AIConfig()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir()
    if experiment_name:
        experiment_dir = os.path.join(os.path.dirname(experiment_dir), experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config.to_json(os.path.join(experiment_dir, 'config.json'))
    
    # Load training data
    logger.info(f"Loading training data from {data_path}")
    training_data = load_training_data(data_path)
    
    # Split data
    train_data, val_data, test_data = split_data(
        training_data,
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    # Save split data
    save_training_data(train_data, os.path.join(experiment_dir, 'train_data.json'))
    save_training_data(val_data, os.path.join(experiment_dir, 'val_data.json'))
    save_training_data(test_data, os.path.join(experiment_dir, 'test_data.json'))
    
    # Initialize AI service
    logger.info("Initializing AI service")
    ai_service = AIService()
    ai_service.initialize(**config.get_model_config())
    
    # Prepare training data
    train_texts = [item['text'] for item in train_data]
    val_texts = [item['text'] for item in val_data]
    
    # Train model
    logger.info("Starting model training")
    start_time = time.time()
    
    history = ai_service.train(
        texts=train_texts,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {format_training_time(training_time)}")
    
    # Evaluate model
    logger.info("Evaluating model")
    test_texts = [item['text'] for item in test_data]
    metrics = ai_service.evaluate(test_texts)
    
    # Save model
    logger.info("Saving model")
    ai_service.save(os.path.join(experiment_dir, 'model'))
    
    # Save results
    results = {
        'training_time': training_time,
        'training_time_formatted': format_training_time(training_time),
        'model_size': format_model_size(calculate_model_size(ai_service.model.weights)),
        'metrics': metrics,
        'history': history
    }
    save_experiment_results(results, experiment_dir)
    
    logger.info("Training completed successfully")
    logger.info(f"Results saved to {experiment_dir}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AI model')
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--name', help='Experiment name')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        config_path=args.config,
        experiment_name=args.name
    ) 