import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os
import logging
from datetime import datetime
import pickle

def setup_logging(log_dir: str = 'server/ai/logs'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'ai_service_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_training_data(data: List[Dict[str, Any]], path: str):
    """Save training data to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_training_data(path: str) -> List[Dict[str, Any]]:
    """Load training data from file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_data(data: List[Any], 
               train_ratio: float = 0.8,
               val_ratio: float = 0.1) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split data into train, validation and test sets"""
    np.random.shuffle(data)
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various metrics for model evaluation"""
    accuracy = np.mean(y_true == y_pred)
    precision = np.mean((y_true == 1) & (y_pred == 1)) / (np.mean(y_pred == 1) + 1e-8)
    recall = np.mean((y_true == 1) & (y_pred == 1)) / (np.mean(y_true == 1) + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

def save_model_weights(weights: Dict[str, np.ndarray], path: str):
    """Save model weights to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(weights, f)

def load_model_weights(path: str) -> Dict[str, np.ndarray]:
    """Load model weights from file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def create_experiment_dir(base_dir: str = 'server/ai/experiments') -> str:
    """Create directory for new experiment"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_experiment_results(results: Dict[str, Any], experiment_dir: str):
    """Save experiment results"""
    results_file = os.path.join(experiment_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

def load_experiment_results(experiment_dir: str) -> Dict[str, Any]:
    """Load experiment results"""
    results_file = os.path.join(experiment_dir, 'results.json')
    with open(results_file, 'r') as f:
        return json.load(f)

def get_latest_experiment(base_dir: str = 'server/ai/experiments') -> str:
    """Get path to latest experiment directory"""
    experiments = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    if not experiments:
        return None
    return os.path.join(base_dir, max(experiments))

def format_training_time(seconds: float) -> str:
    """Format training time in human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_model_size(model_weights: Dict[str, np.ndarray]) -> int:
    """Calculate model size in bytes"""
    total_size = 0
    for weight in model_weights.values():
        total_size += weight.nbytes
    return total_size

def format_model_size(size_bytes: int) -> str:
    """Format model size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB" 