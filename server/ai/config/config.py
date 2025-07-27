from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os

@dataclass
class AIConfig:
    # Model configuration
    vocab_size: int = 10000
    embedding_dim: int = 300
    hidden_sizes: List[int] = [256, 128]
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.1
    
    # Training configuration
    early_stopping_patience: int = 5
    min_delta: float = 0.001
    max_sequence_length: int = 100
    temperature: float = 0.7
    
    # Data processing configuration
    min_word_frequency: int = 2
    use_stemming: bool = True
    use_lemmatization: bool = True
    remove_stopwords: bool = True
    
    # Model saving configuration
    save_best_only: bool = True
    save_frequency: int = 10
    
    @classmethod
    def from_json(cls, path: str) -> 'AIConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate
        }
    
    def get_processor_config(self) -> Dict[str, Any]:
        """Get data processor configuration"""
        return {
            'max_vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'min_word_frequency': self.min_word_frequency,
            'embedding_dim': self.embedding_dim,
            'use_stemming': self.use_stemming,
            'use_lemmatization': self.use_lemmatization,
            'remove_stopwords': self.remove_stopwords,
            'batch_size': self.batch_size
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'min_delta': self.min_delta,
            'save_best_only': self.save_best_only,
            'save_frequency': self.save_frequency
        } 