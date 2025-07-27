import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import json
import pickle
import os
from datetime import datetime
from .neural_network import NeuralNetwork, NetworkConfig
from .data_processor import DataProcessor, ProcessorConfig

class AIService:
    def __init__(self, model_dir: str = 'server/ai/models'):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.processor = None
        self.model = None
        self.training_history = []
        
    def initialize(self, 
                  vocab_size: int = 10000,
                  embedding_dim: int = 300,
                  hidden_sizes: List[int] = [256, 128],
                  learning_rate: float = 0.001):
        """Initialize the AI service with new model"""
        # Initialize data processor
        processor_config = ProcessorConfig(
            max_vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )
        self.processor = DataProcessor(processor_config)
        
        # Initialize neural network
        network_config = NetworkConfig(
            input_size=embedding_dim,
            hidden_sizes=hidden_sizes,
            output_size=vocab_size,
            learning_rate=learning_rate
        )
        self.model = NeuralNetwork(network_config)
        
    def train(self, 
             texts: List[str],
             epochs: int = 100,
             batch_size: int = 32,
             validation_split: float = 0.1) -> Dict:
        """Train the model on texts"""
        if not self.processor or not self.model:
            raise ValueError("Service not initialized. Call initialize() first.")
        
        # Prepare data
        self.processor.fit(texts)
        sequences, embeddings = self.processor.transform(texts)
        
        # Split into train and validation
        split_idx = int(len(embeddings) * (1 - validation_split))
        X_train, X_val = embeddings[:split_idx], embeddings[split_idx:]
        y_train, y_val = sequences[:split_idx], sequences[split_idx:]
        
        # Train model
        history = self.model.train(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val)
        )
        
        # Save training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'history': history
        })
        
        return history
    
    def generate_response(self, 
                         input_text: str,
                         max_length: int = 100,
                         temperature: float = 0.7) -> str:
        """Generate response for input text"""
        if not self.processor or not self.model:
            raise ValueError("Service not initialized. Call initialize() first.")
        
        # Get input embedding
        input_embedding = self.processor.get_embeddings(input_text)
        
        # Generate sequence
        sequence = []
        current_embedding = input_embedding
        
        for _ in range(max_length):
            # Get model prediction
            output = self.model.predict(np.array([current_embedding]))
            
            # Apply temperature
            output = output / temperature
            output = np.exp(output) / np.sum(np.exp(output))
            
            # Sample next word
            next_word_idx = np.random.choice(len(output[0]), p=output[0])
            sequence.append(next_word_idx)
            
            # Update embedding
            current_embedding = self.processor.word_embeddings[next_word_idx]
            
            # Stop if we predict end token
            if next_word_idx == self.processor.vocabulary.get('<END>', -1):
                break
        
        # Convert sequence to text
        words = [self.processor.reverse_vocabulary[idx] for idx in sequence]
        return ' '.join(words)
    
    def save(self, name: str):
        """Save the entire service state"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.model_dir, f"{name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save processor
        self.processor.save(os.path.join(save_dir, 'processor.pkl'))
        
        # Save model
        self.model.save_weights(os.path.join(save_dir, 'model.pkl'))
        
        # Save training history
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(self.training_history, f)
        
        self.logger.info(f"Saved model to {save_dir}")
    
    @classmethod
    def load(cls, path: str) -> 'AIService':
        """Load service from saved state"""
        service = cls(os.path.dirname(path))
        
        # Load processor
        service.processor = DataProcessor.load(
            os.path.join(path, 'processor.pkl')
        )
        
        # Load model
        service.model = NeuralNetwork.load_weights(
            os.path.join(path, 'model.pkl')
        )
        
        # Load training history
        with open(os.path.join(path, 'history.json'), 'r') as f:
            service.training_history = json.load(f)
        
        return service
    
    def evaluate(self, test_texts: List[str]) -> Dict:
        """Evaluate model performance"""
        if not self.processor or not self.model:
            raise ValueError("Service not initialized. Call initialize() first.")
        
        sequences, embeddings = self.processor.transform(test_texts)
        predictions = self.model.predict(embeddings)
        
        # Calculate metrics
        accuracy = np.mean(np.argmax(predictions, axis=1) == sequences)
        perplexity = np.exp(-np.mean(np.log(predictions + 1e-8)))
        
        return {
            'accuracy': float(accuracy),
            'perplexity': float(perplexity)
        }
    
    def get_training_history(self) -> List[Dict]:
        """Get training history"""
        return self.training_history
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.processor or not self.model:
            raise ValueError("Service not initialized. Call initialize() first.")
        
        return {
            'vocab_size': len(self.processor.vocabulary),
            'embedding_dim': self.processor.config.embedding_dim,
            'hidden_sizes': self.model.config.hidden_sizes,
            'learning_rate': self.model.config.learning_rate,
            'training_history': len(self.training_history)
        } 