import numpy as np
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
import pickle
import os

@dataclass
class NetworkConfig:
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    learning_rate: float = 0.01
    momentum: float = 0.9
    dropout_rate: float = 0.2
    batch_size: int = 32
    activation: str = 'relu'

class NeuralNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_network()
        
    def _initialize_network(self):
        """Initialize network weights and biases with Xavier/Glorot initialization"""
        self.weights = []
        self.biases = []
        self.velocities = []  # For momentum
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(self.config.input_size, self.config.hidden_sizes[0]) 
                          * np.sqrt(2.0 / self.config.input_size))
        self.biases.append(np.zeros((1, self.config.hidden_sizes[0])))
        self.velocities.append(np.zeros_like(self.weights[0]))
        
        # Hidden layers
        for i in range(len(self.config.hidden_sizes) - 1):
            self.weights.append(np.random.randn(self.config.hidden_sizes[i], self.config.hidden_sizes[i + 1])
                              * np.sqrt(2.0 / self.config.hidden_sizes[i]))
            self.biases.append(np.zeros((1, self.config.hidden_sizes[i + 1])))
            self.velocities.append(np.zeros_like(self.weights[-1]))
        
        # Last hidden to output layer
        self.weights.append(np.random.randn(self.config.hidden_sizes[-1], self.config.output_size)
                          * np.sqrt(2.0 / self.config.hidden_sizes[-1]))
        self.biases.append(np.zeros((1, self.config.output_size)))
        self.velocities.append(np.zeros_like(self.weights[-1]))
        
        self.dropout_masks = [None] * len(self.weights)
    
    def _activation(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Apply activation function"""
        if self.config.activation == 'relu':
            if derivative:
                return np.where(x > 0, 1, 0)
            return np.maximum(0, x)
        elif self.config.activation == 'sigmoid':
            if derivative:
                return x * (1 - x)
            return 1 / (1 + np.exp(-x))
        elif self.config.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(x) ** 2
            return np.tanh(x)
    
    def _apply_dropout(self, layer: np.ndarray, layer_idx: int) -> np.ndarray:
        """Apply dropout regularization"""
        if self.training:
            mask = np.random.binomial(1, 1 - self.config.dropout_rate, size=layer.shape) / (1 - self.config.dropout_rate)
            self.dropout_masks[layer_idx] = mask
            return layer * mask
        return layer
    
    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
        """Forward propagation with dropout"""
        self.training = training
        self.activations = [X]
        self.z_values = []
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self._activation(z)
            a = self._apply_dropout(a, i)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self._activation(z)
        self.activations.append(output)
        
        return self.activations, output
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> float:
        """Backward propagation with momentum"""
        m = X.shape[0]
        delta = self.activations[-1] - y
        cost = np.mean(np.sum(-y * np.log(self.activations[-1] + 1e-8) - 
                            (1 - y) * np.log(1 - self.activations[-1] + 1e-8)))
        
        # Backpropagate error
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights with momentum
            self.velocities[i] = self.config.momentum * self.velocities[i] - self.config.learning_rate * dW
            self.weights[i] += self.velocities[i]
            self.biases[i] -= self.config.learning_rate * db
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation(self.z_values[i-1], derivative=True)
                if self.dropout_masks[i-1] is not None:
                    delta *= self.dropout_masks[i-1]
        
        return cost
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dict:
        """Train the network with validation"""
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, len(X), self.config.batch_size):
                X_batch = X_shuffled[i:i + self.config.batch_size]
                y_batch = y_shuffled[i:i + self.config.batch_size]
                
                self.forward(X_batch)
                loss = self.backward(X_batch, y_batch)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                _, val_output = self.forward(X_val, training=False)
                val_loss = np.mean(np.sum(-y_val * np.log(val_output + 1e-8) - 
                                        (1 - y_val) * np.log(1 - val_output + 1e-8)))
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_weights('best_model.pkl')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            history['train_loss'].append(loss)
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        _, output = self.forward(X, training=False)
        return output
    
    def save_weights(self, path: str):
        """Save model weights"""
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'config': self.config
            }, f)
    
    @classmethod
    def load_weights(cls, path: str) -> 'NeuralNetwork':
        """Load model weights"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(data['config'])
        model.weights = data['weights']
        model.biases = data['biases']
        return model 