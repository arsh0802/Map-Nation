import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import json
import logging
from collections import Counter
import pickle
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ProcessorConfig:
    max_vocab_size: int = 10000
    max_sequence_length: int = 100
    min_word_frequency: int = 2
    embedding_dim: int = 300
    use_stemming: bool = True
    use_lemmatization: bool = True
    remove_stopwords: bool = True
    batch_size: int = 32

class DataProcessor:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.vocabulary = {}
        self.reverse_vocabulary = {}
        self.word_embeddings = {}
        self.stopwords = set()
        self._load_stopwords()
        
    def _load_stopwords(self):
        """Load stopwords from file"""
        try:
            with open('server/ai/data/stopwords.txt', 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f)
        except FileNotFoundError:
            self.logger.warning("Stopwords file not found. Continuing without stopwords.")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return text.split()
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        if self.config.remove_stopwords:
            return [token for token in tokens if token not in self.stopwords]
        return tokens
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(self._clean_text(text))
            word_counts.update(tokens)
        
        # Filter by frequency
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= self.config.min_word_frequency}
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top max_vocab_size words
        vocabulary = {word: idx + 1 for idx, (word, _) in 
                     enumerate(sorted_words[:self.config.max_vocab_size])}
        
        # Add special tokens
        vocabulary['<PAD>'] = 0
        vocabulary['<UNK>'] = len(vocabulary)
        
        self.vocabulary = vocabulary
        self.reverse_vocabulary = {idx: word for word, idx in vocabulary.items()}
    
    def _generate_embeddings(self):
        """Generate word embeddings using co-occurrence matrix"""
        # Initialize co-occurrence matrix
        co_occurrence = np.zeros((len(self.vocabulary), len(self.vocabulary)))
        window_size = 5
        
        # Build co-occurrence matrix
        for text in self.training_texts:
            tokens = self._tokenize(self._clean_text(text))
            for i, token in enumerate(tokens):
                if token in self.vocabulary:
                    start = max(0, i - window_size)
                    end = min(len(tokens), i + window_size + 1)
                    for j in range(start, end):
                        if i != j and tokens[j] in self.vocabulary:
                            co_occurrence[self.vocabulary[token]][self.vocabulary[tokens[j]]] += 1
        
        # Apply SVD to get embeddings
        U, S, V = np.linalg.svd(co_occurrence)
        self.word_embeddings = U[:, :self.config.embedding_dim]
    
    def preprocess_text(self, text: str) -> List[int]:
        """Convert text to sequence of word indices"""
        tokens = self._tokenize(self._clean_text(text))
        tokens = self._remove_stopwords(tokens)
        
        # Convert to indices
        indices = [self.vocabulary.get(token, self.vocabulary['<UNK>']) 
                  for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.config.max_sequence_length:
            indices.extend([self.vocabulary['<PAD>']] * 
                         (self.config.max_sequence_length - len(indices)))
        else:
            indices = indices[:self.config.max_sequence_length]
        
        return indices
    
    def preprocess_batch(self, texts: List[str]) -> np.ndarray:
        """Process a batch of texts in parallel"""
        with ThreadPoolExecutor() as executor:
            sequences = list(executor.map(self.preprocess_text, texts))
        return np.array(sequences)
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get word embeddings for text"""
        indices = self.preprocess_text(text)
        return np.mean([self.word_embeddings[idx] for idx in indices], axis=0)
    
    def fit(self, texts: List[str]):
        """Build vocabulary and generate embeddings"""
        self.training_texts = texts
        self._build_vocabulary(texts)
        self._generate_embeddings()
    
    def transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Transform texts to sequences and embeddings"""
        sequences = self.preprocess_batch(texts)
        embeddings = np.array([self.get_embeddings(text) for text in texts])
        return sequences, embeddings
    
    def save(self, path: str):
        """Save processor state"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocabulary': self.vocabulary,
                'reverse_vocabulary': self.reverse_vocabulary,
                'word_embeddings': self.word_embeddings,
                'config': self.config
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'DataProcessor':
        """Load processor state"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        processor = cls(data['config'])
        processor.vocabulary = data['vocabulary']
        processor.reverse_vocabulary = data['reverse_vocabulary']
        processor.word_embeddings = data['word_embeddings']
        return processor 