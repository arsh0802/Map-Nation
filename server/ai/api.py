from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import json
from datetime import datetime

from core.ai_service import AIService
from config.config import AIConfig
from utils.helpers import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize AI service
ai_service = None

def load_model(model_path: str):
    """Load AI model"""
    global ai_service
    try:
        ai_service = AIService.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.route('/api/ai/initialize', methods=['POST'])
def initialize():
    """Initialize AI service with new model"""
    try:
        config_data = request.json
        config = AIConfig(**config_data)
        
        global ai_service
        ai_service = AIService()
        ai_service.initialize(**config.get_model_config())
        
        return jsonify({
            'status': 'success',
            'message': 'AI service initialized successfully'
        })
    except Exception as e:
        logger.error(f"Error initializing AI service: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ai/train', methods=['POST'])
def train():
    """Train the model"""
    try:
        if not ai_service:
            return jsonify({
                'status': 'error',
                'message': 'AI service not initialized'
            }), 400
        
        data = request.json
        texts = data.get('texts', [])
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 32)
        validation_split = data.get('validation_split', 0.1)
        
        history = ai_service.train(
            texts=texts,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        return jsonify({
            'status': 'success',
            'history': history
        })
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ai/generate', methods=['POST'])
def generate():
    """Generate response"""
    try:
        if not ai_service:
            return jsonify({
                'status': 'error',
                'message': 'AI service not initialized'
            }), 400
        
        data = request.json
        input_text = data.get('text', '')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.7)
        
        response = ai_service.generate_response(
            input_text=input_text,
            max_length=max_length,
            temperature=temperature
        )
        
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ai/evaluate', methods=['POST'])
def evaluate():
    """Evaluate model performance"""
    try:
        if not ai_service:
            return jsonify({
                'status': 'error',
                'message': 'AI service not initialized'
            }), 400
        
        data = request.json
        texts = data.get('texts', [])
        
        metrics = ai_service.evaluate(texts)
        
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ai/save', methods=['POST'])
def save():
    """Save model"""
    try:
        if not ai_service:
            return jsonify({
                'status': 'error',
                'message': 'AI service not initialized'
            }), 400
        
        data = request.json
        name = data.get('name', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        ai_service.save(name)
        
        return jsonify({
            'status': 'success',
            'message': f'Model saved as {name}'
        })
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ai/info', methods=['GET'])
def get_info():
    """Get model information"""
    try:
        if not ai_service:
            return jsonify({
                'status': 'error',
                'message': 'AI service not initialized'
            }), 400
        
        info = ai_service.get_model_info()
        
        return jsonify({
            'status': 'success',
            'info': info
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Load model if path provided
    model_path = os.getenv('AI_MODEL_PATH')
    if model_path:
        load_model(model_path)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000) 