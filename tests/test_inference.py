import pytest
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add the parent directory to path so we can import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.models.inference import ModelInference, EfficientCNN
from backend.api.core.preprocessor import ImagePreprocessor

@pytest.fixture
def model_inference():
    inference = ModelInference()
    inference.load_models()
    return inference

@pytest.fixture
def preprocessor():
    return ImagePreprocessor()

@pytest.fixture
def sample_image():
    return Image.new('RGB', (224, 224), color='red')

def test_model_loading(model_inference):
    """Test that models load correctly"""
    assert model_inference.models_loaded
    assert 'original' in model_inference.models
    assert 'quantized' in model_inference.models

def test_preprocessing(preprocessor, sample_image):
    """Test image preprocessing"""
    tensor = preprocessor.preprocess(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)

def test_inference(model_inference, preprocessor, sample_image):
    """Test inference on sample image"""
    tensor = preprocessor.preprocess(sample_image)
    predictions = model_inference.predict(tensor, 'quantized')
    
    assert isinstance(predictions, list)
    assert len(predictions) == 5
    assert all('class' in p for p in predictions)
    assert all('probability' in p for p in predictions)
    assert all(0 <= p['probability'] <= 1 for p in predictions)

def test_batch_inference(model_inference, preprocessor):
    """Test batch inference"""
    images = [Image.new('RGB', (224, 224), color='red') for _ in range(3)]
    tensors = preprocessor.preprocess_batch(images)
    
    predictions = model_inference.predict_batch(tensors, 'quantized')
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    assert all(len(p) == 5 for p in predictions)

def test_model_info(model_inference):
    """Test model info endpoint"""
    info = model_inference.get_model_info()
    assert 'device' in info
    assert 'models_loaded' in info
    assert 'classes' in info
    assert 'num_classes' in info
    assert info['num_classes'] == 10

if __name__ == "__main__":
    pytest.main([__file__])