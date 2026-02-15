import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class EfficientCNN(nn.Module):
    """CNN model architecture"""
    def __init__(self, num_classes=10):
        super(EfficientCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelInference:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        self.models_loaded = False
        
    def load_models(self):
        """Load both original and quantized models"""
        try:
            # Create model directories if they don't exist
            original_dir = Path(__file__).parent / 'original'
            quantized_dir = Path(__file__).parent / 'quantized'
            original_dir.mkdir(exist_ok=True)
            quantized_dir.mkdir(exist_ok=True)
            
            # For now, create dummy models if they don't exist
            # In practice, you'd train and save real models
            if not (original_dir / 'model_original.pth').exists():
                logger.warning("No pre-trained model found. Using random weights.")
                model = EfficientCNN(num_classes=10)
                torch.save(model.state_dict(), original_dir / 'model_original.pth')
            
            if not (quantized_dir / 'model_quantized.pth').exists():
                model = EfficientCNN(num_classes=10)
                torch.save(model.state_dict(), quantized_dir / 'model_quantized.pth')
            
            # Load original model
            original_path = original_dir / 'model_original.pth'
            model = EfficientCNN(num_classes=10)
            model.load_state_dict(torch.load(original_path, map_location='cpu'))
            model.to(self.device)
            model.eval()
            self.models['original'] = model
            logger.info("Original model loaded successfully")
            
            # Load quantized model
            quantized_path = quantized_dir / 'model_quantized.pth'
            model = EfficientCNN(num_classes=10)
            model.load_state_dict(torch.load(quantized_path, map_location='cpu'))
            model.to(self.device)
            model.eval()
            self.models['quantized'] = model
            logger.info("Quantized model loaded successfully")
            
            self.models_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict(self, input_tensor: torch.Tensor, model_type: str = 'quantized') -> List[Dict]:
        """Run inference on input tensor"""
        if model_type not in self.models:
            model_type = 'quantized' if 'quantized' in self.models else 'original'
        
        model = self.models[model_type]
        
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, 5)
        
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            predictions.append({
                'class': self.class_names[idx],
                'class_id': int(idx),
                'probability': float(prob)
            })
        
        return predictions
    
    def predict_batch(self, input_tensors: torch.Tensor, model_type: str = 'quantized') -> List[List[Dict]]:
        """Run inference on batch of images"""
        if model_type not in self.models:
            model_type = 'quantized' if 'quantized' in self.models else 'original'
        
        model = self.models[model_type]
        input_tensors = input_tensors.to(self.device)
        
        with torch.no_grad():
            outputs = model(input_tensors)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        batch_predictions = []
        for probs in probabilities:
            top_probs, top_indices = torch.topk(probs, 5)
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                predictions.append({
                    'class': self.class_names[idx],
                    'class_id': int(idx),
                    'probability': float(prob)
                })
            batch_predictions.append(predictions)
        
        return batch_predictions
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'device': str(self.device),
            'models_loaded': list(self.models.keys()),
            'classes': self.class_names,
            'num_classes': len(self.class_names)
        }
        return info