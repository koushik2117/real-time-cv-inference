
import torch
from torchvision import transforms
from PIL import Image
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for inference"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tensor = self.transform(image)
            return tensor
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """Preprocess a batch of images"""
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors)
    
    def decode_base64(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return image
        except Exception as e:
            logger.error(f"Base64 decoding error: {str(e)}")
            raise