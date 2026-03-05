import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPConfig

class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        print(f"Loading Image Encoder: {model_name}...")
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        try:
            self.model = CLIPVisionModel.from_pretrained(model_name)
        except AttributeError:
            # Fix for AttributeError: 'CLIPConfig' object has no attribute 'hidden_size'
            print("Loading model using explicit vision config...")
            config = CLIPConfig.from_pretrained(model_name)
            self.model = CLIPVisionModel.from_pretrained(model_name, config=config.vision_config)
        
        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, image):
        """
        Args:
            image: PIL Image or list of PIL Images
        Returns:
            embedding: (batch_size, hidden_dim)
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        # Use pooler_output for global representation (batch_size, 768)
        return outputs.pooler_output
