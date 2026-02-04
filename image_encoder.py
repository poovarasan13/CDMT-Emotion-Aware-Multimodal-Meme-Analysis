import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        print(f"Loading Image Encoder: {model_name}...")
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        
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
