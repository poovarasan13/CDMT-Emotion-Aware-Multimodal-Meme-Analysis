import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, image_dim=768, text_dim=384, hidden_dim=512):
        super().__init__()
        print(f"Initializing Fusion Module (Image: {image_dim}, Text: {text_dim} -> Hidden: {hidden_dim})")
        
        # Project both to same dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Fusion mechanism: Concatenation + MLP
        # Input to MLP will be hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, image_emb, text_emb):
        """
        Args:
            image_emb: (batch_size, image_dim)
            text_emb: (batch_size, text_dim)
        Returns:
            fused_emb: (batch_size, hidden_dim)
        """
        img_p = self.image_proj(image_emb)
        txt_p = self.text_proj(text_emb)
        
        # Concatenate
        combined = torch.cat((img_p, txt_p), dim=1)
        
        # Fuse
        fused = self.classifier(combined)
        return fused
