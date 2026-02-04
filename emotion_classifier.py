import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=5):
        super().__init__()
        self.emotions = ["humor", "sarcasm", "offensive", "motivational", "neutral"]
        print(f"Initializing Emotion Classifier for {self.emotions}")
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, fused_embedding):
        """
        Args:
            fused_embedding: (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.head(fused_embedding)
    
    def predict(self, fused_embedding):
        """
        Returns label and probabilities.
        """
        logits = self.forward(fused_embedding)
        probs = torch.softmax(logits, dim=1)
        
        # Get max prob class
        max_prob, idx = torch.max(probs, dim=1)
        
        return self.emotions[idx.item()], probs.detach().numpy()[0]
