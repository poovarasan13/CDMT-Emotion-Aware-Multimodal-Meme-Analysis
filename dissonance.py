import torch
import numpy as np
import os

class CognitiveDissonance:
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        
        # Step 1: Define Human Emotion Embeddings (FIXED anchors)
        # In a real system, these would be learned centroids from human-annotated data.
        # Step 1: Define Human Emotion Embeddings (FIXED anchors)
        weights_path = os.path.join(os.path.dirname(__file__), "emotion_anchors.pth")
        
        if os.path.exists(weights_path):
            self.human_emotion_embeddings = torch.load(weights_path)
            print("Loaded data-driven emotion anchors.")
        else:
            # Fallback to reproducible random anchors if no data-anchors exist
            torch.manual_seed(42)  
            self.human_emotion_embeddings = {
                "humor": torch.randn(embedding_dim),
                "sarcasm": torch.randn(embedding_dim),
                "offensive": torch.randn(embedding_dim),
                "motivational": torch.randn(embedding_dim),
                "neutral": torch.randn(embedding_dim)
            }
            # Normalize them
            for k in self.human_emotion_embeddings:
                self.human_emotion_embeddings[k] = torch.nn.functional.normalize(
                    self.human_emotion_embeddings[k].unsqueeze(0), p=2, dim=1
                ).squeeze(0)
            print("Using default random anchors.")
            
    def compute_cds(self, multimodal_embedding, predicted_emotion):
        """
        Computes Cognitive Dissonance Score.
        CDS = Euclidean Distance (AI_Embedding, Human_Embedding[Predicted_Emotion])
        """
        if predicted_emotion not in self.human_emotion_embeddings:
            return 0.0
            
        # Ensure input is normalized
        ai_emb = torch.nn.functional.normalize(multimodal_embedding, p=2, dim=1).squeeze(0)
        human_emb = self.human_emotion_embeddings[predicted_emotion]
        
        # Euclidean distance
        dist = torch.norm(ai_emb - human_emb, p=2).item()
        return dist
        
    def detect_disagreement(self, cds, threshold=1.38):
        """
        If CDS > Threshold, it implies the AI's internal representation 
        is far from the 'canonical' human representation of that emotion,
        suggesting potential disagreement or ambiguity.
        """
        is_disagreement = cds > threshold
        
        if is_disagreement:
            reason = "High dissonance detected. The AI's feature representation diverges significantly from the expected human prototype for this emotion. This often happens with sarcasm or cultural nuances."
        else:
            reason = "Low dissonance. AI representation aligns with human expectation."
            
        return is_disagreement, reason
