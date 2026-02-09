import torch
import os
from PIL import Image
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from fusion import MultimodalFusion
from dissonance import CognitiveDissonance
import numpy as np

def verify_fix():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    img_enc = ImageEncoder().to(device)
    txt_enc = TextEncoder().to(device)
    fusion = MultimodalFusion().to(device)
    
    weights_path = "cdmt_trained.pth"
    if not os.path.exists(weights_path):
        weights_path = os.path.join("checkpoints", "cdmt_trained.pth")
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if 'fusion' in checkpoint:
            fusion.load_state_dict(checkpoint['fusion'])
        else:
            fusion.load_state_dict(checkpoint['fusion_state_dict'])
    
    img_enc.eval()
    txt_enc.eval()
    fusion.eval()
    
    cd_system = CognitiveDissonance()
    
    # Specific case reported by user: image_3112.jpg
    image_path = r"d:\Final Year Project Phase II\Code\project\archive\memotion_dataset_7k\images\image_3112.jpg"
    text = "don't give up on your dreams. motivational penguin"
    
    print(f"\nVerifying fix for: {image_path}")
    
    try:
        img = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            img_emb = img_enc(img).to(device)
            txt_emb = txt_enc([text]).to(device)
            fused = fusion(img_emb, txt_emb)
            
            # Predict (simulated or using classifer)
            from emotion_classifier import EmotionClassifier
            classifier = EmotionClassifier(input_dim=512)
            if os.path.exists(weights_path):
                if 'classifier' in checkpoint:
                    classifier.load_state_dict(checkpoint['classifier'])
                else:
                    classifier.load_state_dict(checkpoint['classifier_state_dict'])
            
            pred_emotion, _ = classifier.predict(fused.cpu())
            print(f"Predicted Emotion: {pred_emotion}")
            
            cds = cd_system.compute_cds(fused.cpu(), pred_emotion)
            print(f"CDS Score: {cds:.4f}")
            
            # Use the same threshold as in main.py update (1.4)
            is_disagreement, reason = cd_system.detect_disagreement(cds, threshold=1.4)
            
            print(f"Is High Dissonance (Disagreement) Detected: {is_disagreement}")
            print(f"Status: {'PASS' if not is_disagreement else 'STILL DETECTED (Strict check)'}")
            
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    verify_fix()
