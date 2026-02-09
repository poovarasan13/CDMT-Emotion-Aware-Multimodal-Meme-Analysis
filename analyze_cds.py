import torch
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from fusion import MultimodalFusion
from dissonance import CognitiveDissonance
import numpy as np

def analyze_cds():
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
        try:
            if 'fusion' in checkpoint:
                fusion.load_state_dict(checkpoint['fusion'])
            else:
                fusion.load_state_dict(checkpoint['fusion_state_dict'])
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    img_enc.eval()
    txt_enc.eval()
    fusion.eval()
    
    cd_system = CognitiveDissonance()
    
    csv_path = os.path.join("archive", "memotion_dataset_7k", "processed_train_data.csv")
    df = pd.read_csv(csv_path)
    
    # Scan 500 images for high scores
    sample_df = df.head(500)
    results_list = []
    
    # Optimize: init classifier once
    from emotion_classifier import EmotionClassifier
    classifier = EmotionClassifier(input_dim=512)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        try:
            if 'classifier' in checkpoint:
                classifier.load_state_dict(checkpoint['classifier'])
            else:
                classifier.load_state_dict(checkpoint['classifier_state_dict'])
        except:
            pass

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        try:
            img = Image.open(row['image_path']).convert('RGB')
            txt = [row['text']]
            label = row['label']
            
            with torch.no_grad():
                img_emb = img_enc(img).to(device)
                txt_emb = txt_enc(txt).to(device)
                fused = fusion(img_emb, txt_emb)
                
                cds = cd_system.compute_cds(fused.cpu(), label)
                if cds > 1.36:
                    results_list.append({
                        'path': row['image_path'],
                        'text': row['text'],
                        'label': label,
                        'cds': cds
                    })
        except Exception as e:
            continue
            
    if results_list:
        # Sort by CDS descending
        results_list.sort(key=lambda x: x['cds'], reverse=True)
        
        print("\nDiscovered High Dissonance Samples (> 1.36):")
        for r in results_list:
            print(f"- Path: {r['path']}")
            print(f"  CDS: {r['cds']:.4f}")
            print(f"  Emotion: {r['label']}")

if __name__ == "__main__":
    analyze_cds()
