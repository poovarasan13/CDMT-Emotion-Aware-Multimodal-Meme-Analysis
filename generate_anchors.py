import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from fusion import MultimodalFusion
from dissonance import CognitiveDissonance

def generate_anchors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load trained models
    img_enc = ImageEncoder().to(device)
    txt_enc = TextEncoder().to(device)
    fusion = MultimodalFusion().to(device)
    
    weights_path = os.path.join("checkpoints", "cdmt_trained.pth")
    if not os.path.exists(weights_path):
        weights_path = "cdmt_trained.pth" 
        if not os.path.exists(weights_path):
            print("No trained weights found!")
            return

    print(f"Loading weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    fusion.load_state_dict(checkpoint['fusion'])
    
    img_enc.eval()
    txt_enc.eval()
    fusion.eval()

    # 2. Load Dataset
    csv_path = os.path.join("archive", "memotion_dataset_7k", "processed_train_data.csv")
    df = pd.read_csv(csv_path)
    
    # We only need a subset of each class to get a decent anchor
    emotions = ["humor", "sarcasm", "offensive", "motivational", "neutral"]
    centroids = {}

    for emotion in emotions:
        print(f"Generating anchor for: {emotion}")
        subset = df[df['label'] == emotion].head(100) # Use 100 samples per class
        if len(subset) == 0:
            print(f"Warning: No samples found for {emotion}, using random.")
            centroids[emotion] = torch.randn(512)
            continue
            
        embeddings = []
        for idx, row in tqdm(subset.iterrows(), total=len(subset)):
            try:
                img = Image.open(row['image_path']).convert('RGB')
                txt = [row['text']]
                
                with torch.no_grad():
                    iframes = img_enc(img).to(device)
                    tframes = txt_enc(txt).to(device)
                    fused = fusion(iframes, tframes)
                    # Normalize before averaging
                    norm_fused = torch.nn.functional.normalize(fused, p=2, dim=1)
                    embeddings.append(norm_fused.cpu())
            except Exception as e:
                continue
        
        if embeddings:
            mean_emb = torch.stack(embeddings).mean(dim=0).squeeze(0)
            # Re-normalize the mean
            centroids[emotion] = torch.nn.functional.normalize(mean_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
        else:
            centroids[emotion] = torch.randn(512)

    # 3. Save Anchors
    save_path = "emotion_anchors.pth"
    torch.save(centroids, save_path)
    print(f"Anchors saved to {save_path}")
    print("Now update dissonance.py to load these anchors.")

if __name__ == "__main__":
    generate_anchors()
