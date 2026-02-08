import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import argparse
import os
from tqdm import tqdm
import logging
import glob

from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from fusion import MultimodalFusion
from emotion_classifier import EmotionClassifier
from dissonance import CognitiveDissonance

# Force safe torch load if needed (though we train from scratch here)
# torch.serialization.add_safe_globals(...)

# Configure Logging
def setup_logging(log_file="training1.log", resume=False):
    # If resuming, append to log; otherwise overwrite
    file_mode = 'a' if resume else 'w'
    
    # Remove existing handlers to avoid duplication if function called multiple times
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode=file_mode),
            logging.StreamHandler()
        ]
    )

class MemeDataset(Dataset):
    def __init__(self, csv_file, transform=None, max_samples=None):
        self.data = pd.read_csv(csv_file)
        if max_samples:
            self.data = self.data.head(max_samples)
            logging.info(f"Limiting dataset to {max_samples} samples.")
            
        self.label_map = {
            "humor": 0, "sarcasm": 1, "offensive": 2, "motivational": 3, "neutral": 4
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        text = row['text']
        label_str = row['label']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black') # Fallback
            
        label = self.label_map.get(label_str, 4) # Default to neutral
        
        return image, text, torch.tensor(label)

class ContrastiveLoss(nn.Module):
    """
    Cognitive Dissonance Loss / Contrastive Loss
    Encourages the model's multimodal embedding to align with the 'Human Emotion Embedding' 
    prediction for the ground truth label.
    """
    def __init__(self, human_embeddings):
        super().__init__()
        self.human_embeddings = human_embeddings
        self.mse = nn.MSELoss()
        
    def forward(self, multimodal_embedding, labels):
        # multimodal_embedding: (B, D)
        # labels: (B) -> indices
        
        # Get target human embeddings for the batch
        target_embs = []
        emotion_keys = ["humor", "sarcasm", "offensive", "motivational", "neutral"]
        for label_idx in labels:
            key = emotion_keys[label_idx.item()]
            target_embs.append(self.human_embeddings[key])
            
        target_embs = torch.stack(target_embs).to(multimodal_embedding.device)
        
        # Normalize both
        pred_norm = torch.nn.functional.normalize(multimodal_embedding, p=2, dim=1)
        target_norm = torch.nn.functional.normalize(target_embs, p=2, dim=1)
        
        return self.mse(pred_norm, target_norm)

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train(args):
    setup_logging(args.log_file, resume=bool(args.resume))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}...")
    
    # 1. Load Data
    full_csv = os.path.join("archive", "memotion_dataset_7k", "processed_train_data.csv")
    if not os.path.exists(full_csv):
        logging.info("Preprocessed csv not found! Running preprocessing...")
        import preprocess_data
        preprocess_data.preprocess_dataset(
            os.path.join("archive", "memotion_dataset_7k", "labels.csv"), 
            os.path.join("archive", "memotion_dataset_7k", "images"), 
            full_csv
        )
        
    dataset = MemeDataset(full_csv, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 2. Initialize Models
    img_enc = ImageEncoder().to(device)
    txt_enc = TextEncoder().to(device)
    fusion = MultimodalFusion().to(device)
    classifier = EmotionClassifier().to(device)
    dissonance_mod = CognitiveDissonance(embedding_dim=512)
    
    # Move fixed human embeddings to device
    human_embs = {k: v.to(device) for k, v in dissonance_mod.human_emotion_embeddings.items()}
    
    # Optimizers (only optimizing Fusion and Classifier heads, keeping Encoders frozen)
    optimizer = torch.optim.Adam([
        {'params': fusion.parameters()},
        {'params': classifier.parameters()}
    ], lr=args.lr)
    
    # Losses
    criterion_cls = nn.CrossEntropyLoss()
    criterion_cd = ContrastiveLoss(human_embs)

    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            fusion.load_state_dict(checkpoint['fusion_state_dict'])
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at '{args.resume}'")
    
    logging.info("Starting Training Loop...")
    
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        img_enc.eval() # Always eval (frozen)
        txt_enc.eval()
        fusion.train()
        classifier.train()
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, texts, labels in progress:
            labels = labels.to(device)
            
            # Zero grad
            optimizer.zero_grad()
            
            # Forward
            with torch.no_grad(): # Encoders frozen
                 img_feats = img_enc(images).to(device)
                 txt_feats = txt_enc(texts).to(device)
            
            fused = fusion(img_feats, txt_feats)
            logits = classifier(fused)
            
            # Calculate Losses
            loss_class = criterion_cls(logits, labels)
            loss_cd = criterion_cd(fused, labels)
            
            # Weighted sum (lambda can be tuned)
            loss = loss_class + 0.1 * loss_cd
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({"Loss": loss.item()})
            
            # Store predictions for metrics
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(dataloader)
        
        # Calculate Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        logging.info(f"Epoch {epoch+1} Summary:")
        logging.info(f"  Loss: {avg_loss:.4f}")
        logging.info(f"  Accuracy: {acc:.4f}")
        logging.info(f"  F1 Score (Weighted): {f1:.4f}")
        logging.info("-" * 30)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'fusion_state_dict': fusion.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        # Save latest
        latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)
        # Save per epoch (optional, maybe every 5 epochs or just keep latest to save space?)
        # For now, let's just keep the latest and maybe a specific one if needed.
        # But user asked for stop/start, so usually latest is enough. 
        # I'll also save one with epoch number just in case needed.
        epoch_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, epoch_path)
        logging.info(f"Saved checkpoint to {latest_path} and {epoch_path}")
        
    # Final Summary
    logging.info("=" * 30)
    logging.info("Training Completed!")
    if 'acc' in locals():
        logging.info(f"Final Model Metrics (Epoch {epoch+1}):")
        logging.info(f"  Loss: {avg_loss:.4f}")
        logging.info(f"  Accuracy: {acc:.4f}")
        logging.info(f"  F1 Score: {f1:.4f}")
    else:
        logging.info("No training steps performed (already finished?).")
    logging.info("=" * 30)

    # Final Model Save
    logging.info("Saving final model...")
    save_path = os.path.join(args.checkpoint_dir, "cdmt_trained.pth")
    torch.save({
        'fusion': fusion.state_dict(),
        'classifier': classifier.state_dict()
    }, save_path)
    logging.info(f"Model saved to {save_path}")

def collate_fn(batch):
    # Custom collate to handle PIL images and list of texts
    images, texts, labels = zip(*batch)
    return list(images), list(texts), torch.stack(list(labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--log_file", type=str, default="training.log", help="Path to log file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    train(args)
