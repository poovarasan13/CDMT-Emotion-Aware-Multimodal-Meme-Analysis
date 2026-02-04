import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        print(f"Loading Text Encoder: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, text):
        """
        Args:
            text: str or list of str
        Returns:
            embedding: (batch_size, hidden_dim)
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = self.model(**inputs)
        
        # Mean pooling for sentence representation
        # (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
