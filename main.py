import streamlit as st
import os
import sys

# DEBUG: Fix for potential Illegal Instruction (AVX) on some CPUs
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# Force torch to use CPU if needed, though script handles device checking
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 

print("DEBUG: Starting main.py", file=sys.stderr)

import torch
import numpy as np
from PIL import Image

print(f"DEBUG: Torch Version: {torch.__version__}", file=sys.stderr)
print(f"DEBUG: Numpy Version: {np.__version__}", file=sys.stderr)


# Add project directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_image, extract_text_from_image
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from fusion import MultimodalFusion
from emotion_classifier import EmotionClassifier
from dissonance import CognitiveDissonance
import explainability
import numpy as np

# Page Config
st.set_page_config(
    page_title="CDMT Meme Analysis",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🧩 CDMT: Emotion-Aware Multimodal Meme Analysis")
st.markdown("""
**Cognitive Dissonance Regularized Multimodal Transformer**
*A Demo Implementation for Meme Understanding & Human-AI Disagreement Detection*
""")

# Sidebar settings
st.sidebar.header("Settings")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.text(f"Running on: {device.upper()}")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    st.sidebar.text("Loading Image Encoder...")
    img_enc = ImageEncoder()
    
    st.sidebar.text("Loading Text Encoder...")
    txt_enc = TextEncoder()
    
    st.sidebar.text("Loading Fusion Module...")
    # Dimensions match the specific models: CLIP-ViT-B/32 (768), MiniLM (384)
    fusion_mod = MultimodalFusion(image_dim=768, text_dim=384, hidden_dim=512)
    
    st.sidebar.text("Loading Classifier...")
    classifier = EmotionClassifier(input_dim=512)
    
    st.sidebar.text("Initializing CD Metrics...")
    cd_metric = CognitiveDissonance(embedding_dim=512)
    
    # LOAD TRAINED WEIGHTS IF AVAILABLE
    weights_path = "cdmt_trained.pth"
    if not os.path.exists(weights_path):
        # Check checkpoints folder
        weights_path = os.path.join("checkpoints", "cdmt_trained.pth")
    
    if os.path.exists(weights_path):
        st.sidebar.success(f"Loading trained weights from {weights_path}")
        try:
            # Load with force_reload=True if needed, but here simple load
            # Check for generic torch load issues (weights_only=True default in newer torch)
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu')) 
            
            # Support both shard/dict formats
            if 'fusion' in checkpoint:
                fusion_mod.load_state_dict(checkpoint['fusion'])
                classifier.load_state_dict(checkpoint['classifier'])
            else:
                fusion_mod.load_state_dict(checkpoint['fusion_state_dict'])
                classifier.load_state_dict(checkpoint['classifier_state_dict'])
                
            st.sidebar.text("Weights applied successfully.")
        except Exception as e:
            st.sidebar.error(f"Failed to load weights: {e}")
    else:
        st.sidebar.warning("No trained weights found. Using random init.")
            
    return img_enc, txt_enc, fusion_mod, classifier, cd_metric

with st.spinner("Initializing models... (this may take a minute)"):
    img_encoder, txt_encoder, fusion_module, emotion_clf, cd_system = load_models()
    ST_MODELS_LOADED = True

st.sidebar.success("All models loaded successfully!")

# --- Main Interface ---

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Upload Meme")
    uploaded_file = st.file_uploader("Choose a meme image...", type=["jpg", "png", "jpeg"], help="Select a sample from the dataset images folder for best results.")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Meme', width=400) # Fixed width for better layout
        
        # --- Processing ---
        if st.button("Analyze Meme", type="primary"):
            with st.spinner("Processing..."):
                # 1. OCR
                image_np = np.array(image)
                extracted_text = extract_text_from_image(image_np)
                
                # 2. Encoding
                # Image
                img_emb = img_encoder(image) # (1, 768)
                
                # Text
                txt_emb = txt_encoder([extracted_text]) # (1, 384)
                
                # 3. Fusion
                multimodal_vector = fusion_module(img_emb, txt_emb) # (1, 512)
                
                # 4. Prediction
                pred_emotion, pred_probs = emotion_clf.predict(multimodal_vector)

                # 5. CD Metric
                cds_score = cd_system.compute_cds(multimodal_vector, pred_emotion)
                
                is_disagreement, disagreement_reason = cd_system.detect_disagreement(cds_score, threshold=0.8)
                
                st.session_state['results'] = {
                    'text': extracted_text,
                    'emotion': pred_emotion,
                    'probs': pred_probs,
                    'cds': cds_score,
                    'is_disagreement': is_disagreement,
                    'reason': disagreement_reason
                }

with col2:
    if 'results' in st.session_state and uploaded_file:
        res = st.session_state['results']
        
        st.subheader("2. Analysis Results")
        
        # Extracted Text
        st.info(f"**Extracted Text:** {res['text']}")
        
        # Top Result
        st.markdown(f"### Predicted Emotion: **{res['emotion'].upper()}**")
        
        # Probability Chart
        st.pyplot(explainability.plot_emotion_probabilities(
            emotion_clf.emotions, res['probs']
        ))
        
        st.subheader("3. Cognitive Dissonance & Reliability")
        
        # CD Score Display
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Cognitive Dissonance Score (CDS)", f"{res['cds']:.4f}")
        
        if res['is_disagreement']:
            metric_col2.error("DISAGREEMENT DETECTED")
        else:
            metric_col2.success("Consensus Likely")
            
        st.markdown(f"**Interpretation:** *{res['reason']}*")
        
        st.markdown("---")
        st.caption("Demo implementation of Emotion-Aware Multimodal Transformer Phase I")

    elif not uploaded_file:
        st.info("Please upload an image to start.")
