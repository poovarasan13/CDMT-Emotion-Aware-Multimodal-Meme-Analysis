import os
from PIL import Image
import torch
import easyocr
import numpy as np

# Initialize EasyOCR reader once to save time (lazy load can be better but this is simple)
# If easyocr fails/too heavy, we fallback to dummy
try:
    reader = easyocr.Reader(['en'], gpu=False) # Force CPU for demo stability
except Exception as e:
    print(f"Warning: EasyOCR not available. Using mock text. Error: {e}")
    reader = None

def load_image(image_path):
    """
    Loads an image from path and converts to RGB.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def extract_text_from_image(image_array):
    """
    Extracts text from an image numpy array using EasyOCR.
    """
    if reader is None:
        return "Mock text: This is a meme about coding."
    
    try:
        result = reader.readtext(image_array, detail=0)
        text = " ".join(result)
        return text if text.strip() else "[No text detected]"
    except Exception as e:
        print(f"OCR Error: {e}")
        return "[Error extracting text]"

def load_meme_and_text(image_path):
    """
    Loads image and extracts text.
    Returns: (PIL.Image, str)
    """
    image = load_image(image_path)
    if image is None:
        return None, None
    
    # Convert PIL to numpy for EasyOCR
    image_np = np.array(image)
    
    text = extract_text_from_image(image_np)
    
    return image, text
