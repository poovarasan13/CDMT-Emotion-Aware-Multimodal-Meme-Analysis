import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

def plot_emotion_probabilities(emotions, probs):
    """
    Returns a matplotlib figure of emotion probabilities.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=emotions, y=probs, palette="viridis", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence Score")
    ax.set_title("Predicted Emotion Probabilities")
    
    # Add values on top of bars
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p:.2f}", ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    return fig

def create_explanation_text(emotion, variance, extracted_text):
    """
    Generates a natural language explanation.
    """
    explanation = f"""
    **Analysis Summary:**
    - **Detected Emotion:** {emotion.upper()}
    - **Text Context:** "{extracted_text[:50]}..."
    
    The model combined visual cues with the text semantics. 
    """
    return explanation
