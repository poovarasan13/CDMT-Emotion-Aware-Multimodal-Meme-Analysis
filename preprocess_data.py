import pandas as pd
import os

def preprocess_dataset(input_csv, image_dir, output_csv, filter_existing_images=True):
    print(f"Reading {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # MAPPING LOGIC
    # The dataset has 'multimodal_annotation_humans' with values: 
    # 'general', 'twisted_meaning', 'very_twisted', 'not_sarcastic'
    # We map these to our target classes: 'humor', 'sarcasm', 'offensive', 'motivational', 'neutral'
    
    label_map = {
        'general': 'humor',          # Assumption: General memes are intended to be humorous
        'twisted_meaning': 'sarcasm', # Twisted often implies sarcasm
        'very_twisted': 'offensive',  # Higher degree of twisted might imply offensive/edgy
        'not_sarcastic': 'neutral',   # Explicitly not sarcastic -> neutral/straightforward
        'motivational': 'motivational' # If exists
    }
    
    # Apply mapping
    # Using 'multimodal_annotation_humans' as the ground truth label
    if 'multimodal_annotation_humans' not in df.columns:
        print(f"Column 'multimodal_annotation_humans' not found. Available: {df.columns}")
        return

    df['mapped_label'] = df['multimodal_annotation_humans'].map(label_map)
    
    # Fill unmapped with 'neutral' or drop
    df['mapped_label'] = df['mapped_label'].fillna('neutral')
    
    processed_data = []
    
    print("Validating images...")
    for idx, row in df.iterrows():
        img_name = row['image_name']
        img_path = os.path.join(image_dir, img_name)
        text = row['text_corrected']
        label = row['mapped_label']
        
        # Check text
        if pd.isna(text) or str(text).strip() == "":
            text = "[No Text]"
            
        # Check image existence
        if filter_existing_images:
            if not os.path.exists(img_path):
                # Try extensions if missing (sometimes .jpg vs .png mismatch)
                continue
                
        processed_data.append({
            'image_path': os.path.abspath(img_path),
            'text': str(text),
            'label': label
        })
        
    new_df = pd.DataFrame(processed_data)
    print(f"Processed {len(new_df)} valid samples out of {len(df)}")
    
    new_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_CSV = os.path.join(BASE_DIR, "dataset", "combined_data.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")
    OUTPUT_CSV = os.path.join(BASE_DIR, "dataset", "train_data.csv")
    
    if os.path.exists(INPUT_CSV):
        preprocess_dataset(INPUT_CSV, IMAGE_DIR, OUTPUT_CSV)
    else:
        print(f"Input CSV not found at {INPUT_CSV}")
