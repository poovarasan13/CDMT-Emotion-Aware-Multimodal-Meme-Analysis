import pandas as pd
import os

def preprocess_dataset(input_csv, image_dir, output_csv):
    print(f"Reading new dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    processed_data = []

    def map_priority(row):
        # 1. Motivational Priority
        if str(row['motivational']).strip().lower() == 'motivational':
            return 'motivational'
        
        # 2. Offensive Priority (Very or Hateful)
        if str(row['offensive']).strip().lower() in ['very_offensive', 'hateful_offensive']:
            return 'offensive'
            
        # 3. Sarcasm Priority (Twisted or Very Twisted)
        if str(row['sarcasm']).strip().lower() in ['twisted_meaning', 'very_twisted']:
            return 'sarcasm'
            
        # 4. Humor Priority (Anything better than not_funny)
        if str(row['humour']).strip().lower() in ['funny', 'very_funny', 'hilarious']:
            return 'humor'
            
        # 5. Default
        return 'neutral'

    print("Mapping labels and validating images...")
    for idx, row in df.iterrows():
        img_name = row['image_name']
        img_path = os.path.join(image_dir, img_name)
        text = row['text_corrected']
        label = map_priority(row)
        
        if pd.isna(text) or str(text).strip() == "":
            text = "[No Text]"
            
        if not os.path.exists(img_path):
            continue
                
        processed_data.append({
            'image_path': os.path.abspath(img_path),
            'text': str(text),
            'label': label
        })
        
    new_df = pd.DataFrame(processed_data)
    print(f"Processed {len(new_df)} samples.")
    print("\nLabel Distribution:")
    print(new_df['label'].value_counts())
    
    new_df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_CSV = os.path.join(BASE_DIR, "archive", "memotion_dataset_7k", "labels.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "archive", "memotion_dataset_7k", "images")
    OUTPUT_CSV = os.path.join(BASE_DIR, "archive", "memotion_dataset_7k", "processed_train_data.csv")
    
    if os.path.exists(INPUT_CSV):
        preprocess_dataset(INPUT_CSV, IMAGE_DIR, OUTPUT_CSV)
    else:
        print(f"Input CSV not found at {INPUT_CSV}")
