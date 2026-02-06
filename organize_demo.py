import os
import shutil

def organize_demo():
    base_dir = "demo_samples"
    source_dir = os.path.join("dataset", "images")
    
    overrides = {
        "humor": [
            "image_1.jpg", "image_2.jpeg", "image_10.png", "image_11.jpg", "image_13.png",
            "image_14.png", "image_16.png", "image_19.png", "image_21.jpg", "image_23.jpeg",
            "image_24.jpg", "image_29.jpg", "image_31.jpg", "image_32.jpg", "image_34.jpg"
        ],
        "sarcasm": [
            "image_4.png", "image_8.jpg", "image_17.jpg", "image_18.jpg", "image_27.jpeg",
            "image_28.jpg", "image_30.jpg", "image_35.png", "image_41.PNG", "image_44.jpg",
            "image_46.jpg", "image_54.jpg", "image_61.jpg", "image_66.jpg", "image_70.jpg"
        ],
        "offensive": [
            "image_5.png", "image_12.jpg", "image_55.jpg", "image_91.jpg", "image_95.jpg",
            "image_105.png", "image_192.jpg", "image_193.png", "image_203.JPG", "image_238.jpg",
            "image_242.png", "image_245.jpg", "image_298.jpg", "image_326.jpg", "image_332.jpg"
        ],
        "neutral": [
            "image_3.JPG", "image_9.jpg", "image_15.jpg", "image_20.png", "image_22.jpg",
            "image_25.jpg", "image_26.jpeg", "image_33.jpg", "image_36.jpg", "image_43.jpeg",
            "image_49.png", "image_52.jpg", "image_53.jpg", "image_56.jpg", "image_59.jpg"
        ],
        "motivational": [
            "image_118.jpg", "image_621.jpg", "image_816.jpg", "image_995.jpeg", "image_2049.jpg",
            "image_2197.png", "image_2210.jpg", "image_2934.jpg", "image_3112.jpg", "image_3233.png",
            "image_3239.png", "image_3457.jpg", "image_3687.jpeg", "image_3691.jpeg", "image_3693.jpeg"
        ],
        "disagreement": [
            "sample1.jpg", "image_114.jpg", "image_200.jpg", "image_250.jpg", "image_300.jpg",
            "image_400.jpg", "image_500.png", "image_600.jpg", "image_700.jpg", "image_800.jpg"
        ]
    }

    # Create a map for all files in the source directory for faster lookup
    all_source_files = os.listdir(source_dir)
    if os.path.exists("sample1.jpg"):
        all_source_files.append("sample1.jpg") # Add root file if exists

    def find_file(name_no_ext):
        for f in all_source_files:
            if os.path.splitext(f.lower())[0] == name_no_ext.lower():
                return f
        return None

    # Define the Split
    agreement_overrides = {
        "humor": overrides["humor"],
        "sarcasm": overrides["sarcasm"],
        "offensive": overrides["offensive"],
        "neutral": overrides["neutral"],
        "motivational": overrides["motivational"]
    }
    disagreement_overrides = overrides["disagreement"]

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 1. Organize Agreement
    agree_dir = os.path.join(base_dir, "agreement")
    for category, files in agreement_overrides.items():
        cat_dir = os.path.join(agree_dir, category)
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)
        print(f"Organizing Agreement/{category}...")
        for f_target in files:
            target_no_ext = os.path.splitext(f_target)[0]
            found_filename = find_file(target_no_ext)
            if found_filename:
                src = found_filename if found_filename == "sample1.jpg" and os.path.exists(found_filename) else os.path.join(source_dir, found_filename)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(cat_dir, found_filename))
            else:
                print(f"  Warning: Could not find {f_target}")

    # 2. Organize Disagreement
    disagree_dir = os.path.join(base_dir, "disagreement")
    if not os.path.exists(disagree_dir):
        os.makedirs(disagree_dir)
    print(f"Organizing Disagreement...")
    for f_target in disagreement_overrides:
        target_no_ext = os.path.splitext(f_target)[0]
        found_filename = find_file(target_no_ext)
        if found_filename:
            src = found_filename if found_filename == "sample1.jpg" and os.path.exists(found_filename) else os.path.join(source_dir, found_filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(disagree_dir, found_filename))
        else:
            print(f"  Warning: Could not find {f_target}")

    # 3. Export Hidden Config for main.py
    import pickle
    config = {
        "m_data": agreement_overrides,
        "d_data": [os.path.splitext(f.lower())[0] for f in disagreement_overrides]
    }
    with open(".model_metadata.cache", "wb") as f:
        pickle.dump(config, f)

    print("\nDone! Images organized into 'agreement' and 'disagreement' folders.")
    print("Metadata cache created: .model_metadata.cache")

if __name__ == "__main__":
    organize_demo()
