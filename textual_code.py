import pandas as pd
import numpy as np
import os
import re
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# --- Configuration ---
# Get the absolute path to the directory where the script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The 'dataset' directory is a sibling to the script's directory.
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

# Use a lightweight, pre-trained Transformer model.
MODEL_NAME = 'distilbert-base-uncased'
BATCH_SIZE = 32  # You can reduce to 16 if GPU memory is limited
MAX_LENGTH = 128  # Max token length for Transformer

# --- Utility Functions ---
def clean_catalog_text(text):
    """Cleans the catalog content for Transformer processing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"Value:.*", "", text)
    text = re.sub(r"Unit:.*", "", text)
    text = re.sub(r"(Item Name:|Bullet Point \d:|Product Description:)", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_features(df, text_column, model_name, batch_size=BATCH_SIZE):
    """
    Tokenizes text data and generates embeddings using a pre-trained Transformer model.
    Optimized for GPU memory efficiency.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set model to evaluation mode

    df['cleaned_text'] = df[text_column].apply(clean_catalog_text)
    text_features = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['cleaned_text'][i:i+batch_size].tolist()
        
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=MAX_LENGTH
        )
        for key in encoded:
            encoded[key] = encoded[key].to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            text_features.append(embeddings)

        # Free memory
        del encoded, outputs
        torch.cuda.empty_cache()

    return np.concatenate(text_features, axis=0)

# --- Main Script ---
def run_text_processor_pipeline():
    """
    Main function to load data, process text, and save features.
    """
    print("Starting text feature extraction pipeline...")
    
    # --- 1. Load Data ---
    try:
        # File is in the same directory as the script.
        df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure train.csv is in the correct directory: {os.path.abspath(DATASET_DIR)}.")
        return

    # --- 2. Extract Text Features ---
    print(f"Extracting text features using {MODEL_NAME}...")
    text_features = extract_text_features(df, 'catalog_content', MODEL_NAME)
    
    print(f"Text features extracted successfully! Shape: {text_features.shape}")
    
    # --- 3. Save Features ---
    # Save the features as a NumPy array for later use.
    np.save('text_features.npy', text_features)
    
    # Also save the structured features
    df['price_log'] = np.log1p(df['price'])
    df[['sample_id', 'catalog_content', 'price', 'price_log']].to_csv(
        'structured_data_with_text_features.csv', index=False
    )
    
    print("Saved extracted text features to 'text_features.npy' and structured data to 'structured_data_with_text_features.csv'.")

if __name__ == "__main__":
    run_text_processor_pipeline()