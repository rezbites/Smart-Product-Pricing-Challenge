import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# --- Configuration ---
# Get the absolute path to the directory where the script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Build absolute paths to the data and features
DATA_DIR = os.path.join(SCRIPT_DIR, 'dataset')
IMAGE_FEATURES_PATH = os.path.join(SCRIPT_DIR, 'cnn_image_features.npy')
TEXT_FEATURES_PATH = os.path.join(SCRIPT_DIR, 'text_features.npy')
STRUCTURED_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')

# --- Utility Functions ---
def smape(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

# --- Main Script ---
def run_ensemble_pipeline():
    """
    Main function to load features, train a stacked model, and evaluate performance.
    This version uses only image and text features.
    """
    print("Starting Two-Model Stacking Ensemble Pipeline...")

    # --- 1. Load Data and Features with Error Handling ---
    try:
        df_train = pd.read_csv(STRUCTURED_DATA_PATH)
        image_features = np.load(IMAGE_FEATURES_PATH)
        text_features = np.load(TEXT_FEATURES_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all data and feature files are in their correct locations.")
        return

    # --- 2. Feature Combination ---
    # Combine image and text features into a single array
    if not (len(image_features) == len(text_features)):
        print("Error: The number of samples in the feature arrays do not match. Please check your feature extraction scripts.")
        return

    combined_features = np.concatenate(
        [image_features, text_features], axis=1
    )
    
    # Split the data
    X = pd.DataFrame(combined_features)
    y = np.log1p(df_train['price'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Train the Ensemble Meta-Model (LightGBM) ---
    # The pipeline is now simpler as there is no structured data to preprocess
    lgbm_model = lgb.LGBMRegressor(random_state=42)
    
    print("\nTraining the final LightGBM ensemble model...")
    lgbm_model.fit(X_train, y_train)
    print("Training complete.")

    # --- 4. Evaluate the Ensemble Model ---
    y_pred_log = lgbm_model.predict(X_val)
    y_pred_actual = np.expm1(y_pred_log)
    y_val_actual = np.expm1(y_val)
    final_smape = smape(y_val_actual, y_pred_actual)

    print(f"\nFinal Ensemble Validation SMAPE Score: {final_smape:.4f}%")

if __name__ == "__main__":
    run_ensemble_pipeline()