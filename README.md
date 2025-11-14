Smart Product Pricing Challenge (Multimodal Ensemble)

This repository contains the solution for the Smart Product Pricing Challenge, which focuses on predicting optimal product prices using a complex dataset comprising structured metadata, unstructured text descriptions, and visual product images.

Our approach leverages a Multimodal Stacking Ensemble combining the predictive power of Deep Learning (CNN, Transformer) with the speed and efficiency of Gradient Boosting (LightGBM).

Technical Achievements

Multimodal Fusion: Implemented a stacking ensemble architecture that combines three distinct feature modalities: structured product data, visual features, and semantic text features.

Visual Feature Extraction (CNN): Used Transfer Learning with TensorFlow/Keras and the EfficientNetB0 architecture to extract robust image features. Optimized training against GPU Out-of-Memory (OOM) errors by setting a low batch size and dynamically resizing inputs.

Textual Feature Extraction (Transformer): Used a PyTorch backend to implement DistilBERT for semantic feature extraction from product descriptions, capturing contextual price signals.

Final Meta-Learner: Trained a LightGBM Regressor on the concatenated feature matrix to determine the final price prediction, significantly improving upon single-model accuracy.

Evaluation: The entire pipeline is optimized for the Symmetric Mean Absolute Percentage Error (SMAPE) metric.

Project Structure

Your current working directory should look like this:

NEW FOLDER/
├── dataset/
│   ├── train.csv         (Raw training data)   
├── images/               (Contains all product image files)
├── src/
│   ├── ensemble_model.py (FINAL MODEL: Stacking & Prediction)
│   ├── sample_code.py    (CNN Model Training & Image Feature Extraction)
│   └── textual_code.py   (Transformer Text Feature Extraction)
├── best_cnn_model.h5     (Trained CNN Model Weights)
├── cnn_image_features.npy  (Output: CNN Image Embeddings)
├── text_features.npy     (Output: DistilBERT Text Embeddings)
└── requirements.txt


Setup and Installation

Due to known dependency conflicts between TensorFlow 2.5.0 and PyTorch 1.7.1 for older CUDA/NumPy versions, a specific set of packages must be used.

Create and Activate Virtual Environment (Python 3.9 recommended):

python -m venv venv
.\venv\Scripts\activate


Install Required Dependencies:

pip install -r requirements.txt


Running the Pipeline

The pipeline is designed to be run in three sequential steps. Each step saves its output as a file, which the next step consumes.

Step 1: Extract Text Features (DistilBERT)

This script loads train.csv and uses the Transformer model to generate text embeddings.

# This will generate text_features.npy and structured_data_with_text_features.csv
python src/textual_code.py


Step 2: Extract Image Features (EfficientNetB0)

This script trains the CNN (if best_cnn_model.h5 is missing) and extracts the image embeddings.

# This will generate best_cnn_model.h5 (if training) and cnn_image_features.npy
python src/sample_code.py


(Note: If training fails due to OOM, reduce IMAGE_SIZE in sample_code.py to (128, 128) or (96, 96).)

Step 3: Train Final Ensemble Model (LightGBM)

This script loads all three feature files (.npy and CSV), combines them, trains the final LightGBM model, and reports the final validation SMAPE score.

# This will train the final stacking model and output the performance metric.
python src/ensemble_model.py
