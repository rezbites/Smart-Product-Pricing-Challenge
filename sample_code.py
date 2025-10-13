import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import train_test_split
import re
import gc

tf.keras.backend.clear_session()
gc.collect()

# --- Configuration ---
# Get the absolute path to the directory where the script is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Build absolute paths to the data and images directories based on the new structure.
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
IMAGE_DIR = os.path.join(SCRIPT_DIR, 'images')

BATCH_SIZE = 1
IMAGE_SIZE = (128, 128)
MODEL_SAVE_PATH = 'best_cnn_model.h5'
# Using a small batch size for feature extraction to prevent OOM
FEATURE_EXTRACTION_BATCH_SIZE = 1

# --- SMAPE Metric Function ---
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

# --- Custom Keras Callback for SMAPE ---
class SmapeCallback(Callback):
    def __init__(self, val_df, val_dataset):
        super().__init__()
        self.val_df = val_df
        self.val_dataset = val_dataset
    
    def on_epoch_end(self, epoch, logs=None):
        y_pred_log = self.model.predict(self.val_dataset, verbose=0)
        y_pred_actual = np.expm1(y_pred_log)
        y_true_actual = np.expm1(self.val_df['price_log'].values)
        validation_smape = smape(y_true_actual, y_pred_actual.flatten())
        print(f" - val_smape: {validation_smape:.4f}%")

# --- Main Script ---
def run_cnn_model_pipeline():
    print("Starting ML Challenge CNN pipeline...")

    # --- 1. Load Data with Error Handling ---
    try:
        df_train = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
        df_train['image_file'] = df_train['image_link'].apply(lambda x: os.path.basename(x))
        df_train['price_log'] = np.log1p(df_train['price'])
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure train.csv is in the correct directory: {os.path.abspath(DATASET_DIR)}.")
        return

    # --- 2. Filter out Missing Images ---
    try:
        existing_images = set(os.listdir(IMAGE_DIR))
        initial_count = len(df_train)
        df_train = df_train[df_train['image_file'].isin(existing_images)]
        if len(df_train) < initial_count:
            print(f"Warning: Removed {initial_count - len(df_train)} records due to missing image files.")
        if df_train.empty:
            print("Error: No image files were found in the training data. Exiting.")
            return
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure your image directory is correct: {os.path.abspath(IMAGE_DIR)}.")
        return
    
    # --- 3. Split Data ---
    train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42)
    print(f"Training on {len(train_df)} images, validating on {len(val_df)} images.")

    # --- 4. tf.data dataset replacement ---
    def preprocess_image(file_path, label, image_size=(192, 192)):
        try:
            image = tf.io.read_file(file_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            if tf.shape(image).shape[0] == 0:
                raise tf.errors.InvalidArgumentError(None, None, "Could not decode image.")
            image = tf.image.resize(image, image_size)
            image = image / 255.0
            return image, label
        except Exception as e:
            print(f"Error processing image {file_path.numpy()}: {e}")
            placeholder_image = tf.zeros(list(image_size) + [3], dtype=tf.float32)
            return placeholder_image, label

    def build_dataset(df, image_dir, batch_size, shuffle=True):
        file_paths = [os.path.join(image_dir, fname) for fname in df['image_file']]
        labels = df['price_log'].values.astype('float32')
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = build_dataset(train_df, IMAGE_DIR, BATCH_SIZE, shuffle=True)
    val_dataset = build_dataset(val_df, IMAGE_DIR, BATCH_SIZE, shuffle=False)
    
    # --- 5. Build and Compile the Optimized CNN Model ---
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(192, 192, 3))
    base_model.trainable = False
    
    inputs = base_model.input
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # --- 6. Train the Model with Callbacks ---
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    smape_callback = SmapeCallback(val_df, val_dataset)
    
    print("\nStarting CNN model training...")
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=10,
                        callbacks=[checkpoint, early_stopping, smape_callback])
    print("\nTraining complete. The best model has been saved.")
    
    # --- 7. Feature Extraction from the Best Model ---
    # Create the dataset for all training images for feature extraction
    all_train_dataset = build_dataset(df_train, IMAGE_DIR, FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)

    best_model = load_model(MODEL_SAVE_PATH, compile=False)
    feature_extractor = Model(inputs=best_model.input, outputs=best_model.layers[-4].output)
    
    print("\nExtracting image features from the entire training dataset...")
    image_features = feature_extractor.predict(all_train_dataset, verbose=1)
    print(f"Image features extracted successfully! Shape: {image_features.shape}")
    
    np.save('cnn_image_features.npy', image_features)
    df_train[['sample_id']].to_csv('cnn_image_sample_ids.csv', index=False)
    
    print("\nSaved extracted image features and sample IDs.")

if __name__ == "__main__":
    run_cnn_model_pipeline()