# 02_train_lstm_model.py

import numpy as np
import pandas as pd
import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# --- MODIFICATION START: Import tensorflow.keras.backend ---
import tensorflow.keras.backend as K # To clear session
# --- MODIFICATION END ---
import matplotlib.pyplot as plt
import traceback
import tensorflow as tf
import gc # Garbage collector

# --- Configuration ---
PROCESSED_DATA_DIR = "data_processed_lstm_garch_ret" # Use data from simplified features run
MODEL_SAVE_DIR = "models"
# --- MODIFICATION START: Base Model Filename ---
BASE_MODEL_FILENAME = "lstm_usdinr_model_run_{}.h5" # Template for numbered models
N_ENSEMBLE_RUNS = 3 # Number of models to train for the ensemble
# --- MODIFICATION END ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- LSTM Model Parameters ---
# Use parameters from the best run (Suggestion 7)
LSTM_UNITS_1 = 100
LSTM_UNITS_2 = 50
DROPOUT_RATE = 0.25
EPOCHS = 100
BATCH_SIZE = 16 # Keep smaller batch size
EARLY_STOPPING_PATIENCE = 10
LR_REDUCTION_PATIENCE = 5

# --- Function to Build Model ---
# Encapsulate model building to ensure fresh model each time
def build_lstm_model(n_timesteps, n_features):
    print(f"\nBuilding model instance...")
    print(f"Input shape: (Timesteps={n_timesteps}, Features={n_features})")
    inputs = tf.keras.Input(shape=(n_timesteps, n_features))
    bilstm1 = Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True))(inputs)
    dropout1 = Dropout(DROPOUT_RATE)(bilstm1)
    bilstm2 = Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=False))(dropout1) # Last LSTM before Dense is False
    dropout2 = Dropout(DROPOUT_RATE)(bilstm2)
    dense_intermediate = Dense(LSTM_UNITS_2 // 2, activation='relu')(dropout2)
    dropout_final = Dropout(DROPOUT_RATE)(dense_intermediate)
    outputs = Dense(1)(dropout_final)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mae')
    print("Model instance built and compiled.")
    return model

# --- 1. Load Processed Sequence Data ---
# (Load data remains the same as Suggestion 7 / last run)
print(f"Loading processed sequence data from '{PROCESSED_DATA_DIR}'...")
try:
    X_train_seq = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train_seq.npy'))
    y_train_seq = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train_seq_ret.npy'))
    X_val_seq = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val_seq.npy'))
    y_val_seq = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val_seq_ret.npy'))
    # We don't need test data during the training loop itself for ensembling
except FileNotFoundError as e: print(f"Error loading data: {e}"); exit()
except Exception as e: print(f"An error occurred loading data: {e}"); traceback.print_exc(); exit()
print("Sequence data loaded successfully.")
# ... (print shapes) ...
try:
    loaded_seq_len = int(np.load(os.path.join(PROCESSED_DATA_DIR, 'sequence_length.npy'))[0])
    print(f"Loaded sequence length from file: {loaded_seq_len}")
    n_timesteps = loaded_seq_len
    n_features = X_train_seq.shape[2]
except FileNotFoundError: print("Error: sequence_length.npy not found."); exit()


# --- MODIFICATION START: Training Loop for Ensemble ---
print(f"\n--- Starting Ensemble Training ({N_ENSEMBLE_RUNS} runs) ---")

for i in range(N_ENSEMBLE_RUNS):
    run_num = i + 1
    print(f"\n--- Training Run {run_num}/{N_ENSEMBLE_RUNS} ---")

    # Clear session and build fresh model
    K.clear_session()
    gc.collect()
    model = build_lstm_model(n_timesteps, n_features)
    model.summary()

    # Define Callbacks for this run
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=LR_REDUCTION_PATIENCE, verbose=1, min_lr=1e-6)

    # Train the model instance
    print(f"Training model instance {run_num}...")
    try:
        history = model.fit( X_train_seq, y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(X_val_seq, y_val_seq), callbacks=[early_stopping, reduce_lr], verbose=1 ) # Set verbose=1 or 2
        print(f"Model training instance {run_num} finished.")

        # Save the trained model instance
        model_path = os.path.join(MODEL_SAVE_DIR, BASE_MODEL_FILENAME.format(run_num))
        try:
            model.save(model_path)
            print(f"LSTM model instance {run_num} saved to {model_path}")
        except Exception as save_e:
            print(f"Error saving LSTM model instance {run_num}: {save_e}"); traceback.print_exc()

        # Optional: Plot and save history for this run individually
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label=f'Run {run_num} Train MAE')
        plt.plot(history.history['val_loss'], label=f'Run {run_num} Val MAE')
        plt.title(f'Model Loss During Training (Run {run_num})')
        plt.ylabel('MAE on Returns'); plt.xlabel('Epoch'); plt.legend(loc='upper right'); plt.grid(True)
        history_plot_filename = os.path.join(MODEL_SAVE_DIR, f'lstm_training_history_run_{run_num}.png')
        plt.savefig(history_plot_filename); print(f"Saved training history plot to {history_plot_filename}"); plt.close()

    except Exception as train_e:
        print(f"Error during model training instance {run_num}: {train_e}"); traceback.print_exc()

    # Clean up memory (optional but good practice in loops)
    del model
    del history
    gc.collect()

print(f"\n--- Ensemble Training Complete ---")
print(f"Models saved in '{MODEL_SAVE_DIR}' with pattern '{BASE_MODEL_FILENAME.format('<run_num>')}'")
print("Proceed to '05_evaluate_ensemble.py' to evaluate the ensemble performance.")

# --- Remove Evaluation and Plotting from this script ---
# The evaluation will now happen in the separate ensemble script.
# --- END MODIFICATION ---