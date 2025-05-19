import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def configure_gpu():
    """Allow dynamic GPU memory growth and XLA if a GPU is available."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) for dynamic memory growth.")
    else:
        print("No GPU found, running on CPU.")

    # Enable XLA JIT compilation
    tf.config.optimizer.set_jit(True)
    print("Enabled XLA JIT compilation.")


def load_and_preprocess(csv_path):
    print("Loading data from", csv_path)
    df = pd.read_csv(csv_path)
    print("Raw data shape:", df.shape)

    target_col = 'VALOR_HORA'
    drop_cols = ['ESTACION_CA', 'CONT_ID']
    feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
    print("Using features:", feature_cols)

    X_all = df[feature_cols].values.astype('float32')
    y_all = df[target_col].values.astype('float32').reshape(-1, 1)

    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    print(f"Split data: {len(X_train)} train samples, {len(X_test)} test samples")

    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)
    print("Features scaled.")

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test  = scaler_y.transform(y_test)
    print("Target scaled.")

    return X_train, y_train, X_test, y_test, scaler_y, len(feature_cols)


def build_model(window_size, n_features):
    print("Building LSTM model...")
    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model


def plot_and_save(history, y_true, y_pred, residuals, graphs_dir):
    os.makedirs(graphs_dir, exist_ok=True)

    # Training history
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend(); plt.grid(True);
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'training_history.png'))
    plt.close()

    # Actual vs Predicted (first 200)
    plt.figure(figsize=(10,5))
    plt.plot(y_true[:200], label='Actual', alpha=0.7)
    plt.plot(y_pred[:200], label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted (first 200)')
    plt.xlabel('Sample'); plt.ylabel('VALOR_HORA')
    plt.legend(); plt.grid(True);
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'actual_vs_predicted.png'))
    plt.close()

    # Residuals scatter
    plt.figure(figsize=(8,5))
    plt.scatter(np.arange(len(residuals)), residuals, alpha=0.3)
    plt.axhline(0, linestyle='--', color='r')
    plt.title('Residuals Plot')
    plt.xlabel('Sample'); plt.ylabel('Actual - Predicted')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'residuals_scatter.png'))
    plt.close()

    # Residuals histogram
    plt.figure(figsize=(8,5))
    plt.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual'); plt.ylabel('Frequency')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'residuals_hist.png'))
    plt.close()

    # Predicted vs Actual scatter
    plt.figure(figsize=(8,8))
    plt.scatter(y_true, y_pred, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.title('Predicted vs Actual')
    plt.xlabel('Actual VALOR_HORA'); plt.ylabel('Predicted VALOR_HORA')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'pred_vs_actual_scatter.png'))
    plt.close()


def main():
    print("=== TRAIN_VALOR_HORA START ===")
    mixed_precision.set_global_policy('mixed_float16')
    print("Enabled mixed precision")
    configure_gpu()
    
    #DATA_PATH = Path(r"C:/Users/evpue/Desktop/Progra/AI/content/output/dataset_listo_para_entrenar.csv")
    #GRAPHS_DIR = Path(r"C:/Users/evpue/Desktop/Progra/AI/content/graphs")
    
    DATA_PATH   = "content//output//dataset_listo_para_entrenar.csv"
    GRAPHS_DIR  = "content//graphs"
    WINDOW = 24; BATCH=256; EPOCHS=20

    X_tr, y_tr, X_te, y_te, scaler_y, nf = load_and_preprocess(DATA_PATH)

    train_gen = TimeseriesGenerator(X_tr, y_tr, length=WINDOW, batch_size=BATCH)
    test_gen  = TimeseriesGenerator(X_te, y_te, length=WINDOW, batch_size=BATCH)
    print(f"Batches: train={len(train_gen)}, test={len(test_gen)}")

    model = build_model(WINDOW, nf)
    ck = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(train_gen, validation_data=test_gen,
                        epochs=EPOCHS, callbacks=[ck,es], verbose=2)

    # Predictions
    y_pred_scaled = model.predict(test_gen)
    y_true_scaled = y_te[WINDOW:WINDOW+len(y_pred_scaled)]
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)
    residuals = (y_true - y_pred).flatten()

    #metrics
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4%}, R2: {r2:.4f}")

    # Save metrics to file
    with open(os.path.join(GRAPHS_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"MAE: {mae:.6f}\nRMSE: {rmse:.6f}\nMAPE: {mape:.6%}\nR2: {r2:.6f}\n")

    # Save plots
    plot_and_save(history, y_true.flatten(), y_pred.flatten(), residuals, GRAPHS_DIR)
    print("=== TRAIN_VALOR_HORA END ===")

if __name__=='__main__': main()
