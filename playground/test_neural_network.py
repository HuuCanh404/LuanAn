#!/usr/bin/env python3
"""
Neural Network Model - Du doan CPU
Su dung MLPRegressor de cai thien accuracy
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
CSV_PATH = Path(__file__).resolve().parent / "cpu_ram_disk_net.csv"
WINDOW_SIZE = 20  # Tang window lon hon
PREDICT_AHEAD = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

USE_COLS = [
    "cpu_percent",
    "ram_percent", 
    "disk_read_Bps",
    "disk_write_Bps",
    "net_in_Bps",
    "net_out_Bps",
]


def load_and_prepare_data():
    """Load va chuan bi du lieu"""
    df = pd.read_csv(CSV_PATH)
    df = df[USE_COLS]
    
    # Feature Engineering
    # Moving averages
    for col in ['cpu_percent', 'ram_percent']:
        df[f'{col}_ma3'] = df[col].rolling(3).mean()
        df[f'{col}_ma5'] = df[col].rolling(5).mean()
        df[f'{col}_ma10'] = df[col].rolling(10).mean()
        df[f'{col}_std5'] = df[col].rolling(5).std()
        df[f'{col}_diff'] = df[col].diff()
    
    # Lag features for CPU
    for lag in [1, 2, 3, 5, 10]:
        df[f'cpu_lag{lag}'] = df['cpu_percent'].shift(lag)
    
    # Network/Disk totals
    df['net_total'] = df['net_in_Bps'] + df['net_out_Bps']
    df['disk_total'] = df['disk_read_Bps'] + df['disk_write_Bps']
    
    # Log transform for skewed features
    for col in ['disk_read_Bps', 'disk_write_Bps', 'net_in_Bps', 'net_out_Bps']:
        df[f'{col}_log'] = np.log1p(df[col])
    
    # Fill NaN
    df = df.fillna(method='bfill').fillna(0)
    
    return df


def build_dataset(data, window_size, predict_ahead):
    """Build X, y for time series"""
    X, y = [], []
    
    for i in range(len(data) - window_size - predict_ahead):
        X.append(data[i:i + window_size].flatten())
        y.append(data[i + window_size + predict_ahead, 0])  # cpu_percent
    
    return np.array(X), np.array(y)


def main():
    print("=" * 70)
    print("NEURAL NETWORK MODEL - HE THONG DU DOAN CPU")
    print("=" * 70)

    # Load data
    print("\n[1] Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"    Samples: {len(df)}")
    print(f"    Features: {df.shape[1]}")

    # Scale data
    print("\n[2] Scaling data...")
    scaler_X = StandardScaler()
    scaled = scaler_X.fit_transform(df)
    
    # Luu min/max cua CPU de inverse transform
    cpu_mean = df['cpu_percent'].mean()
    cpu_std = df['cpu_percent'].std()

    # Build dataset
    print("\n[3] Building dataset...")
    X, y = build_dataset(scaled, WINDOW_SIZE, PREDICT_AHEAD)
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )
    print(f"\n[4] Train/Test split:")
    print(f"    Train: {len(X_train)}")
    print(f"    Test: {len(X_test)}")

    # Try different NN configurations
    configs = [
        {'hidden_layer_sizes': (64,), 'name': 'MLP (64)'},
        {'hidden_layer_sizes': (128, 64), 'name': 'MLP (128-64)'},
        {'hidden_layer_sizes': (256, 128, 64), 'name': 'MLP (256-128-64)'},
        {'hidden_layer_sizes': (128, 64, 32), 'name': 'MLP (128-64-32)'},
    ]

    print("\n" + "=" * 70)
    print("[5] TRAINING NEURAL NETWORKS...")
    print("=" * 70)

    results = []
    for config in configs:
        name = config.pop('name')
        print(f"\n    Training {name}...", end=" ", flush=True)
        
        model = MLPRegressor(
            **config,
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics (on scaled data)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Convert back to CPU percentage for interpretable MAE
        mae_cpu = mae * cpu_std
        rmse_cpu = rmse * cpu_std
        
        results.append({
            'name': name,
            'mae': mae_cpu,
            'rmse': rmse_cpu,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'config': config
        })
        
        print(f"Done! R2={r2:.4f}, MAE={mae_cpu:.2f}%")

    # Sort by R2
    results.sort(key=lambda x: x['r2'], reverse=True)

    # Print comparison
    print("\n" + "=" * 70)
    print("[RESULTS] NEURAL NETWORK COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} | {'MAE':>10} | {'RMSE':>10} | {'R2':>10}")
    print("-" * 70)
    
    for r in results:
        marker = " [BEST]" if r == results[0] else ""
        print(f"{r['name']:<25} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | {r['r2']:>10.4f}{marker}")

    # Best model
    best = results[0]
    print("\n" + "=" * 70)
    print(f"[BEST] {best['name']}")
    print("=" * 70)
    
    # Accuracy by threshold (nho scale lai)
    print("\n    Accuracy by threshold (scaled):")
    for threshold in [0.5, 1.0, 1.5, 2.0]:
        errors = np.abs(best['y_test'] - best['y_pred'])
        correct = errors <= threshold
        acc = np.mean(correct) * 100
        print(f"    +-{threshold} std: {acc:.2f}%")

    # Sample predictions
    print("\n    Sample predictions (scaled values):")
    print(f"    {'Actual':>10} | {'Predicted':>10} | {'Error':>10}")
    print("    " + "-" * 36)
    for i in range(min(10, len(best['y_test']))):
        actual = best['y_test'][i]
        pred = best['y_pred'][i]
        error = actual - pred
        print(f"    {actual:>10.4f} | {pred:>10.4f} | {error:>+10.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    old_r2 = 0.1985
    new_r2 = best['r2']
    improvement = ((new_r2 - old_r2) / abs(old_r2)) * 100 if old_r2 != 0 else 0
    
    print(f"    Original LinearRegression R2: {old_r2:.4f}")
    print(f"    Best Neural Network R2: {new_r2:.4f}")
    print(f"    Improvement: {improvement:+.1f}%")
    print(f"    MAE: {best['mae']:.2f}% CPU")

    if new_r2 > 0.5:
        print("\n    [SUCCESS] Da cai thien dang ke!")
    elif new_r2 > old_r2:
        print("\n    [OK] Co cai thien nhe")
    else:
        print("\n    [INFO] Khong cai thien - Data co the qua nhieu noise")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
