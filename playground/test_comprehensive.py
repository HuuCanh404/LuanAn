#!/usr/bin/env python3
"""
Final Comprehensive Test - Du doan CPU
Approach: Classification (trend) + Optimized Regression
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = Path(__file__).resolve().parent / "cpu_ram_disk_net.csv"

# Config
WINDOW_SIZE = 10
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


def load_data():
    df = pd.read_csv(CSV_PATH)
    return df[USE_COLS]


def create_classification_target(y, threshold=2.0):
    """
    Chuyen ve classification:
    0 = Giam (CPU giam > threshold%)
    1 = On dinh 
    2 = Tang (CPU tang > threshold%)
    """
    labels = []
    for i in range(len(y) - PREDICT_AHEAD):
        diff = y[i + PREDICT_AHEAD] - y[i]
        if diff < -threshold:
            labels.append(0)  # Giam
        elif diff > threshold:
            labels.append(2)  # Tang
        else:
            labels.append(1)  # On dinh
    return np.array(labels)


def build_regression_dataset(df, window_size, predict_ahead):
    """Build dataset cho regression"""
    X, y = [], []
    cpu_values = df['cpu_percent'].values
    
    # Normalize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    
    for i in range(len(scaled) - window_size - predict_ahead):
        # Features: window cua tat ca features + statistics
        window = scaled[i:i + window_size]
        
        # Flatten window
        features = window.flatten()
        
        # Them statistics cua window
        cpu_window = cpu_values[i:i + window_size]
        stats = [
            np.mean(cpu_window),
            np.std(cpu_window),
            np.min(cpu_window),
            np.max(cpu_window),
            cpu_window[-1] - cpu_window[0],  # trend
        ]
        features = np.concatenate([features, stats])
        
        X.append(features)
        y.append(cpu_values[i + window_size + predict_ahead])
    
    return np.array(X), np.array(y), scaler


def build_classification_dataset(df, window_size, predict_ahead, threshold=3.0):
    """Build dataset cho classification"""
    X, y = [], []
    cpu_values = df['cpu_percent'].values
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    
    for i in range(len(scaled) - window_size - predict_ahead):
        window = scaled[i:i + window_size]
        features = window.flatten()
        
        # Statistics
        cpu_window = cpu_values[i:i + window_size]
        stats = [
            np.mean(cpu_window),
            np.std(cpu_window),
            cpu_window[-1] - cpu_window[0],
        ]
        features = np.concatenate([features, stats])
        
        # Target: trend
        current_cpu = cpu_values[i + window_size - 1]
        future_cpu = cpu_values[i + window_size + predict_ahead]
        diff = future_cpu - current_cpu
        
        if diff < -threshold:
            label = 0  # Giam
        elif diff > threshold:
            label = 2  # Tang
        else:
            label = 1  # On dinh
        
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)


def main():
    print("=" * 70)
    print("COMPREHENSIVE CPU PREDICTION TEST")
    print("=" * 70)

    df = load_data()
    print(f"\n[DATA] Loaded {len(df)} samples")

    # ============================================
    # PART 1: REGRESSION WITH OPTIMIZATIONS
    # ============================================
    print("\n" + "=" * 70)
    print("PART 1: OPTIMIZED REGRESSION")
    print("=" * 70)

    X_reg, y_reg, scaler = build_regression_dataset(df, WINDOW_SIZE, PREDICT_AHEAD)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )
    
    print(f"\n    Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"    Features per sample: {X_train.shape[1]}")

    # Train optimized model
    print("\n    Training Gradient Boosting (optimized)...")
    model_reg = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    model_reg.fit(X_train, y_train)
    
    y_pred = model_reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n    [REGRESSION RESULTS]")
    print(f"    MAE:  {mae:.4f}%")
    print(f"    RMSE: {rmse:.4f}%")
    print(f"    R2:   {r2:.4f}")

    # Accuracy with thresholds  
    print("\n    Accuracy by error threshold:")
    for thresh in [2, 5, 10]:
        errors = np.abs(y_test - y_pred)
        acc = np.mean(errors <= thresh) * 100
        print(f"    +-{thresh}% error: {acc:.1f}% correct")

    # ============================================
    # PART 2: CLASSIFICATION (TREND PREDICTION)
    # ============================================
    print("\n" + "=" * 70)
    print("PART 2: TREND CLASSIFICATION")
    print("=" * 70)

    X_cls, y_cls = build_classification_dataset(df, WINDOW_SIZE, PREDICT_AHEAD, threshold=3.0)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )

    # Class distribution
    unique, counts = np.unique(y_test_c, return_counts=True)
    print(f"\n    Class distribution in test:")
    labels_map = {0: 'Giam', 1: 'On dinh', 2: 'Tang'}
    for u, c in zip(unique, counts):
        print(f"    {labels_map[u]}: {c} ({c/len(y_test_c)*100:.1f}%)")

    print("\n    Training Random Forest Classifier...")
    model_cls = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model_cls.fit(X_train_c, y_train_c)
    
    y_pred_c = model_cls.predict(X_test_c)
    
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"\n    [CLASSIFICATION RESULTS]")
    print(f"    Accuracy: {acc*100:.2f}%")
    print(f"\n    Classification Report:")
    print("    " + "-" * 50)
    
    for line in classification_report(y_test_c, y_pred_c, 
                                       target_names=['Giam', 'On dinh', 'Tang']).split('\n'):
        if line.strip():
            print(f"    {line}")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n    [REGRESSION] Optimized Gradient Boosting")
    print(f"    - R2 Score: {r2:.4f}")
    print(f"    - MAE: {mae:.2f}% CPU")
    print(f"    - Accuracy (+-5%): {np.mean(np.abs(y_test - y_pred) <= 5) * 100:.1f}%")
    
    print(f"\n    [CLASSIFICATION] Random Forest Trend Prediction")
    print(f"    - Accuracy: {acc*100:.1f}%")
    print(f"    - Predicts: Tang/On dinh/Giam")
    
    # Comparison with original
    print("\n" + "-" * 70)
    original_r2 = 0.1985
    improvement = ((r2 - original_r2) / abs(original_r2)) * 100
    print(f"    Original LinearRegression R2: {original_r2:.4f}")
    print(f"    Current Best R2: {r2:.4f}")
    print(f"    Improvement: {improvement:+.1f}%")
    
    if r2 > 0.4:
        print("\n    [SUCCESS] Significant improvement achieved!")
    elif r2 > original_r2:
        print("\n    [OK] Some improvement achieved")
    else:
        print("\n    [NOTE] Limited improvement - CPU workload has inherent randomness")
        print("          Consider: ARIMA, LSTM, or collecting more relevant features")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
