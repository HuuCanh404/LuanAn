#!/usr/bin/env python3
"""
Test Accuracy cho Model Dự Đoán CPU
Đánh giá độ chính xác của LinearRegression model dựa trên 4 chỉ số:
CPU, RAM, disk_read, net_out
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
CSV_PATH = Path(__file__).resolve().parent / "cpu_ram_disk_net.csv"
WINDOW_SIZE = 10
PREDICT_AHEAD = 5
TARGET_COL = "cpu_percent"
TEST_SIZE = 0.2  # 20% cho test
RANDOM_STATE = 42

USE_COLS = [
    "cpu_percent",
    "ram_percent",
    "disk_read_Bps",
    "net_out_Bps",
]


def load_data():
    """Load và chuẩn bị dữ liệu từ CSV"""
    df = pd.read_csv(CSV_PATH)
    df = df[USE_COLS]
    return df


def build_dataset(scaled):
    """
    Xây dựng dataset cho training/testing
    X: window của WINDOW_SIZE điểm dữ liệu (flatten)
    y: giá trị target tại t + PREDICT_AHEAD
    """
    X, y = [], []
    target_idx = USE_COLS.index(TARGET_COL)

    for i in range(len(scaled) - WINDOW_SIZE - PREDICT_AHEAD):
        X.append(scaled[i:i + WINDOW_SIZE].flatten())
        y.append(scaled[i + WINDOW_SIZE + PREDICT_AHEAD][target_idx])

    return np.array(X), np.array(y)


def inverse_transform_cpu(pred_norm, scaler):
    """Chuyển đổi giá trị normalized về giá trị thực"""
    cpu_idx = USE_COLS.index(TARGET_COL)
    cpu_min = scaler.data_min_[cpu_idx]
    cpu_max = scaler.data_max_[cpu_idx]
    return pred_norm * (cpu_max - cpu_min) + cpu_min


def calculate_accuracy_threshold(y_true, y_pred, threshold_percent):
    """
    Tính accuracy dựa trên threshold
    Một prediction được coi là đúng nếu sai số <= threshold_percent
    """
    errors = np.abs(y_true - y_pred)
    threshold = threshold_percent * np.abs(y_true) / 100  # threshold tương đối
    threshold = np.maximum(threshold, 1.0)  # minimum threshold 1%
    correct = errors <= threshold
    return np.mean(correct) * 100


def main():
    print("=" * 60)
    print("[TEST] TEST ACCURACY - HE THONG DU DOAN CPU")
    print("=" * 60)

    # Load data
    print("\n[*] Loading data...")
    df = load_data()
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {USE_COLS}")

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Build dataset
    X, y = build_dataset(scaled)
    print(f"   Dataset size after windowing: {len(X)}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )
    print(f"\n[*] Train/Test Split:")
    print(f"   Train size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")

    # Train model
    print("\n[*] Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("   Model trained successfully!")

    # Predict
    print("\n[*] Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Inverse transform về giá trị CPU thực
    y_train_real = inverse_transform_cpu(y_train, scaler)
    y_test_real = inverse_transform_cpu(y_test, scaler)
    y_pred_train_real = inverse_transform_cpu(y_pred_train, scaler)
    y_pred_test_real = inverse_transform_cpu(y_pred_test, scaler)

    # Calculate metrics
    print("\n" + "=" * 60)
    print("[TRAIN] TRAINING METRICS")
    print("=" * 60)
    train_mae = mean_absolute_error(y_train_real, y_pred_train_real)
    train_rmse = np.sqrt(mean_squared_error(y_train_real, y_pred_train_real))
    train_r2 = r2_score(y_train_real, y_pred_train_real)

    print(f"   MAE:  {train_mae:.4f}%")
    print(f"   RMSE: {train_rmse:.4f}%")
    print(f"   R²:   {train_r2:.4f}")

    print("\n" + "=" * 60)
    print("[TEST] TEST METRICS")
    print("=" * 60)
    test_mae = mean_absolute_error(y_test_real, y_pred_test_real)
    test_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_test_real))
    test_r2 = r2_score(y_test_real, y_pred_test_real)

    print(f"   MAE:  {test_mae:.4f}%")
    print(f"   RMSE: {test_rmse:.4f}%")
    print(f"   R²:   {test_r2:.4f}")

    # Accuracy với các thresholds
    print("\n" + "=" * 60)
    print("[ACCURACY] BY THRESHOLD (Test Set)")
    print("=" * 60)
    for threshold in [5, 10, 15, 20]:
        acc = calculate_accuracy_threshold(y_test_real, y_pred_test_real, threshold)
        print(f"   ±{threshold}% threshold: {acc:.2f}% accuracy")

    # Sample predictions
    print("\n" + "=" * 60)
    print("[SAMPLE] PREDICTIONS (First 10 test samples)")
    print("=" * 60)
    print(f"{'Actual':>12} | {'Predicted':>12} | {'Error':>12}")
    print("-" * 42)
    for i in range(min(10, len(y_test_real))):
        actual = y_test_real[i]
        pred = y_pred_test_real[i]
        error = actual - pred
        print(f"{actual:>12.2f} | {pred:>12.2f} | {error:>+12.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    print(f"   Model: LinearRegression")
    print(f"   Window Size: {WINDOW_SIZE}")
    print(f"   Predict Ahead: {PREDICT_AHEAD}s")
    print(f"   Test R² Score: {test_r2:.4f}")
    print(f"   Test MAE: {test_mae:.4f}%")

    if test_r2 > 0.7:
        print("\n   [OK] Model co do chinh xac TOT!")
    elif test_r2 > 0.4:
        print("\n   [WARN] Model co do chinh xac TRUNG BINH")
    else:
        print("\n   [FAIL] Model can cai thien")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
