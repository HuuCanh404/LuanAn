#!/usr/bin/env python3
"""
Ba mo hinh AI cho he thong du doan CPU (Three AI Models for CPU Forecasting)

Mo hinh 1: GradientBoostingRegressor - Du doan gia tri CPU (regression)
Mo hinh 2: RandomForestClassifier    - Du doan xu huong CPU (classification)
Mo hinh 3: MLPRegressor              - Neural network du doan CPU (regression)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)

# =========================
# CONFIG
# =========================
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

TREND_THRESHOLD = 3.0  # % CPU change to classify as increase/decrease
BASELINE_R2 = 0.1985  # LinearRegression baseline R2 score for comparison


# =========================
# DATA PREPARATION
# =========================
def load_data(csv_path):
    """Load du lieu tu CSV file."""
    df = pd.read_csv(csv_path)
    return df[USE_COLS]


def add_features(df):
    """Feature engineering: them cac features moi tu du lieu goc."""
    df = df.copy()

    # Moving averages
    for col in ["cpu_percent", "ram_percent"]:
        df[f"{col}_ma3"] = df[col].rolling(3).mean()
        df[f"{col}_ma5"] = df[col].rolling(5).mean()
        df[f"{col}_std5"] = df[col].rolling(5).std()
        df[f"{col}_diff"] = df[col].diff()

    # Lag features for CPU
    for lag in [1, 2, 3, 5]:
        df[f"cpu_lag{lag}"] = df["cpu_percent"].shift(lag)

    # Totals
    df["net_total"] = df["net_in_Bps"] + df["net_out_Bps"]
    df["disk_total"] = df["disk_read_Bps"] + df["disk_write_Bps"]

    # Log transform for skewed features
    for col in ["disk_read_Bps", "disk_write_Bps", "net_in_Bps", "net_out_Bps"]:
        df[f"{col}_log"] = np.log1p(df[col])

    df = df.ffill().bfill().fillna(0)
    return df


# =========================
# DATASET BUILDERS
# =========================
def build_regression_dataset(df, window_size=WINDOW_SIZE, predict_ahead=PREDICT_AHEAD):
    """Build dataset cho regression (Mo hinh 1 va 3)."""
    X, y = [], []
    cpu_values = df["cpu_percent"].values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    for i in range(len(scaled) - window_size - predict_ahead):
        window = scaled[i : i + window_size]
        features = window.flatten()

        # Statistics cua CPU window
        cpu_window = cpu_values[i : i + window_size]
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


def build_classification_dataset(
    df, window_size=WINDOW_SIZE, predict_ahead=PREDICT_AHEAD, threshold=TREND_THRESHOLD
):
    """Build dataset cho classification (Mo hinh 2)."""
    X, y = [], []
    cpu_values = df["cpu_percent"].values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    for i in range(len(scaled) - window_size - predict_ahead):
        window = scaled[i : i + window_size]
        features = window.flatten()

        # Statistics
        cpu_window = cpu_values[i : i + window_size]
        stats = [
            np.mean(cpu_window),
            np.std(cpu_window),
            cpu_window[-1] - cpu_window[0],
        ]
        features = np.concatenate([features, stats])

        # Target: trend label
        current_cpu = cpu_values[i + window_size - 1]
        future_cpu = cpu_values[i + window_size + predict_ahead]
        diff = future_cpu - current_cpu

        if diff < -threshold:
            label = 0  # Giam (decrease)
        elif diff > threshold:
            label = 2  # Tang (increase)
        else:
            label = 1  # On dinh (stable)

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


TREND_LABELS = {0: "Giam", 1: "On dinh", 2: "Tang"}


# =========================
# BA MO HINH AI (THREE AI MODELS)
# =========================
def create_model_1():
    """Mo hinh 1: GradientBoostingRegressor - Du doan gia tri CPU.

    Regression model su dung ensemble cua decision trees.
    Toi uu cho du doan gia tri CPU % trong tuong lai.
    """
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )


def create_model_2():
    """Mo hinh 2: RandomForestClassifier - Du doan xu huong CPU.

    Classification model du doan xu huong CPU:
    - Giam (decrease)
    - On dinh (stable)
    - Tang (increase)
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_model_3():
    """Mo hinh 3: MLPRegressor - Neural network du doan CPU.

    Multi-Layer Perceptron voi 3 hidden layers (256-128-64 neurons).
    Su dung ReLU activation va Adam optimizer.
    """
    return MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )


# =========================
# EVALUATION
# =========================
def evaluate_regression(model, X_train, X_test, y_train, y_test):
    """Danh gia regression model. Tra ve dict cac metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Accuracy by error thresholds
    errors = np.abs(y_test - y_pred)
    acc_5 = np.mean(errors <= 5) * 100
    acc_10 = np.mean(errors <= 10) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "accuracy_5pct": acc_5,
        "accuracy_10pct": acc_10,
        "y_pred": y_pred,
    }


def evaluate_classification(model, X_train, X_test, y_train, y_test):
    """Danh gia classification model. Tra ve dict cac metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Giam", "On dinh", "Tang"], zero_division=0
    )

    return {
        "accuracy": acc,
        "classification_report": report,
        "y_pred": y_pred,
    }


# =========================
# MAIN: CHAY CA 3 MO HINH
# =========================
def run_all_models(csv_path):
    """Chay va danh gia ca 3 mo hinh AI.

    Args:
        csv_path: Duong dan den file CSV chua du lieu he thong.

    Returns:
        Dict chua ket qua cua 3 mo hinh.
    """
    print("=" * 70)
    print("BA MO HINH AI - HE THONG DU DOAN CPU")
    print("=" * 70)

    # Load data
    df = load_data(csv_path)
    print(f"\n[DATA] Loaded {len(df)} samples, {len(USE_COLS)} features")

    # ============================================
    # MO HINH 1: GradientBoostingRegressor
    # ============================================
    print("\n" + "=" * 70)
    print("MO HINH 1: GradientBoostingRegressor (Regression)")
    print("=" * 70)

    X_reg, y_reg, scaler = build_regression_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

    model_1 = create_model_1()
    results_1 = evaluate_regression(model_1, X_train, X_test, y_train, y_test)

    print(f"  MAE:  {results_1['mae']:.4f}%")
    print(f"  RMSE: {results_1['rmse']:.4f}%")
    print(f"  R2:   {results_1['r2']:.4f}")
    print(f"  Accuracy (+-5%): {results_1['accuracy_5pct']:.1f}%")
    print(f"  Accuracy (+-10%): {results_1['accuracy_10pct']:.1f}%")

    # ============================================
    # MO HINH 2: RandomForestClassifier
    # ============================================
    print("\n" + "=" * 70)
    print("MO HINH 2: RandomForestClassifier (Classification)")
    print("=" * 70)

    X_cls, y_cls = build_classification_dataset(df)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train_c)}, Test: {len(X_test_c)}")

    # Show class distribution
    unique, counts = np.unique(y_test_c, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {TREND_LABELS[u]}: {c} ({c / len(y_test_c) * 100:.1f}%)")

    model_2 = create_model_2()
    results_2 = evaluate_classification(
        model_2, X_train_c, X_test_c, y_train_c, y_test_c
    )

    print(f"  Accuracy: {results_2['accuracy'] * 100:.2f}%")
    print(f"\n  Classification Report:")
    for line in results_2["classification_report"].split("\n"):
        if line.strip():
            print(f"  {line}")

    # ============================================
    # MO HINH 3: MLPRegressor (Neural Network)
    # ============================================
    print("\n" + "=" * 70)
    print("MO HINH 3: MLPRegressor - Neural Network (Regression)")
    print("=" * 70)

    # Use same regression dataset but with extra feature engineering
    df_nn = add_features(df)
    scaler_nn = StandardScaler()
    scaled_nn = scaler_nn.fit_transform(df_nn)

    X_nn, y_nn = [], []
    cpu_values = df_nn["cpu_percent"].values
    for i in range(len(scaled_nn) - WINDOW_SIZE - PREDICT_AHEAD):
        X_nn.append(scaled_nn[i : i + WINDOW_SIZE].flatten())
        y_nn.append(cpu_values[i + WINDOW_SIZE + PREDICT_AHEAD])
    X_nn, y_nn = np.array(X_nn), np.array(y_nn)

    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        X_nn, y_nn, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )
    print(
        f"  Train: {len(X_train_nn)}, Test: {len(X_test_nn)}, Features: {X_train_nn.shape[1]}"
    )

    model_3 = create_model_3()
    results_3 = evaluate_regression(model_3, X_train_nn, X_test_nn, y_train_nn, y_test_nn)

    print(f"  MAE:  {results_3['mae']:.4f}%")
    print(f"  RMSE: {results_3['rmse']:.4f}%")
    print(f"  R2:   {results_3['r2']:.4f}")
    print(f"  Accuracy (+-5%): {results_3['accuracy_5pct']:.1f}%")
    print(f"  Accuracy (+-10%): {results_3['accuracy_10pct']:.1f}%")

    # ============================================
    # TONG KET (SUMMARY)
    # ============================================
    print("\n" + "=" * 70)
    print("TONG KET BA MO HINH AI")
    print("=" * 70)

    print(f"\n  {'Mo hinh':<45} | {'Metric':>15}")
    print("  " + "-" * 65)
    print(
        f"  {'1. GradientBoostingRegressor (Regression)':<45} | R2 = {results_1['r2']:.4f}"
    )
    print(
        f"  {'2. RandomForestClassifier (Classification)':<45} | Acc = {results_2['accuracy'] * 100:.1f}%"
    )
    print(
        f"  {'3. MLPRegressor / Neural Network (Regression)':<45} | R2 = {results_3['r2']:.4f}"
    )

    original_r2 = BASELINE_R2
    best_r2 = max(results_1["r2"], results_3["r2"])
    improvement = ((best_r2 - original_r2) / abs(original_r2)) * 100

    print(f"\n  Baseline LinearRegression R2: {original_r2:.4f}")
    print(f"  Best R2 (regression): {best_r2:.4f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print("=" * 70)

    return {
        "model_1_gradient_boosting": results_1,
        "model_2_random_forest_cls": results_2,
        "model_3_neural_network": results_3,
    }
