#!/usr/bin/env python3
"""
IMPROVED Test Accuracy - So sanh nhieu model
Cai thien do chinh xac du doan CPU
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
CSV_PATH = Path(__file__).resolve().parent / "cpu_ram_disk_net.csv"
WINDOW_SIZE = 15  # Tang window size
PREDICT_AHEAD = 5
TARGET_COL = "cpu_percent"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Su dung TAT CA cac features
USE_COLS = [
    "cpu_percent",
    "ram_percent",
    "disk_read_Bps",
    "disk_write_Bps",  # Them
    "net_in_Bps",      # Them
    "net_out_Bps",
]


def load_data():
    """Load va chuan bi du lieu tu CSV"""
    df = pd.read_csv(CSV_PATH)
    df = df[USE_COLS]
    return df


def add_features(df):
    """Them cac features moi - Feature Engineering"""
    # Moving averages
    df['cpu_ma5'] = df['cpu_percent'].rolling(window=5).mean()
    df['cpu_ma10'] = df['cpu_percent'].rolling(window=10).mean()
    df['ram_ma5'] = df['ram_percent'].rolling(window=5).mean()
    
    # Difference (trend)
    df['cpu_diff'] = df['cpu_percent'].diff()
    df['ram_diff'] = df['ram_percent'].diff()
    
    # Lag features
    df['cpu_lag1'] = df['cpu_percent'].shift(1)
    df['cpu_lag2'] = df['cpu_percent'].shift(2)
    df['cpu_lag5'] = df['cpu_percent'].shift(5)
    
    # Statistics
    df['cpu_std5'] = df['cpu_percent'].rolling(window=5).std()
    df['net_total'] = df['net_in_Bps'] + df['net_out_Bps']
    df['disk_total'] = df['disk_read_Bps'] + df['disk_write_Bps']
    
    # Fill NaN
    df = df.fillna(method='bfill').fillna(0)
    return df


def build_dataset(scaled, n_features):
    """Xay dung dataset voi window"""
    X, y = [], []
    
    for i in range(len(scaled) - WINDOW_SIZE - PREDICT_AHEAD):
        X.append(scaled[i:i + WINDOW_SIZE].flatten())
        y.append(scaled[i + WINDOW_SIZE + PREDICT_AHEAD][0])  # cpu_percent la cot dau

    return np.array(X), np.array(y)


def inverse_transform_cpu(pred_norm, scaler):
    """Chuyen doi gia tri normalized ve gia tri thuc"""
    cpu_min = scaler.data_min_[0]
    cpu_max = scaler.data_max_[0]
    return pred_norm * (cpu_max - cpu_min) + cpu_min


def evaluate_model(model, X_train, X_test, y_train, y_test, scaler, model_name):
    """Danh gia mot model"""
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    # Inverse transform
    y_test_real = inverse_transform_cpu(y_test, scaler)
    y_pred_test_real = inverse_transform_cpu(y_pred_test, scaler)
    
    mae = mean_absolute_error(y_test_real, y_pred_test_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_test_real))
    r2 = r2_score(y_test_real, y_pred_test_real)
    
    return {
        'name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_test_real': y_test_real,
        'y_pred_real': y_pred_test_real
    }


def main():
    print("=" * 70)
    print("IMPROVED MODEL COMPARISON - HE THONG DU DOAN CPU")
    print("=" * 70)

    # Load and prepare data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"    Original samples: {len(df)}")
    print(f"    Original features: {len(USE_COLS)}")
    
    # Feature engineering
    print("\n[2] Feature engineering...")
    df = add_features(df)
    print(f"    Total features after engineering: {df.shape[1]}")
    print(f"    Features: {list(df.columns)}")

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # Build dataset
    X, y = build_dataset(scaled, df.shape[1])
    print(f"\n[3] Dataset prepared:")
    print(f"    Samples: {len(X)}")
    print(f"    Features per sample: {X.shape[1]}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )
    print(f"\n[4] Train/Test Split:")
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # Define models to compare
    models = [
        (LinearRegression(), "Linear Regression"),
        (Ridge(alpha=1.0), "Ridge Regression"),
        (Lasso(alpha=0.1), "Lasso Regression"),
        (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), "Random Forest"),
        (GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42), "Gradient Boosting"),
    ]

    # Evaluate all models
    print("\n" + "=" * 70)
    print("[5] EVALUATING MODELS...")
    print("=" * 70)
    
    results = []
    for model, name in models:
        print(f"\n    Training {name}...", end=" ")
        result = evaluate_model(model, X_train, X_test, y_train, y_test, scaler, name)
        results.append(result)
        print(f"Done! R2={result['r2']:.4f}")

    # Sort by R2
    results.sort(key=lambda x: x['r2'], reverse=True)

    # Print comparison table
    print("\n" + "=" * 70)
    print("[RESULTS] MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} | {'MAE':>10} | {'RMSE':>10} | {'R2':>10}")
    print("-" * 70)
    
    for r in results:
        status = "[BEST]" if r == results[0] else ""
        print(f"{r['name']:<25} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | {r['r2']:>10.4f} {status}")

    # Best model details
    best = results[0]
    print("\n" + "=" * 70)
    print(f"[BEST MODEL] {best['name']}")
    print("=" * 70)
    print(f"    MAE:  {best['mae']:.4f}%")
    print(f"    RMSE: {best['rmse']:.4f}%")
    print(f"    R2:   {best['r2']:.4f}")

    # Accuracy by threshold
    print("\n    Accuracy by threshold:")
    for threshold in [5, 10, 15, 20]:
        errors = np.abs(best['y_test_real'] - best['y_pred_real'])
        correct = errors <= threshold
        acc = np.mean(correct) * 100
        print(f"    +-{threshold}%: {acc:.2f}%")

    # Sample predictions
    print("\n    Sample predictions (first 10):")
    print(f"    {'Actual':>10} | {'Predicted':>10} | {'Error':>10}")
    print("    " + "-" * 36)
    for i in range(min(10, len(best['y_test_real']))):
        actual = best['y_test_real'][i]
        pred = best['y_pred_real'][i]
        error = actual - pred
        print(f"    {actual:>10.2f} | {pred:>10.2f} | {error:>+10.2f}")

    # Improvement summary
    print("\n" + "=" * 70)
    print("[SUMMARY] IMPROVEMENT RESULTS")
    print("=" * 70)
    
    old_r2 = 0.1985  # Tu ket qua truoc
    new_r2 = best['r2']
    improvement = ((new_r2 - old_r2) / abs(old_r2)) * 100 if old_r2 != 0 else 0
    
    print(f"    Previous R2 (LinearRegression): {old_r2:.4f}")
    print(f"    New R2 ({best['name']}): {new_r2:.4f}")
    print(f"    Improvement: {improvement:+.1f}%")
    
    if new_r2 > 0.7:
        print("\n    [SUCCESS] Model co do chinh xac TOT!")
    elif new_r2 > 0.4:
        print("\n    [OK] Model co do chinh xac TRUNG BINH")
    else:
        print("\n    [INFO] Model can cai thien them, thu LSTM/Neural Network")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
