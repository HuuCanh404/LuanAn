"""
system-forecast-do-an
Dự đoán mức sử dụng CPU dựa trên hành vi người dùng.

Mô hình đang dùng:
  - Baseline : LinearRegression (R² ≈ 0.20)
  - Tốt nhất : GradientBoostingRegressor (regression) +
               RandomForestClassifier (phân loại xu hướng Tăng/Ổn định/Giảm)

Chạy thử nghiệm trong thư mục playground/:
  python playground/test_accuracy.py          # baseline LinearRegression
  python playground/test_neural_network.py    # MLPRegressor
  python playground/test_comprehensive.py     # GradientBoosting + RandomForest
"""


def main():
    print("system-forecast-do-an")
    print("Models:")
    print("  Baseline : LinearRegression")
    print("  Best     : GradientBoostingRegressor (regression)")
    print("           + RandomForestClassifier (trend: Tăng/Ổn định/Giảm)")
    print()
    print("Run experiments in playground/:")
    print("  python playground/test_accuracy.py")
    print("  python playground/test_neural_network.py")
    print("  python playground/test_comprehensive.py")


if __name__ == "__main__":
    main()
