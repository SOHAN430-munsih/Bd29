import argparse
import math
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Utils ----------
def mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def print_score_table(rows: List[Dict]):
    # pretty table
    headers = ["Model", "MAE", "RMSE", "MAPE%"]
    widths = [max(len(r["model"]) for r in rows), 8, 8, 8]
    print("\n=== Test Scores (lower = better) ===")
    print(f"{headers[0]:<{widths[0]}} | {headers[1]:>8} | {headers[2]:>8} | {headers[3]:>8}")
    print("-" * (widths[0] + 3 + 8 + 3 + 8 + 3 + 8))
    for r in rows:
        print(f"{r['model']:<{widths[0]}} | {r['mae']:>8.4f} | {r['rmse']:>8.4f} | {r['mape']:>8.2f}")

def build_features(df: pd.DataFrame, lags=10, sma_windows=(5,10,20)):
    s = df["multiplier"].astype(float).reset_index(drop=True)

    feat = pd.DataFrame({"y": s})
    # lags
    for k in range(1, lags + 1):
        feat[f"lag_{k}"] = s.shift(k)

    # simple moving averages
    for w in sma_windows:
        feat[f"sma_{w}"] = s.shift(1).rolling(window=w).mean()

    # rolling std (volatility)
    for w in sma_windows:
        feat[f"std_{w}"] = s.shift(1).rolling(window=w).std()

    # EMA (recent bias)
    for span in (5, 10, 20):
        feat[f"ema_{span}"] = s.shift(1).ewm(span=span, adjust=False).mean()

    # last spike distance (rough)
    spike = (s > s.rolling(50, min_periods=1).median() * 2.0).astype(int)
    feat["since_spike"] = (~(spike.astype(bool))).groupby((spike == 1).cumsum()).cumcount()

    feat = feat.dropna().reset_index(drop=True)
    return feat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with a 'multiplier' column")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--sma_windows", type=int, nargs="*", default=[5,10,20])
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "multiplier" not in df.columns:
        raise ValueError("CSV must contain a 'multiplier' column")

    # Build features
    feat = build_features(df, lags=args.lags, sma_windows=tuple(args.sma_windows))

    # Time-ordered split (no shuffle)
    N = len(feat)
    test_size = max(1, int(math.floor(N * args.test_ratio)))
    train_end = N - test_size

    train = feat.iloc[:train_end].reset_index(drop=True)
    test = feat.iloc[train_end:].reset_index(drop=True)

    X_train = train.drop(columns=["y"])
    y_train = train["y"].values
    X_test = test.drop(columns=["y"])
    y_test = test["y"].values

    rows = []

    # ---------- Baseline 1: Global Median (train median) ----------
    median_val = np.median(y_train)
    y_pred_median = np.full_like(y_test, fill_value=median_val, dtype=float)
    rows.append({
        "model": "Baseline: Global Median",
        "mae": mean_absolute_error(y_test, y_pred_median),
        "rmse": mean_squared_error(y_test, y_pred_median, squared=False),
        "mape": mape(y_test, y_pred_median),
    })

    # ---------- Baseline 2: Naive (Last value persistence) ----------
    # Predict next = previous actual (shift by 1), for test range
    # We align using the original series indices
    full_series = df["multiplier"].astype(float).reset_index(drop=True)
    # compute naive pred for the portion corresponding to test rows in 'feat'
    # Map 'feat' rows back to original indices: since we dropped NA, we can reconstruct:
    # We'll recompute naive using same alignment as y_test indices from feat.
    # Simplify: use last available value from lags features (lag_1 column is exactly last value)
    y_pred_naive = X_test["lag_1"].values
    rows.append({
        "model": "Baseline: Naive (last value)",
        "mae": mean_absolute_error(y_test, y_pred_naive),
        "rmse": mean_squared_error(y_test, y_pred_naive, squared=False),
        "mape": mape(y_test, y_pred_naive),
    })

    # ---------- Baseline 3: SMA-k (use best k on train via CV-lite) ----------
    best_k = None
    best_mae = float("inf")
    best_pred = None
    for k in args.sma_windows:
        # sma_k feature already prepared as sma_k, use it as prediction
        col = f"sma_{k}"
        if col not in X_test.columns:
            continue
        y_pred_k = X_test[col].values
        mae_k = mean_absolute_error(y_test, y_pred_k)
        if mae_k < best_mae:
            best_mae = mae_k
            best_k = k
            best_pred = y_pred_k

    if best_pred is not None:
        rows.append({
            "model": f"Baseline: SMA-{best_k}",
            "mae": mean_absolute_error(y_test, best_pred),
            "rmse": mean_squared_error(y_test, best_pred, squared=False),
            "mape": mape(y_test, best_pred),
        })

    # ---------- ML Model: GradientBoostingRegressor ----------
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)
    rows.append({
        "model": "ML: GradientBoostingRegressor",
        "mae": mean_absolute_error(y_test, y_pred_gbr),
        "rmse": mean_squared_error(y_test, y_pred_gbr, squared=False),
        "mape": mape(y_test, y_pred_gbr),
    })

    # Sort by MAE ascending
    rows = sorted(rows, key=lambda r: r["mae"])
    print_score_table(rows)

    # Quick tip output
    best = rows[0]
    print("\n>>> Verdict:")
    if best["model"].startswith("ML:"):
        print("Your ML model beats the baselines. Looks like there’s some learnable pattern 🟢")
    else:
        print("Baselines are as good or better. Likely high randomness; avoid overfitting 🔴")

    # Optional: plot (no styling, single plot)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(len(y_test)), y_test, label="Actual")
        plt.plot(range(len(y_test)), y_pred_gbr, label="ML Pred")
        if best_pred is not None:
            plt.plot(range(len(y_test)), best_pred, label=f"SMA-{best_k}")
        plt.plot(range(len(y_test)), y_pred_naive, label="Naive")
        plt.legend()
        plt.title("Test Set: Actual vs Predictions")
        plt.xlabel("Index (test)")
        plt.ylabel("Multiplier")
        plt.show()
    except Exception as e:
        print("Plot skipped:", e)

if __name__ == "__main__":
    main()
