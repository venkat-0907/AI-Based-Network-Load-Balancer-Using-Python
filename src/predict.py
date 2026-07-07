"""
predict.py
----------
Loads the trained models and predicts, for one or more servers:
  - load_status         (Low / Medium / High)   [classification]
  - predicted_response_time_ms                  [regression]
  - a recommendation of which server should receive the next request

Usage examples
--------------
1) Predict for a single server passed as CLI flags:

   python3 src/predict.py --single \
       --cpu 72 --mem 65 --conn 500 --rps 300 \
       --latency 20 --disk 15 --bandwidth 150 --error 0.5

2) Predict for a batch of servers from a CSV (must contain the 8 feature
   columns used in training) and get a routing recommendation:

   python3 src/predict.py --batch data/sample_live_servers.csv

3) No arguments -> runs a demo using data/sample_live_servers.csv
   (auto-generated if missing) and prints a full routing decision.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "models"
FEATURE_COLS = [
    "cpu_usage_pct",
    "memory_usage_pct",
    "active_connections",
    "requests_per_sec",
    "network_latency_ms",
    "disk_io_mb_s",
    "bandwidth_mbps",
    "error_rate_pct",
]

SAMPLE_LIVE_PATH = "data/sample_live_servers.csv"


def load_models():
    clf = joblib.load(f"{MODEL_DIR}/load_classifier.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/feature_scaler.joblib")
    le = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
    reg = joblib.load(f"{MODEL_DIR}/response_time_regressor.joblib")
    return clf, scaler, le, reg


def predict_dataframe(df: pd.DataFrame, clf, scaler, le, reg) -> pd.DataFrame:
    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    class_pred = le.inverse_transform(clf.predict(X_scaled))
    class_proba = clf.predict_proba(X_scaled)
    confidence = class_proba.max(axis=1)

    response_time_pred = reg.predict(X)

    result = df.copy()
    result["predicted_load_status"] = class_pred
    result["prediction_confidence"] = confidence.round(3)
    result["predicted_response_time_ms"] = response_time_pred.round(2)
    return result


def make_routing_decision(result: pd.DataFrame) -> pd.DataFrame:
    """Rank servers: prefer Low/Medium load over High, then lowest predicted
    response time. Returns dataframe sorted best-first with a 'rank' col."""
    load_rank_map = {"Low": 0, "Medium": 1, "High": 2}
    result = result.copy()
    result["_load_rank"] = result["predicted_load_status"].map(load_rank_map)
    result = result.sort_values(
        by=["_load_rank", "predicted_response_time_ms"]
    ).reset_index(drop=True)
    result["rank"] = result.index + 1
    result = result.drop(columns=["_load_rank"])
    return result


def ensure_sample_file():
    if os.path.exists(SAMPLE_LIVE_PATH):
        return
    # Snapshot of 5 servers with varied live conditions (demo purposes)
    demo = pd.DataFrame([
        {"server_id": "srv-1", "cpu_usage_pct": 55, "memory_usage_pct": 48, "active_connections": 420,
         "requests_per_sec": 260, "network_latency_ms": 14, "disk_io_mb_s": 10, "bandwidth_mbps": 120, "error_rate_pct": 0.2},
        {"server_id": "srv-2", "cpu_usage_pct": 88, "memory_usage_pct": 82, "active_connections": 610,
         "requests_per_sec": 340, "network_latency_ms": 38, "disk_io_mb_s": 18, "bandwidth_mbps": 180, "error_rate_pct": 2.1},
        {"server_id": "srv-3", "cpu_usage_pct": 40, "memory_usage_pct": 35, "active_connections": 300,
         "requests_per_sec": 210, "network_latency_ms": 9, "disk_io_mb_s": 8, "bandwidth_mbps": 95, "error_rate_pct": 0.05},
        {"server_id": "srv-4", "cpu_usage_pct": 95, "memory_usage_pct": 90, "active_connections": 700,
         "requests_per_sec": 380, "network_latency_ms": 55, "disk_io_mb_s": 22, "bandwidth_mbps": 200, "error_rate_pct": 4.8},
        {"server_id": "srv-5", "cpu_usage_pct": 62, "memory_usage_pct": 58, "active_connections": 480,
         "requests_per_sec": 275, "network_latency_ms": 16, "disk_io_mb_s": 12, "bandwidth_mbps": 135, "error_rate_pct": 0.4},
    ])
    demo.to_csv(SAMPLE_LIVE_PATH, index=False)


def main():
    parser = argparse.ArgumentParser(description="AI Load Balancer - Prediction CLI")
    parser.add_argument("--single", action="store_true", help="Predict for a single server via CLI flags")
    parser.add_argument("--batch", type=str, help="Path to CSV of live server metrics")
    parser.add_argument("--cpu", type=float, default=50)
    parser.add_argument("--mem", type=float, default=50)
    parser.add_argument("--conn", type=float, default=300)
    parser.add_argument("--rps", type=float, default=200)
    parser.add_argument("--latency", type=float, default=15)
    parser.add_argument("--disk", type=float, default=10)
    parser.add_argument("--bandwidth", type=float, default=100)
    parser.add_argument("--error", type=float, default=0.2)
    args = parser.parse_args()

    clf, scaler, le, reg = load_models()

    if args.single:
        df = pd.DataFrame([{
            "server_id": "input-server",
            "cpu_usage_pct": args.cpu,
            "memory_usage_pct": args.mem,
            "active_connections": args.conn,
            "requests_per_sec": args.rps,
            "network_latency_ms": args.latency,
            "disk_io_mb_s": args.disk,
            "bandwidth_mbps": args.bandwidth,
            "error_rate_pct": args.error,
        }])
        result = predict_dataframe(df, clf, scaler, le, reg)
        row = result.iloc[0]
        print("=" * 60)
        print("SINGLE SERVER PREDICTION")
        print("=" * 60)
        print(f"Predicted load status      : {row['predicted_load_status']}")
        print(f"Prediction confidence       : {row['prediction_confidence']*100:.1f}%")
        print(f"Predicted response time    : {row['predicted_response_time_ms']:.2f} ms")
        return

    batch_path = args.batch if args.batch else SAMPLE_LIVE_PATH
    if not args.batch:
        ensure_sample_file()

    df = pd.read_csv(batch_path)
    result = predict_dataframe(df, clf, scaler, le, reg)
    ranked = make_routing_decision(result)

    print("=" * 90)
    print("AI LOAD BALANCER - LIVE ROUTING DECISION")
    print("=" * 90)
    display_cols = ["rank", "server_id", "cpu_usage_pct", "memory_usage_pct",
                     "predicted_load_status", "prediction_confidence", "predicted_response_time_ms"]
    print(ranked[display_cols].to_string(index=False))
    best = ranked.iloc[0]
    print("-" * 90)
    print(f">>> ROUTE NEXT REQUEST TO: {best['server_id']}  "
          f"(predicted load: {best['predicted_load_status']}, "
          f"predicted response time: {best['predicted_response_time_ms']:.2f} ms)")


if __name__ == "__main__":
    main()
