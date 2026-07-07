"""
load_balancer_simulator.py
---------------------------
Simulates a stream of incoming client requests hitting a pool of 5
servers and compares two routing strategies:

  1. ROUND ROBIN         - classic baseline, requests distributed in
                            fixed rotation regardless of server health.
  2. AI-BASED (this project) - each request is routed using the trained
                            regressor's predicted response time; the
                            server predicted to be fastest right now
                            gets the request, and its live load state
                            is updated afterward (feedback loop).

At the end it prints a summary table and saves a comparison graph to
outputs/graphs/routing_comparison.png plus a full request log CSV to
outputs/logs/routing_simulation_log.csv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

MODEL_DIR = "models"
GRAPH_DIR = "outputs/graphs"
LOG_DIR = "outputs/logs"

FEATURE_COLS = [
    "cpu_usage_pct", "memory_usage_pct", "active_connections", "requests_per_sec",
    "network_latency_ms", "disk_io_mb_s", "bandwidth_mbps", "error_rate_pct",
]

NUM_REQUESTS = 500
SERVER_IDS = ["srv-1", "srv-2", "srv-3", "srv-4", "srv-5"]
SERVER_CAPACITY = {"srv-1": 1.00, "srv-2": 0.85, "srv-3": 1.15, "srv-4": 0.70, "srv-5": 1.05}
SERVER_BASE_LATENCY = {"srv-1": 12, "srv-2": 18, "srv-3": 9, "srv-4": 25, "srv-5": 11}

rng = np.random.default_rng(7)


def init_server_state():
    """Each server starts with light, slightly randomized baseline load."""
    state = {}
    for sid in SERVER_IDS:
        state[sid] = {
            "active_connections": rng.integers(50, 150),
            "cpu_usage_pct": rng.uniform(20, 40),
            "memory_usage_pct": rng.uniform(20, 40),
            "requests_per_sec": rng.uniform(50, 120),
            "network_latency_ms": SERVER_BASE_LATENCY[sid] + rng.uniform(0, 3),
            "disk_io_mb_s": rng.uniform(2, 8),
            "bandwidth_mbps": rng.uniform(30, 60),
            "error_rate_pct": rng.uniform(0, 0.2),
        }
    return state


def apply_request_load(state, sid, capacity):
    """Simulate the effect of routing ONE new request to server `sid`:
    connections/CPU/etc rise, and response time is computed from a
    queueing-style formula tied to current saturation."""
    s = state[sid]
    s["active_connections"] += 1
    s["requests_per_sec"] += rng.uniform(0.5, 1.5)
    s["cpu_usage_pct"] = min(99.5, s["cpu_usage_pct"] + rng.uniform(0.15, 0.5) / capacity)
    s["memory_usage_pct"] = min(98, s["memory_usage_pct"] + rng.uniform(0.1, 0.35) / capacity)
    s["disk_io_mb_s"] += rng.uniform(0.02, 0.08)
    s["bandwidth_mbps"] += rng.uniform(0.3, 0.9)

    saturation = max(0.0, (s["cpu_usage_pct"] - 60) / 40)
    s["network_latency_ms"] = SERVER_BASE_LATENCY[sid] + 40 * saturation ** 2 + rng.exponential(1.5)
    s["error_rate_pct"] = min(100, (saturation ** 2) * rng.uniform(0, 5))

    actual_response_time = (
        20 + 1.6 * s["cpu_usage_pct"] + 0.9 * s["memory_usage_pct"]
        + 2.2 * s["network_latency_ms"] + 55 * saturation ** 3
        + rng.gamma(2.0, 3.0)
    )

    # small natural recovery each tick (background request completion)
    s["active_connections"] = max(10, s["active_connections"] - rng.uniform(0, 0.6))
    s["cpu_usage_pct"] = max(15, s["cpu_usage_pct"] - rng.uniform(0, 0.4))
    s["memory_usage_pct"] = max(15, s["memory_usage_pct"] - rng.uniform(0, 0.3))

    return actual_response_time


def state_to_feature_row(state, sid):
    s = state[sid]
    return [s[c] for c in FEATURE_COLS]


def run_round_robin(clf, scaler, le, reg):
    state = init_server_state()
    idx = 0
    logs = []
    for i in range(NUM_REQUESTS):
        sid = SERVER_IDS[idx % len(SERVER_IDS)]
        idx += 1
        rt = apply_request_load(state, sid, SERVER_CAPACITY[sid])
        logs.append({"request_id": i, "strategy": "RoundRobin", "server_id": sid, "response_time_ms": rt})
    return pd.DataFrame(logs)


def run_ai_based(clf, scaler, le, reg):
    state = init_server_state()
    logs = []
    for i in range(NUM_REQUESTS):
        X = np.array([state_to_feature_row(state, sid) for sid in SERVER_IDS])
        X_scaled = scaler.transform(X)
        load_preds = le.inverse_transform(clf.predict(X_scaled))
        rt_preds = reg.predict(X)

        load_rank_map = {"Low": 0, "Medium": 1, "High": 2}
        scores = [(load_rank_map[load_preds[j]], rt_preds[j], j) for j in range(len(SERVER_IDS))]
        scores.sort(key=lambda t: (t[0], t[1]))
        best_j = scores[0][2]
        sid = SERVER_IDS[best_j]

        rt = apply_request_load(state, sid, SERVER_CAPACITY[sid])
        logs.append({"request_id": i, "strategy": "AI-Based", "server_id": sid, "response_time_ms": rt})
    return pd.DataFrame(logs)


def main():
    clf = joblib.load(f"{MODEL_DIR}/load_classifier.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/feature_scaler.joblib")
    le = joblib.load(f"{MODEL_DIR}/label_encoder.joblib")
    reg = joblib.load(f"{MODEL_DIR}/response_time_regressor.joblib")

    rr_df = run_round_robin(clf, scaler, le, reg)
    ai_df = run_ai_based(clf, scaler, le, reg)
    full_log = pd.concat([rr_df, ai_df], ignore_index=True)
    full_log.to_csv(f"{LOG_DIR}/routing_simulation_log.csv", index=False)

    summary = full_log.groupby("strategy")["response_time_ms"].agg(
        ["mean", "median", "std", "max", "min"]
    ).round(2)
    p95 = full_log.groupby("strategy")["response_time_ms"].quantile(0.95).round(2)
    summary["p95"] = p95

    print("=" * 70)
    print(f"ROUTING STRATEGY COMPARISON  ({NUM_REQUESTS} requests each)")
    print("=" * 70)
    print(summary.to_string())

    improvement = (
        (summary.loc["RoundRobin", "mean"] - summary.loc["AI-Based", "mean"])
        / summary.loc["RoundRobin", "mean"] * 100
    )
    print("-" * 70)
    print(f">>> AI-Based routing reduces AVERAGE response time by {improvement:.1f}% "
          f"vs Round Robin")

    # distribution per server (how balanced was the load?)
    dist = full_log.groupby(["strategy", "server_id"]).size().unstack(fill_value=0)
    print("\nRequests routed per server:")
    print(dist.to_string())

    # ---- Graph: response time comparison (rolling average) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for strategy, color in [("RoundRobin", "#E57373"), ("AI-Based", "#4CAF50")]:
        sub = full_log[full_log["strategy"] == strategy].reset_index(drop=True)
        rolling = sub["response_time_ms"].rolling(20, min_periods=1).mean()
        axes[0].plot(sub.index, rolling, label=strategy, color=color, linewidth=2)
    axes[0].set_title("Response Time Over Time (20-request rolling avg)")
    axes[0].set_xlabel("Request #")
    axes[0].set_ylabel("Response Time (ms)")
    axes[0].legend()

    sns.boxplot(data=full_log, x="strategy", y="response_time_ms",
                palette={"RoundRobin": "#E57373", "AI-Based": "#4CAF50"}, ax=axes[1])
    axes[1].set_title("Response Time Distribution by Strategy")
    axes[1].set_ylabel("Response Time (ms)")

    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/routing_comparison.png", dpi=150)
    plt.close(fig)

    # ---- Graph: load distribution per server ----
    fig, ax = plt.subplots(figsize=(8, 5))
    dist.T.plot(kind="bar", ax=ax, color=["#E57373", "#4CAF50"])
    ax.set_title("Requests Routed per Server: Round Robin vs AI-Based")
    ax.set_ylabel("Number of Requests")
    ax.set_xlabel("Server")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/requests_per_server.png", dpi=150)
    plt.close(fig)

    with open(f"{LOG_DIR}/simulation_summary.txt", "w") as f:
        f.write(f"ROUTING STRATEGY COMPARISON ({NUM_REQUESTS} requests each)\n\n")
        f.write(summary.to_string())
        f.write(f"\n\nAI-Based routing reduces AVERAGE response time by {improvement:.1f}% vs Round Robin\n\n")
        f.write("Requests routed per server:\n")
        f.write(dist.to_string())


if __name__ == "__main__":
    main()
