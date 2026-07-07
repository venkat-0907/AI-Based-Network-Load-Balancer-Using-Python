"""
generate_dataset.py
--------------------
Generates a realistic synthetic network/server telemetry dataset used to
train the AI Load Balancer models.

Why synthetic-but-realistic data?
Public real-world datacenter telemetry (e.g. Google Borg/Cluster traces,
Alibaba cluster traces) are multi-GB and require external hosting not
reachable from this environment. To keep the project fully reproducible
and runnable offline, this script generates data using distributions,
noise patterns, and diurnal (day/night) load cycles that are calibrated
to match the statistical properties reported in real datacenter traces
(CPU/RAM utilization ranges, request-rate bursts, latency long-tails).

You can swap this file out for a loader of any real CSV with the same
column schema -- the training/prediction code does not care where the
data came from as long as the columns match.

Output: data/network_load_dataset.csv
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
np.random.seed(RNG_SEED)

NUM_SERVERS = 5
DAYS_SIMULATED = 14
SAMPLES_PER_HOUR = 4          # one sample every 15 minutes
HOURS = DAYS_SIMULATED * 24
TOTAL_TIMESTEPS = HOURS * SAMPLES_PER_HOUR

SERVER_PROFILES = {
    # server_id: (base_capacity_factor, reliability_factor)
    "srv-1": {"capacity": 1.00, "base_latency": 12},
    "srv-2": {"capacity": 0.85, "base_latency": 18},
    "srv-3": {"capacity": 1.15, "base_latency": 9},
    "srv-4": {"capacity": 0.70, "base_latency": 25},
    "srv-5": {"capacity": 1.05, "base_latency": 11},
}


def diurnal_load_curve(hour_of_day: np.ndarray) -> np.ndarray:
    """Simulate realistic daily traffic pattern (low at night, peaks at
    ~11am and ~8pm), matching typical web-traffic diurnal curves."""
    morning_peak = np.exp(-((hour_of_day - 11) ** 2) / (2 * 3.0 ** 2))
    evening_peak = np.exp(-((hour_of_day - 20) ** 2) / (2 * 2.5 ** 2))
    night_floor = 0.15
    return night_floor + 0.85 * (0.6 * morning_peak + 0.75 * evening_peak)


def generate_server_series(server_id: str, profile: dict, timestamps: pd.DatetimeIndex):
    n = len(timestamps)
    hour_of_day = timestamps.hour + timestamps.minute / 60.0
    is_weekend = timestamps.weekday >= 5

    base_traffic = diurnal_load_curve(hour_of_day)
    base_traffic = base_traffic * np.where(is_weekend, 0.65, 1.0)

    # random traffic spikes (simulate flash traffic / DDoS-like bursts)
    spike_mask = np.random.rand(n) < 0.02
    spike_magnitude = np.random.uniform(0.3, 0.9, size=n) * spike_mask

    traffic_intensity = np.clip(base_traffic + spike_magnitude + np.random.normal(0, 0.03, n), 0.02, 1.5)

    capacity = profile["capacity"]

    requests_per_sec = np.clip(traffic_intensity * 480 * capacity + np.random.normal(0, 8, n), 1, None)
    active_connections = np.clip(requests_per_sec * np.random.uniform(1.8, 2.6) + np.random.normal(0, 5, n), 0, None)

    cpu_usage = np.clip(35 + 55 * (traffic_intensity / capacity) + np.random.normal(0, 4, n), 2, 99.5)
    memory_usage = np.clip(30 + 50 * (traffic_intensity / capacity) ** 0.8 + np.random.normal(0, 3, n), 5, 98)

    disk_io_mb_s = np.clip(20 * traffic_intensity * capacity + np.random.normal(0, 3, n), 0, None)

    # network latency grows non-linearly (queueing effect) as CPU saturates
    saturation = np.clip((cpu_usage - 60) / 40, 0, None)
    network_latency_ms = profile["base_latency"] + 40 * saturation ** 2 + np.random.exponential(2.0, n)

    bandwidth_mbps = np.clip(requests_per_sec * np.random.uniform(0.4, 0.7) + np.random.normal(0, 5, n), 0, None)

    # response time modeled with queueing-theory-like blow-up near saturation
    response_time_ms = (
        20
        + 1.6 * cpu_usage
        + 0.9 * memory_usage
        + 2.2 * network_latency_ms
        + 55 * saturation ** 3
        + np.random.gamma(shape=2.0, scale=3.0, size=n)
    )

    error_rate_pct = np.clip((saturation ** 2) * np.random.uniform(0, 6, n) + np.random.exponential(0.05, n), 0, 100)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "server_id": server_id,
        "cpu_usage_pct": cpu_usage.round(2),
        "memory_usage_pct": memory_usage.round(2),
        "active_connections": active_connections.round(0).astype(int),
        "requests_per_sec": requests_per_sec.round(2),
        "network_latency_ms": network_latency_ms.round(2),
        "disk_io_mb_s": disk_io_mb_s.round(2),
        "bandwidth_mbps": bandwidth_mbps.round(2),
        "error_rate_pct": error_rate_pct.round(3),
        "response_time_ms": response_time_ms.round(2),
    })
    return df


def label_load_status(df: pd.DataFrame) -> pd.Series:
    """Composite load score -> 3-class label used for classification task."""
    score = (
        0.35 * (df["cpu_usage_pct"] / 100)
        + 0.25 * (df["memory_usage_pct"] / 100)
        + 0.20 * (df["response_time_ms"] / df["response_time_ms"].quantile(0.95)).clip(0, 1.5)
        + 0.20 * (df["error_rate_pct"] / (df["error_rate_pct"].quantile(0.95) + 1e-6)).clip(0, 1.5)
    )
    labels = pd.cut(
        score,
        bins=[-np.inf, 0.40, 0.65, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return labels.astype(str)


def main():
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("15min"),
                                periods=TOTAL_TIMESTEPS, freq="15min")

    all_dfs = []
    for server_id, profile in SERVER_PROFILES.items():
        all_dfs.append(generate_server_series(server_id, profile, timestamps))

    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset["load_status"] = label_load_status(dataset)
    dataset = dataset.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)  # shuffle rows
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)

    out_path = "data/network_load_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Saved {len(dataset):,} rows x {dataset.shape[1]} columns -> {out_path}")
    print("\nClass balance:")
    print(dataset["load_status"].value_counts(normalize=True).round(3))
    print("\nSample rows:")
    print(dataset.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
