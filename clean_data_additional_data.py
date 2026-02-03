from raw_data import *
np.random.seed(42)
n_users = 500  # change if you want
df["user_id"] = np.random.randint(1, n_users + 1, size=len(df))
start = pd.Timestamp("2025-01-01")
end = pd.Timestamp("2025-03-01")
df["timestamp"] = pd.to_datetime(
    np.random.randint(start.value//10**9, end.value//10**9, size=len(df)),
    unit="s"
)
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
df = df.set_index("timestamp")
df = df.dropna().reset_index(drop=True)
df = df.rename(columns={'network_packet_size': 'data_volume'})

df_ctx = df.copy(deep=True)               # df remains unchanged
df_ctx["ts"] = pd.to_datetime(df_ctx.index)
df_ctx = df_ctx.sort_values(["user_id", "ts"]).reset_index(drop=True)
windows = ["1D", "7D"]
numeric_cols = [
    "data_volume",
    "login_attempts",
    "session_duration",
    "failed_logins",
    "ip_reputation_score"
]

def add_time_rolling_features(group, windows, numeric_cols):
    group = group.sort_values("ts").copy()

    for w in windows:
        # events count in past window
        tmp = group[["ts", "failed_logins"]].copy()
        tmp["past"] = tmp["failed_logins"].shift(1)
        group[f"events_count_{w}"] = tmp.rolling(w, on="ts")["past"].count().to_numpy()

        # rolling stats for numeric columns
        for col in numeric_cols:
            tmp = group[["ts", col]].copy()
            tmp["past"] = tmp[col].shift(1)

            r = tmp.rolling(w, on="ts")["past"]
            group[f"{col}_mean_{w}"] = r.mean().to_numpy()
            group[f"{col}_max_{w}"]  = r.max().to_numpy()
            group[f"{col}_sum_{w}"]  = r.sum().to_numpy()

    return group

df_ctx = (
    df_ctx.groupby("user_id", group_keys=False)
          .apply(add_time_rolling_features, windows=windows, numeric_cols=numeric_cols)
)
eps = 1e-6
for col in ["login_attempts", "failed_logins", "session_duration", "ip_reputation_score", "data_volume"]:
    df_ctx[f"{col}_minus_mean_7D"] = df_ctx[col] - df_ctx[f"{col}_mean_7D"]
    df_ctx[f"{col}_vs_mean_7D"] = df_ctx[col] / (df_ctx[f"{col}_mean_7D"] + eps)
prev_browser = df_ctx.groupby("user_id")["browser_type"].shift(1)
df_ctx["browser_change_flag"] = ((prev_browser.notna()) & (df_ctx["browser_type"] != prev_browser)).astype(int)
df_ctx["browser_unknown_flag"] = (df_ctx["browser_type"].astype(str).str.lower() == "unknown").astype(int)
df_ctx = df_ctx.fillna(0)
df_ctx.head()
