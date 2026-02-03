from raw_data import *
from clean_data_additional_data import *

import matplotlib.pyplot as plt

# ----------------------------
# 0) Quick sanity checks
# ----------------------------
required_cols = [
    "attack_detected",
    "failed_logins", "failed_logins_vs_mean_7D",
    "session_duration", "session_duration_vs_mean_7D",
    "ip_reputation_score_vs_mean_7D",
    "data_volume_vs_mean_7D",
    "browser_change_flag"
]

missing = [c for c in required_cols if c not in df_ctx.columns]
if missing:
    print("Missing columns:", missing)
else:
    print("All required columns are present ✅")

# ----------------------------
# 1) Boxplot: Failed logins vs baseline
# ----------------------------
df_ctx.boxplot(column="failed_logins_vs_mean_7D", by="attack_detected", grid=False)
plt.title("Failed Logins Relative to 7-Day Baseline")
plt.suptitle("")
plt.xlabel("Attack Detected (0 = Normal, 1 = Attack)")
plt.ylabel("Failed Logins / 7-Day Mean")
plt.show()

# ----------------------------
# 2) Bar charts: raw vs contextual failed logins
# ----------------------------
fig, ax = plt.subplots()
df_ctx.groupby("attack_detected")["failed_logins"].mean().plot(kind="bar", ax=ax)
ax.set_title("Average Failed Logins (Raw)")
ax.set_xlabel("Attack Detected")
ax.set_ylabel("Failed Logins")
plt.show()

fig, ax = plt.subplots()
df_ctx.groupby("attack_detected")["failed_logins_vs_mean_7D"].mean().plot(kind="bar", ax=ax)
ax.set_title("Average Failed Logins (Context-Aware)")
ax.set_xlabel("Attack Detected")
ax.set_ylabel("Failed Logins vs 7-Day Mean")
plt.show()

# ----------------------------
# 3) Boxplot: Session duration vs baseline
# ----------------------------
df_ctx.boxplot(column="session_duration_vs_mean_7D", by="attack_detected", grid=False)
plt.title("Session Duration Relative to User Baseline")
plt.suptitle("")
plt.xlabel("Attack Detected (0 = Normal, 1 = Attack)")
plt.ylabel("Session Duration / 7-Day Mean")
plt.show()

# ----------------------------
# 4) Boxplot: IP reputation vs baseline
# ----------------------------
df_ctx.boxplot(column="ip_reputation_score_vs_mean_7D", by="attack_detected", grid=False)
plt.title("IP Reputation Drift During Attacks")
plt.suptitle("")
plt.xlabel("Attack Detected (0 = Normal, 1 = Attack)")
plt.ylabel("IP Reputation vs 7-Day Mean")
plt.show()

# ----------------------------
# 5) Bar chart: Browser change rate
# ----------------------------
browser_change_rate = df_ctx.groupby("attack_detected")["browser_change_flag"].mean()
browser_change_rate.plot(kind="bar")
plt.title("Browser Change Frequency")
plt.xlabel("Attack Detected")
plt.ylabel("Change Probability")
plt.show()

# ----------------------------
# 6) Boxplot: Data volume vs baseline
# ----------------------------
df_ctx.boxplot(column="data_volume_vs_mean_7D", by="attack_detected", grid=False)
plt.title("Data Volume Relative to User Baseline")
plt.suptitle("")
plt.xlabel("Attack Detected (0 = Normal, 1 = Attack)")
plt.ylabel("Data Volume vs 7-Day Mean")
plt.show()

# ----------------------------
# 7) Correlation heatmap (context features)
# ----------------------------
corr_cols = [
    "attack_detected",
    "failed_logins_vs_mean_7D",
    "session_duration_vs_mean_7D",
    "ip_reputation_score_vs_mean_7D",
    "data_volume_vs_mean_7D",
    "browser_change_flag"
]

corr = df_ctx[corr_cols].corr()

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Correlation Between Context Features and Attacks")
plt.show()

# ----------------------------
# 8) Summary bar chart: context features (project headline figure)
# ----------------------------
df_ctx.groupby("attack_detected")[[
    "failed_logins_vs_mean_7D",
    "session_duration_vs_mean_7D",
    "data_volume_vs_mean_7D"
]].mean().plot(kind="bar")

plt.title("Contextual Feature Comparison: Attack vs Normal")
plt.xlabel("Attack Detected")
plt.ylabel("Average Deviation from Baseline")
plt.show()