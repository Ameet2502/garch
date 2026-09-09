import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v2")
arr = np.load(os.path.join(RESULTS_DIR, "arrays.npz"), allow_pickle=True)

# --- Figure 1: OOS volatility forecasts vs realized -------------------------
dates = pd.to_datetime(arr["oos_target_dates"])
y_true = arr["oos_y_true"]
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, y_true, color="black", lw=1.3, label="Realized volatility (actual)")
ax.plot(dates, arr["oos_riskmetrics"], lw=1.0, alpha=0.8, label="RiskMetrics EWMA")
ax.plot(dates, arr["oos_garch"], lw=1.0, alpha=0.8, label="Classical GARCH(1,1)")
ax.plot(dates, arr["oos_har"], lw=1.2, label="Time-Series Aggregated GARCH-GRU (HAR)")
ax.plot(dates, arr["oos_multitask"], lw=1.2, label="Regime-Aware Multi-Task GARCH-GRU")
ax.set_title("S&P 500 Out-of-Sample (2025-2026) Volatility Forecasts vs Realized")
ax.set_ylabel("Volatility (percentage points)")
ax.set_xlabel("Date")
ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "figure_oos_forecasts.png"), dpi=150)
print("saved figure_oos_forecasts.png")

# --- Figure 2: causal HMM regime probabilities + change points -------------
full_dates = pd.to_datetime(arr["dates"])
hmm_calm, hmm_neutral, hmm_stress = arr["hmm_calm"], arr["hmm_neutral"], arr["hmm_stress"]

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
ax.stackplot(full_dates, hmm_calm, hmm_neutral, hmm_stress,
             colors=["#2ca02c", "#ff7f0e", "#d62728"], alpha=0.75,
             labels=["Bullish/Calm", "Neutral/Transition", "Bearish/Stress"])
ax.set_ylim(0, 1)
ax.set_ylabel("Causal (forward-filtered)\nregime probability")
ax.set_title("3-State Gaussian HMM: Live Regime Probabilities & Detected Change Points (S&P 500, 2013-2026)")
ax.legend(loc="upper left", fontsize=8, ncol=3)
ax.axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2024-06-30"), color="gray", alpha=0.05)
ax.axvspan(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"), color="blue", alpha=0.05)
ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"), color="green", alpha=0.05)

ax2 = axes[1]
returns_placeholder_dates = full_dates
icss_dates = full_dates[arr["icss_idx"]]
pelt_dates = full_dates[arr["pelt_idx"]]
zshock_dates = full_dates[arr["zshock_idx"]]
ax2.eventplot([icss_dates], lineoffsets=2, colors="tab:blue", label="ICSS breaks")
ax2.eventplot([pelt_dates], lineoffsets=1, colors="tab:purple", label="PELT breaks")
ax2.eventplot([zshock_dates], lineoffsets=0, colors="tab:red", label="z<=-3 shocks")
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(["z<=-3 shock", "PELT", "ICSS"])
ax2.set_xlabel("Date")
ax2.set_title(f"Change-point detections: ICSS={len(arr['icss_idx'])}  PELT={len(arr['pelt_idx'])}  "
              f"z<=-3 shocks={len(arr['zshock_idx'])}", fontsize=9)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "figure_regimes_changepoints.png"), dpi=150)
print("saved figure_regimes_changepoints.png")
