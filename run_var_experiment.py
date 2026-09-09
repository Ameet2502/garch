"""VaR backtest (paper Sec. 5 / Table 4 / Figure 3 equivalent) on S&P 500,
one-day-ahead forecasts, alpha=0.05 (95% confidence)."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.data_utils import build_dataset, make_windows, split_windows
from src.train import train_one, eval_classical_garch, DEFAULT_HP, prepare_windows_for_model
from src.var import historical_quantile, backtest

ALPHA = 0.05
WINDOW = 22
HORIZON = 1
SEED = 0

MODELS = ["eh_GARCH_GRU", "eh_GARCH_LSTM", "bl_GARCH_LSTM", "pl_GARCH_GRU", "pl_GARCH_LSTM"]
CLASSICAL = [("GARCH(1,1)", False), ("GJR-GARCH", True)]


def main():
    tuned = json.load(open("results/tuned_hp.json")) if os.path.exists("results/tuned_hp.json") else {}
    ds = build_dataset("data/SP500_raw.csv", "S&P500", date_col="Date", close_col="Close")

    date_to_idx = {d: i for i, d in enumerate(ds.dates)}

    # historical (pre-test) standardized-residual quantile, shared across models
    n_test = 252
    pre_eps = ds.eps[: len(ds.eps) - n_test]
    pre_rv = ds.realized_vol.values[: len(ds.eps) - n_test]
    pre_rv = pre_rv[-len(pre_eps):] if len(pre_rv) >= len(pre_eps) else pre_rv
    # align lengths defensively
    m = min(len(pre_eps), len(pre_rv))
    q_alpha = historical_quantile(pre_eps[-m:], pre_rv[-m:], ALPHA)
    print("q_alpha:", q_alpha)

    results = {}
    preds_for_plot = {}

    for model_name in MODELS:
        hp = dict(DEFAULT_HP)
        if model_name in {"eh_GARCH_GRU", "eh_GARCH_LSTM", "bl_GARCH_LSTM"} and "S&P500" in tuned:
            hp.update({k: v for k, v in tuned["S&P500"].items() if k in hp})
        res = train_one(model_name, ds, WINDOW, HORIZON, hp, seed=SEED)
        sigma_hat = res["test_pred"]
        tgt_dates = res["test_dates"]
        actual_r = np.array([ds.returns[date_to_idx[d]] for d in tgt_dates])
        vr = backtest(model_name, ds.mu, q_alpha, sigma_hat, actual_r, tgt_dates, alpha=ALPHA)
        results[model_name] = vr
        preds_for_plot[model_name] = vr
        print(f"{model_name:15s} violations={vr.n_violations:3d}/{vr.n_forecasts} "
              f"ratio={vr.violation_ratio:.4f}")

    for mname, gjr in CLASSICAL:
        res = eval_classical_garch(ds, horizon=HORIZON, gjr=gjr, window=WINDOW)
        sigma_hat = res["test_pred"]
        tgt_dates = res["test_dates"]
        actual_r = np.array([ds.returns[date_to_idx[d]] for d in tgt_dates])
        vr = backtest(mname, ds.mu, q_alpha, sigma_hat, actual_r, tgt_dates, alpha=ALPHA)
        results[mname] = vr
        preds_for_plot[mname] = vr
        print(f"{mname:15s} violations={vr.n_violations:3d}/{vr.n_forecasts} "
              f"ratio={vr.violation_ratio:.4f}")

    # ---- Table 4 equivalent ----
    rows = []
    for name, vr in results.items():
        rows.append(dict(model=name, violation_numbers=vr.n_violations,
                          violation_ratio=f"{vr.violation_ratio * 100:.2f}%",
                          nominal_alpha=f"{ALPHA * 100:.2f}%"))
    table4 = pd.DataFrame(rows)
    table4.to_csv("results/table4_var_violations.csv", index=False)
    print(table4.to_string(index=False))

    # ---- Figure 3 equivalent ----
    any_vr = next(iter(results.values()))
    dates = any_vr.dates
    actual = any_vr.returns

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, actual, color="lightgray", lw=1.0, label="Actual returns", zorder=1)
    colors = plt.cm.tab10.colors
    for i, (name, vr) in enumerate(results.items()):
        ax.plot(dates, -vr.var_series, lw=1.2, label=f"{name} VaR forecast", color=colors[i % 10])
        viol_idx = np.where(vr.violations)[0]
        if len(viol_idx) > 0:
            ax.scatter(np.array(dates)[viol_idx], actual[viol_idx], marker="x",
                       color=colors[i % 10], s=40, zorder=5)
    ax.set_title(f"Actual Returns, VaR Forecasts, and Violations at {(1-ALPHA)*100:.0f}% "
                 f"Confidence Level on S&P 500")
    ax.set_ylabel("Return")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("results/plots/figure3_var.png", dpi=150)
    print("Saved plot to results/plots/figure3_var.png")


if __name__ == "__main__":
    main()
