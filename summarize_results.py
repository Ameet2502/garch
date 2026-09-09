"""Aggregate results/raw_results.csv into a Table-3-style summary:
mean (std) of MSE/MAE across seeds, for every (dataset, horizon, model)."""
import pandas as pd

df = pd.read_csv("results/raw_results.csv")

MODEL_ORDER = ["eh_GARCH_GRU", "eh_GARCH_LSTM", "bl_GARCH_LSTM", "pl_GARCH_GRU",
               "pl_GARCH_LSTM", "GRU", "LSTM", "Transformer", "GARCH(1,1)", "GJR-GARCH"]
HORIZON_LABEL = {1: "1D", 3: "3D", 7: "1W"}

rows = []
for (dataset, model, horizon), g in df.groupby(["dataset", "model", "horizon"]):
    rows.append(dict(
        dataset=dataset, model=model, horizon=HORIZON_LABEL.get(horizon, horizon),
        n_seeds=len(g),
        mse_mean=g["mse"].mean(), mse_std=g["mse"].std(ddof=0) if len(g) > 1 else 0.0,
        mae_mean=g["mae"].mean(), mae_std=g["mae"].std(ddof=0) if len(g) > 1 else 0.0,
    ))
summary = pd.DataFrame(rows)
summary["model_order"] = summary["model"].apply(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
summary = summary.sort_values(["dataset", "horizon", "model_order"]).drop(columns="model_order")
summary.to_csv("results/table3_summary.csv", index=False)

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
print(summary.to_string(index=False))
