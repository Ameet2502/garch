"""Main experiment runner: reproduces the structure of the paper's Table 3
(MSE/MAE across models x horizons x datasets, averaged over seeds) plus the
classical GARCH(1,1)/GJR-GARCH benchmarks. Checkpoints every completed run
to results/raw_results.csv so the script is safely re-runnable / resumable
if interrupted.
"""
import os
import sys
import time
import json
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.data_utils import build_dataset
from src.train import train_one, eval_classical_garch, DEFAULT_HP

RESULTS_CSV = "results/raw_results.csv"

DATASETS = {
    "S&P500": dict(path="data/SP500_raw.csv", date_col="Date", close_col="Close"),
    "DJI": dict(path="data/DJI_raw.csv", date_col="Date", close_col="Close"),
}

NEURAL_MODELS = ["eh_GARCH_GRU", "eh_GARCH_LSTM", "bl_GARCH_LSTM",
                  "pl_GARCH_GRU", "pl_GARCH_LSTM", "GRU", "LSTM", "Transformer"]
HORIZONS = [1, 3, 7]
SEEDS = [0, 1, 2]           # reduced from paper's 20 seeds; see README
WINDOW = 22

TUNED_HP_PATH = "results/tuned_hp.json"
HYBRID_MODELS = {"eh_GARCH_GRU", "eh_GARCH_LSTM", "bl_GARCH_LSTM"}


def get_hp(model_name, dataset_name, tuned):
    hp = dict(DEFAULT_HP)
    if model_name in HYBRID_MODELS and dataset_name in tuned:
        hp.update({k: v for k, v in tuned[dataset_name].items() if k in hp})
    return hp


def load_checkpoint():
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=["dataset", "model", "horizon", "seed", "mse", "mae", "n_test", "train_seconds"])


def already_done(df, dataset, model, horizon, seed):
    if len(df) == 0:
        return False
    m = (df["dataset"] == dataset) & (df["model"] == model) & (df["horizon"] == horizon) & (df["seed"] == seed)
    return m.any()


def append_result(row):
    df = load_checkpoint()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)


def main(max_seconds=None):
    tuned = json.load(open(TUNED_HP_PATH)) if os.path.exists(TUNED_HP_PATH) else {}
    datasets_cache = {name: build_dataset(name=name, **cfg) for name, cfg in DATASETS.items()}

    start = time.time()

    # --- classical GARCH benchmarks (deterministic, seed irrelevant) ---
    for dname, ds in datasets_cache.items():
        for horizon in HORIZONS:
            for gjr, mname in [(False, "GARCH(1,1)"), (True, "GJR-GARCH")]:
                df = load_checkpoint()
                if already_done(df, dname, mname, horizon, -1):
                    continue
                t0 = time.time()
                res = eval_classical_garch(ds, horizon=horizon, gjr=gjr, window=WINDOW)
                append_result(dict(dataset=dname, model=mname, horizon=horizon, seed=-1,
                                    mse=res["mse"], mae=res["mae"], n_test=res["n_test"],
                                    train_seconds=time.time() - t0))
                print(f"[classical] {dname} {mname} h={horizon} MSE={res['mse']:.6f} "
                      f"MAE={res['mae']:.6f}", flush=True)

    # --- neural models ---
    for dname, ds in datasets_cache.items():
        for model_name in NEURAL_MODELS:
            hp = get_hp(model_name, dname, tuned)
            for horizon in HORIZONS:
                for seed in SEEDS:
                    df = load_checkpoint()
                    if already_done(df, dname, model_name, horizon, seed):
                        continue
                    if max_seconds and (time.time() - start) > max_seconds:
                        print("Time budget reached, stopping for this call.", flush=True)
                        return
                    t0 = time.time()
                    res = train_one(model_name, ds, WINDOW, horizon, hp, seed=seed)
                    dt = time.time() - t0
                    append_result(dict(dataset=dname, model=model_name, horizon=horizon,
                                        seed=seed, mse=res["mse"], mae=res["mae"],
                                        n_test=res["n_test"], train_seconds=dt))
                    print(f"[{dname}] {model_name:15s} h={horizon} seed={seed} "
                          f"MSE={res['mse']:.6f} MAE={res['mae']:.6f} ({dt:.1f}s)", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_seconds=budget)
