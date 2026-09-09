"""
Expanded evaluation pipeline (v2, revised against the reference deck).

Corrections made after reading the reference deck ("Final_Presentation_
GARCH_GRU", Mphasis/NextLabs internship project) in full:
  - Base model input is the deck's exact 3 "direct sequence features":
    [log_return, cond_vol, std_resid], min-max scaled on TRAIN data only.
    The raw (unscaled) residual eps_t = log_return_t - train_mean drives the
    internal HAR/GARCH recursion on a separate path (dual-path design).
  - HAR equation corrected: gamma_{d,w,m} scale the VARIANCE-history terms,
    theta_{d,w,m} scale the shock (eps^2) terms (slide 25) -- the original
    first draft had these swapped.
  - The final Regime-Aware model stacks 2 HAR-GARCH-GRU layers with dropout
    between them ("Layer 1 -> Dropout -> Layer 2", slide 49), not 1.
  - The regime auxiliary target is the HMM's own SOFT probability vector
    (not a hard argmax label), per "y_regime:[N,3] soft labels" (slide 48).
  - Credit spread now uses BAA10Y (full 1986-2026 history you supplied)
    instead of BAMLH0A0HYM2 (2023-onward only), fixing the gap flagged
    earlier. The 3-state HMM is now fit on 5 macro+volatility features
    (realized vol, VIX, credit spread, term spread, real Fed funds), matching
    the deck's "regimes are built from real macroeconomic signals" (slide 39).
  - VaR backtest now uses the deck's own methodology (src/var2.py):
    VaR_t = mu_train + q_alpha^train * sigma_hat_t, plus Kupiec,
    Christoffersen independence, Binomial tests and Pinball/Lopez loss,
    matching the deck's Part 4 / Part 8 tables directly.

Still-open, documented simplifications relative to the deck: single seed
(not a 5-seed ensemble), no Optuna retuning for the new stages, and the
GMM-vs-HMM per-regime conditional breakdown tables (deck slides 41-45) are
not reproduced here.
"""
from __future__ import annotations
import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from src.data_utils import (load_index_close, compute_log_returns, compute_realized_vol,
                             Dataset, make_dual_path_windows, split_dual_windows_by_date)
from src.garch import fit_garch
from src.riskmetrics import rolling_forecast, LAMBDA_DEFAULT
from src.regime_hmm import fit_hmm_regimes, REGIME_NAMES
from src.changepoint import detect_all
from src.macro_data import build_macro_bundle
from src.har_cell import HARGarchGRUModel
from src.multitask_model import RegimeAwareMultiTaskGARCHGRU, multitask_loss
from src.var2 import historical_quantile_train, backtest as var_backtest

UPLOADS = "/root/.claude/uploads/3bee48eb-2724-5018-a3f3-c9e899063c37"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_START, TRAIN_END = "2014-01-01", "2024-06-30"
VAL_START, VAL_END = "2024-07-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2026-12-31"
WINDOW = 22
HORIZON = 1
SEED = 0
ALPHA = 0.05

HP = dict(hidden_size=16, lr=2e-2, batch_size=128, max_epochs=40, patience=6, dropout=0.1)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_price_dataset() -> Dataset:
    close = load_index_close(os.path.join(DATA_DIR, "SP500_full.csv"),
                              date_col="date", close_col="close",
                              start="2013-01-01", end="2026-12-31")
    returns = compute_log_returns(close)
    mu = float(returns.loc[:TRAIN_END].mean())
    eps = returns - mu
    rv = compute_realized_vol(returns, k=5)
    return Dataset(name="SP500_2013_2026", dates=returns.index, returns=returns.values,
                    eps=eps.values, mu=mu, realized_vol=rv)


def minmax_scale_train_only(raw: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train_vals = raw[train_mask]
    lo, hi = float(np.min(train_vals)), float(np.max(train_vals))
    span = hi - lo if hi > lo else 1.0
    return ((raw - lo) / span).astype(np.float32)


def build_macro_frame_full(ds: Dataset) -> pd.DataFrame:
    paths = {
        "DFF": os.path.join(UPLOADS, "c51b6e80-DFF.csv"),
        "DGS10": os.path.join(UPLOADS, "6f41a934-DGS10.csv"),
        "T10Y3M": os.path.join(UPLOADS, "b22160a5-T10Y3M.csv"),
        "CPIAUCSL": os.path.join(UPLOADS, "3c7c53f6-CPIAUCSL.csv"),
        "VIXCLS": os.path.join(UPLOADS, "4bcdb62c-VIXCLS.csv"),
        "BAA10Y": os.path.join(UPLOADS, "daa592ab-BAA10Y.csv"),   # full history, replaces BAMLH0A0HYM2
    }
    trading_days = pd.DatetimeIndex(ds.dates)
    train_mask = (trading_days >= TRAIN_START) & (trading_days <= TRAIN_END)
    bundle = build_macro_bundle(paths, trading_days, train_mask)
    return bundle.frame


def build_garch_std_channel(ds: Dataset):
    train_mask = (pd.DatetimeIndex(ds.dates) >= TRAIN_START) & (pd.DatetimeIndex(ds.dates) <= TRAIN_END)
    train_eps = ds.eps[train_mask]
    fit = fit_garch(train_eps, gjr=False)
    sigma2_full = fit.forecast_path(ds.eps, start_idx=0)
    return np.sqrt(np.clip(sigma2_full, 1e-12, None)).astype(np.float32), fit


def build_hmm_features(ds: Dataset, macro: pd.DataFrame):
    """3-state Gaussian HMM fit on 5 macro+volatility features (realized
    vol, VIX, BAA10Y credit spread, term spread, real Fed funds), Train
    period only, then applied with a causal forward filter across the full
    series (matches deck slide 39's macro-driven regime construction)."""
    dates = pd.DatetimeIndex(ds.dates)
    rv_full = ds.realized_vol.reindex(dates).ffill().bfill().to_numpy()
    feat = np.stack([
        rv_full,
        macro["vix"].to_numpy(),
        macro["credit_spread"].to_numpy(),
        macro["term_spread"].to_numpy(),
        macro["real_fed_funds"].fillna(0.0).to_numpy(),
    ], axis=-1)
    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    hmm = fit_hmm_regimes(feat[train_mask.to_numpy() if hasattr(train_mask, "to_numpy") else train_mask],
                           n_states=3, random_state=SEED)
    probs = hmm.forward_filter(feat)   # (n, 3), Calm/Neutral/Stress order
    return probs, hmm


def train_single_output(model, train_w, val_w, hp, label=""):
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss()
    Xtr, Etr, ytr = torch.from_numpy(train_w.X), torch.from_numpy(train_w.E), torch.from_numpy(train_w.y)
    ds_tr = torch.utils.data.TensorDataset(Xtr, Etr, ytr)
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=hp["batch_size"], shuffle=True)
    Xval, Eval, yval = torch.from_numpy(val_w.X), torch.from_numpy(val_w.E), torch.from_numpy(val_w.y)

    best_val, best_state, patience_ctr = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(hp["max_epochs"]):
        model.train()
        for xb, eb, yb in loader:
            opt.zero_grad()
            pred = model(xb, eb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval, Eval), yval).item()
        if val_loss < best_val - 1e-7:
            best_val, best_state, patience_ctr = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience_ctr += 1
            if patience_ctr >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    print(f"  [{label}] trained, best_val_mse={best_val:.5f}, stopped_epoch={epoch}")
    return model


def train_multitask(model, train_w, val_w, hp, lambda_aux=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    Xtr, Etr, Rtr, ytr = (torch.from_numpy(train_w.X), torch.from_numpy(train_w.E),
                          torch.from_numpy(train_w.R), torch.from_numpy(train_w.y))
    ds_tr = torch.utils.data.TensorDataset(Xtr, Etr, Rtr, ytr)
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=hp["batch_size"], shuffle=True)
    Xval, Eval, Rval, yval = (torch.from_numpy(val_w.X), torch.from_numpy(val_w.E),
                              torch.from_numpy(val_w.R), torch.from_numpy(val_w.y))

    best_val, best_state, patience_ctr = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(hp["max_epochs"]):
        model.train()
        for xb, eb, rb, yb in loader:
            opt.zero_grad()
            vol_hat, regime_logits = model(xb, eb)
            loss, _ = multitask_loss(vol_hat, yb, regime_logits, rb, lambda_aux=lambda_aux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vol_hat, regime_logits = model(Xval, Eval)
            val_loss, parts = multitask_loss(vol_hat, yval, regime_logits, Rval, lambda_aux=lambda_aux)
        if parts["mse"] < best_val - 1e-7:
            best_val, best_state, patience_ctr = parts["mse"], copy.deepcopy(model.state_dict()), 0
        else:
            patience_ctr += 1
            if patience_ctr >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    print(f"  [multitask] trained, best_val_vol_mse={best_val:.5f}, stopped_epoch={epoch}")
    return model


def eval_regression(preds, y_true):
    mse = float(np.mean((preds - y_true) ** 2))
    mae = float(np.mean(np.abs(preds - y_true)))
    ss_res = float(np.sum((y_true - preds) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    smape = float(np.mean(2 * np.abs(preds - y_true) / (np.abs(preds) + np.abs(y_true) + 1e-8)) * 100)
    return {"mse": mse, "mae": mae, "r2": r2, "smape": smape, "n": len(y_true)}


def main():
    print("Loading price dataset (2013-2026)...")
    ds = load_price_dataset()
    print(f"  {len(ds.dates)} trading days, {ds.dates.min()} .. {ds.dates.max()}")

    print("Loading macro bundle (VIX, BAA10Y credit spread, term spread, real Fed funds)...")
    macro = build_macro_frame_full(ds)

    print("Fitting classical GARCH(1,1) on Train only (-> cond_vol channel)...")
    cond_vol_raw, garch_fit_train = build_garch_std_channel(ds)
    print(f"  omega={garch_fit_train.omega:.4f} alpha={garch_fit_train.alpha:.4f} beta={garch_fit_train.beta:.4f}")

    std_resid_raw = ds.eps / np.clip(cond_vol_raw, 1e-8, None)
    log_return_raw = ds.returns.astype(np.float32)

    dates_idx = pd.DatetimeIndex(ds.dates)
    train_mask_bool = (dates_idx >= TRAIN_START) & (dates_idx <= TRAIN_END)

    log_return_scaled = minmax_scale_train_only(log_return_raw, train_mask_bool)
    cond_vol_scaled = minmax_scale_train_only(cond_vol_raw, train_mask_bool)
    std_resid_scaled = minmax_scale_train_only(std_resid_raw, train_mask_bool)

    print("Fitting 3-state Gaussian HMM on 5 macro+vol features (Train only), causal-filtering full series...")
    hmm_probs, hmm_model = build_hmm_features(ds, macro)
    print(f"  regime states: {REGIME_NAMES}")

    print("Running change-point detectors (diagnostic only)...")
    cp_report = detect_all(ds.returns, dates=dates_idx)
    print(f"  ICSS breaks: {len(cp_report.icss_idx)}, PELT: {len(cp_report.pelt_idx)}, "
          f"z<=-3 shocks: {len(cp_report.zshock_idx)}")
    cp_report.to_frame().to_csv(os.path.join(RESULTS_DIR, "changepoints.csv"), index=False)

    # --- Base 3-feature dual-path windows (Time-Series Aggregated GARCH-GRU) ---
    scaled3 = [log_return_scaled, cond_vol_scaled, std_resid_scaled]
    windows3 = make_dual_path_windows(ds, WINDOW, HORIZON, scaled3, ds.eps)
    train_w3, val_w3, oos_w3 = split_dual_windows_by_date(windows3, TRAIN_START, TRAIN_END,
                                                            VAL_START, VAL_END, OOS_START, OOS_END)

    # --- 6-feature dual-path windows (Regime-Aware Multi-Task model) --------
    scaled6 = scaled3 + [hmm_probs[:, 0], hmm_probs[:, 1], hmm_probs[:, 2]]
    windows6 = make_dual_path_windows(ds, WINDOW, HORIZON, scaled6, ds.eps, regime_probs=hmm_probs)
    train_w6, val_w6, oos_w6 = split_dual_windows_by_date(windows6, TRAIN_START, TRAIN_END,
                                                            VAL_START, VAL_END, OOS_START, OOS_END)
    print(f"  train={len(train_w6.y)} val={len(val_w6.y)} oos={len(oos_w6.y)}")

    date_to_idx = {d: i for i, d in enumerate(ds.dates)}
    results = {}

    # --- RiskMetrics EWMA -----------------------------------------------
    print("Evaluating RiskMetrics EWMA (lambda=0.94)...")
    rm_forecast_var = rolling_forecast(ds.returns, lam=LAMBDA_DEFAULT)

    def riskmetrics_preds(w):
        return np.asarray([np.sqrt(max(rm_forecast_var[date_to_idx[d]], 1e-12)) for d in w.end_dates],
                           dtype=np.float32)

    for split_name, w in [("train", train_w6), ("val", val_w6), ("oos", oos_w6)]:
        results.setdefault("RiskMetrics_EWMA", {})[split_name] = eval_regression(riskmetrics_preds(w), w.y)

    # --- Classical GARCH(1,1) / GJR-GARCH(1,1,1) ------------------------
    print("Evaluating classical GARCH(1,1) / GJR-GARCH(1,1,1)...")
    train_eps = ds.eps[train_mask_bool]

    def garch_preds(fit, sigma2_full, w):
        preds = []
        for end_date in w.end_dates:
            i = date_to_idx[end_date]
            s2 = sigma2_full[i + 1] if (i + 1) < len(sigma2_full) else sigma2_full[-1]
            preds.append(np.sqrt(max(fit.forecast_h_step(s2, HORIZON), 1e-12)))
        return np.asarray(preds, dtype=np.float32)

    garch_fits = {}
    for name, gjr in [("Classical_GARCH11", False), ("Classical_GJR_GARCH", True)]:
        fit = fit_garch(train_eps, gjr=gjr)
        sigma2_full = fit.forecast_path(ds.eps, start_idx=0)
        garch_fits[name] = (fit, sigma2_full)
        for split_name, w in [("train", train_w6), ("val", val_w6), ("oos", oos_w6)]:
            results.setdefault(name, {})[split_name] = eval_regression(garch_preds(fit, sigma2_full, w), w.y)

    # --- Time-Series Aggregated GARCH-GRU (HAR-GARCH-GRU, 3-feature) ---
    print("Training Time-Series Aggregated GARCH-GRU (HAR-GARCH-GRU)...")
    set_seed(SEED)
    har_model = HARGarchGRUModel(input_size=3, hidden_size=HP["hidden_size"], dropout=HP["dropout"])
    har_model = train_single_output(har_model, train_w3, val_w3, HP, label="HAR-GARCH-GRU")
    with torch.no_grad():
        for split_name, w in [("train", train_w3), ("val", val_w3), ("oos", oos_w3)]:
            preds = har_model(torch.from_numpy(w.X), torch.from_numpy(w.E)).numpy()
            results.setdefault("TimeSeriesAggregated_GARCH_GRU", {})[split_name] = eval_regression(preds, w.y)

    # --- Regime-Aware Multi-Task GARCH-GRU (6-feature, 2-layer) ---------
    print("Training Regime-Aware Multi-Task GARCH-GRU (2 layers)...")
    set_seed(SEED)
    mt_model = RegimeAwareMultiTaskGARCHGRU(input_size=6, hidden_size=HP["hidden_size"],
                                             dropout=HP["dropout"], num_layers=2)
    mt_model = train_multitask(mt_model, train_w6, val_w6, HP, lambda_aux=0.001)
    with torch.no_grad():
        for split_name, w in [("train", train_w6), ("val", val_w6), ("oos", oos_w6)]:
            vol_hat, regime_logits = mt_model(torch.from_numpy(w.X), torch.from_numpy(w.E))
            preds = vol_hat.numpy()
            reg_acc = float((regime_logits.argmax(dim=-1).numpy() == w.R.argmax(axis=1)).mean())
            m = eval_regression(preds, w.y)
            m["regime_accuracy"] = reg_acc
            results.setdefault("RegimeAware_MultiTask_GARCH_GRU", {})[split_name] = m

    rows = [{"model": model_name, "split": split_name, **m}
            for model_name, splits in results.items() for split_name, m in splits.items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "table_v2_summary.csv"), index=False)
    print("\n=== SUMMARY (OOS split) ===")
    print(summary[summary.split == "oos"][["model", "mse", "mae", "r2", "smape"]].to_string(index=False))

    # --- VaR backtest (deck's Part 4 methodology: Kupiec/Christoffersen/Binomial/Pinball/Lopez) ---
    # Per the deck (slide 30): "model-specific" quantile -- each model's OWN
    # train-period standardized-residual quantile, not one shared quantile,
    # "lets each model's own residual shape correct for its own biases".
    print("\nRunning VaR backtest on OOS (deck methodology, src/var2.py, model-specific q_alpha)...")
    oos_returns = np.array([ds.returns[date_to_idx[d]] for d in oos_w6.target_dates])
    train_returns_for_q = np.array([ds.returns[date_to_idx[d]] for d in train_w6.target_dates])

    with torch.no_grad():
        mt_oos_vol = mt_model(torch.from_numpy(oos_w6.X), torch.from_numpy(oos_w6.E))[0].numpy()
        har_oos_vol = har_model(torch.from_numpy(oos_w3.X), torch.from_numpy(oos_w3.E)).numpy()
        mt_train_vol = mt_model(torch.from_numpy(train_w6.X), torch.from_numpy(train_w6.E))[0].numpy()
        har_train_vol = har_model(torch.from_numpy(train_w3.X), torch.from_numpy(train_w3.E)).numpy()

    var_rows = []
    for model_name, vol_pred_oos, vol_pred_train in [
        ("RiskMetrics_EWMA", riskmetrics_preds(oos_w6), riskmetrics_preds(train_w6)),
        ("Classical_GARCH11", garch_preds(*garch_fits["Classical_GARCH11"], oos_w6),
         garch_preds(*garch_fits["Classical_GARCH11"], train_w6)),
        ("TimeSeriesAggregated_GARCH_GRU", har_oos_vol, har_train_vol),
        ("RegimeAware_MultiTask_GARCH_GRU", mt_oos_vol, mt_train_vol),
    ]:
        q_alpha = historical_quantile_train(train_returns_for_q, vol_pred_train, ds.mu, ALPHA)
        vr = var_backtest(model_name, ds.mu, q_alpha, vol_pred_oos, oos_returns, alpha=ALPHA)
        var_rows.append({"model": model_name, "q_alpha": q_alpha, "n_obs": vr.n_obs,
                          "expected_violations": vr.expected_violations,
                          "n_violations": vr.n_violations, "violation_ratio": vr.violation_ratio,
                          "kupiec_pvalue": vr.kupiec_pvalue, "kupiec_pass": vr.kupiec_pass,
                          "independence_pvalue": vr.independence_pvalue, "independence_pass": vr.independence_pass,
                          "binomial_pvalue": vr.binomial_pvalue, "binomial_pass": vr.binomial_pass,
                          "mean_pinball_loss": vr.mean_pinball_loss, "mean_lopez_loss": vr.mean_lopez_loss})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(os.path.join(RESULTS_DIR, "table_v2_var.csv"), index=False)
    print(var_df.to_string(index=False))

    # --- Dump arrays for plotting -----------------------------------------
    np.savez(os.path.join(RESULTS_DIR, "arrays.npz"),
              dates=np.asarray(ds.dates), realized_vol_dates=np.asarray(ds.realized_vol.index),
              realized_vol=ds.realized_vol.to_numpy(),
              hmm_calm=hmm_probs[:, 0], hmm_neutral=hmm_probs[:, 1], hmm_stress=hmm_probs[:, 2],
              hmm_hard_label=hmm_probs.argmax(axis=1),
              icss_idx=np.asarray(cp_report.icss_idx), pelt_idx=np.asarray(cp_report.pelt_idx),
              zshock_idx=np.asarray(cp_report.zshock_idx),
              oos_target_dates=np.asarray(oos_w6.target_dates), oos_y_true=oos_w6.y,
              oos_riskmetrics=riskmetrics_preds(oos_w6),
              oos_garch=garch_preds(*garch_fits["Classical_GARCH11"], oos_w6),
              oos_har=har_oos_vol, oos_multitask=mt_oos_vol)

    print("\nDone. Results in", RESULTS_DIR)


if __name__ == "__main__":
    main()
