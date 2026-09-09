# GARCH-GRU: Implementation of Wei, Yang & Cui (2025)

This is a from-scratch PyTorch implementation of **"Integrated GARCH-GRU in
Financial Volatility Forecasting"** (Wei, Yang, Cui — Stevens Institute of
Technology, arXiv:2504.09380), including the full benchmark suite the paper
compares against, a paper-faithful rolling-window evaluation protocol, and a
Value-at-Risk backtest.

## What is implemented, and how faithfully

### Core architecture (exact equations)

| Component | File | Paper reference |
|---|---|---|
| `GARCHGRUCell` — integrated GARCH-GRU | `src/cells.py` | Eq. 15-20 |
| `GARCHLSTMCell` — integrated GARCH-LSTM (comparator) | `src/cells.py` | Eq. 21-28 |
| `EmbeddedGARCHOutputGateLSTMCell` — `bl_GARCH_LSTM` (Zhao et al. 2024 style: GARCH regression replaces the LSTM output gate) | `src/cells.py` | Sec. 2/4.1 "Benchmarks" |
| Pipeline hybrids `pl_GARCH_GRU` / `pl_GARCH_LSTM` (classical GARCH(1,1) conditional std fed as an extra input channel to a plain GRU/LSTM) | `src/train.py::prepare_windows_for_model` | Sec. 4.1 "Benchmarks" |
| Plain `GRU` / `LSTM` / `Transformer` | `src/models.py` | Sec. 4.1 "Benchmarks" |
| Classical `GARCH(1,1)` / `GJR-GARCH(1,1,1)` (from-scratch Gaussian QMLE) | `src/garch.py` | Eq. 3, 14 |
| Output head: `Linear -> softplus -> sqrt` | `src/models.py::VolOutputHead` | Sec. 3.2.1, "constrained output layer" |
| GARCH-parameter reparameterization for stationarity (ω>0, α,β≥0, α+β<1) | `src/cells.py::GarchParams` | Sec. 3.2.1 |
| Custom-cell correctness validation (paired t-test vs `torch.nn.GRUCell`/`LSTMCell`) | `src/validate_cells.py` | Table 2 |
| VaR backtest (non-parametric, empirical quantile of standardized residuals) | `src/var.py` | Eq. 31-34 |

The custom-cell validation reproduces the paper's Table 2 exactly in spirit:
on identical weights, our hand-rolled GRU/LSTM cells match PyTorch's
built-in cells to floating-point precision (mean difference ~1e-11,
p > 0.9), confirming that the GARCH integration — not an implementation
bug — is what differentiates the hybrid models. (Note: PyTorch's `GRUCell`
labels its update gate as "fraction of *old* state retained," the opposite
convention from the paper's Eq. 13 "fraction of *new* candidate written."
The two are the same GRU up to `z <-> 1-z` relabeling; `validate_cells.py`
accounts for this explicitly rather than silently comparing two different
gating conventions.)

### Data

The paper uses daily closes for the **S&P 500, Dow Jones Industrial
Average, and NASDAQ Composite, 2010-01-01 to 2019-12-31**, from what is
presumably a Bloomberg/Yahoo Finance pull. This sandboxed environment's
network egress is restricted to `pypi.org` and `raw.githubusercontent.com`
(Yahoo Finance, Stooq, FRED, and every other financial data host are
blocked at the proxy level) — extensively verified during this session.

- **S&P 500** and **Dow Jones Industrial Average**: real daily OHLC data,
  originally Yahoo-Finance-sourced, obtained from the public GitHub
  repository `fja05680/dow-sp500-100-years`
  (`data/SP500_raw.csv`, `data/DJI_raw.csv`), spanning 2010-01-01 through
  2019-12-23/24 (a handful of the very last trading days of 2019 are
  missing from that mirror; immaterial to a 10-year study).
- **NASDAQ Composite**: **could not be retrieved** — no public
  `raw.githubusercontent.com`-hosted daily history for the Composite index
  (as opposed to individual NASDAQ-listed stocks, which are plentiful) was
  found after an extensive search. Rather than substitute a different
  series and call it "NASDAQ," the empirical study here uses **S&P 500 and
  DJI only**. **To add NASDAQ**, drop a CSV with `Date`/`Close` columns at
  `data/NASDAQ_raw.csv` and add one line to `DATASETS` in
  `run_experiment.py` — no other code changes are needed.

### Rolling-window protocol

Figure 2 of the paper depicts sliding a window across the *entire* sample
to build a supervised dataset of `(22-day window -> realized volatility
h days later)` pairs ("Subsample 1 ... Subsample T-m+1"), which is the
standard way to turn a time series into training examples for a
sequence-to-one regressor. We implement exactly this
(`src/data_utils.py::make_windows`): a 22-day window of the mean-adjusted
return `eps_t` (Eq. 1) is the input sequence; the 5-day realized volatility
(Eq. 30) `h` days after the window's last day is the label. The last 252
label dates are the test set; of the remainder, the last 20% (chronological)
is the validation set used for early stopping (patience 10, Adam,
monitoring MSE — Sec. 4.1). Models are trained **once** per
(dataset, horizon, seed) on this sliding-window dataset, then evaluated on
the 252 held-out test windows — this is both the literal reading of Fig. 2
and the only computationally tractable reading (retraining from scratch on
each of 252 individual 22-day windows, as a more literal but less standard
reading of the prose would imply, was tested and produces materially the
same forecasts at ~250x the compute cost for windows this short).

### What was scaled down for the sandbox's CPU-only, 2-core budget

The paper's design averages over **20 random seeds** and Optuna-tunes every
model. This replication:

- Uses **3 seeds** (0, 1, 2) instead of 20 for the neural models (classical
  GARCH is deterministic, so seeds don't apply there).
- Runs an 8-trial Optuna search (`src/hpo.py`) for `eh_GARCH_GRU` on each
  dataset only, and applies the resulting hyperparameters
  (`results/tuned_hp.json`) to all three embedded-hybrid architectures
  (`eh_GARCH_GRU`, `eh_GARCH_LSTM`, `bl_GARCH_LSTM`); the remaining
  (architecturally simpler, less GARCH-sensitive) benchmarks use a single
  fixed, reasonable configuration (`src/train.py::DEFAULT_HP`) rather than
  their own per-model search.
- Both are `README`-documented, one-line changes to widen
  (`run_experiment.py::SEEDS`, `src/hpo.py`'s `n_trials`) if you have more
  compute available; the code has no other shortcuts.

## Repository layout

```
garch_gru/
  data/                     SP500_raw.csv, DJI_raw.csv (real, see above)
  src/
    data_utils.py           returns, realized vol, sliding-window dataset
    garch.py                GARCH(1,1) / GJR-GARCH(1,1,1) QMLE + h-step forecast
    cells.py                all custom recurrent cells (Eq. 4-28)
    models.py                sequence models + output head + registry
    train.py                training loop, early stopping, classical-GARCH eval
    hpo.py                  Optuna search
    var.py                  VaR backtest
    validate_cells.py       Table-2-style custom-vs-torch equivalence test
  run_experiment.py         main sweep -> results/raw_results.csv (resumable)
  summarize_results.py      -> results/table3_summary.csv
  run_var_experiment.py     -> results/table4_var_violations.csv + Figure-3 plot
  results/                  all output CSVs and plots
```

## Reproducing / extending

```bash
pip install torch pandas numpy scipy optuna matplotlib

python3 -m src.validate_cells            # Table 2 equivalent
python3 run_experiment.py                # Table 3 equivalent (resumable; ctrl-C safe)
python3 summarize_results.py             # aggregate to results/table3_summary.csv
python3 run_var_experiment.py            # Table 4 + Figure 3 equivalent
```

To run the full paper-scale study: set `SEEDS = list(range(20))` in
`run_experiment.py` and raise `n_trials` in `src/hpo.py`; everything else
is unchanged. Expect roughly (20/3) x (8 hp trials you add) x the runtimes
already observed here.

## Honest summary of results

See `results/table3_summary.csv` / `results/table4_var_violations.csv` and
the accompanying write-up (`REPORT.md`) for the actual numbers from this
run, an honest comparison against the paper's reported figures, and where
this replication does and does not confirm the paper's central claim (that
tightly-coupled GARCH-GRU integration outperforms looser hybrid designs,
plain neural nets, and classical GARCH).
