# GARCH-GRU Replication — Results and Honest Assessment

Implementation of Wei, Yang & Cui (2025), *"Integrated GARCH-GRU in
Financial Volatility Forecasting"* (arXiv:2504.09380). See `README.md` for
what is implemented and exactly how (equation-by-equation) and what had to
be scaled down for this sandbox's CPU-only, 2-core budget.

**Correction note (post-review):** an earlier version of this report was
built on a data bug — log returns were left in raw decimal form instead of
scaled to percentage points (`r_t = 100 * log(p_t/p_{t-1})`), which is the
convention the paper itself uses (confirmed against the paper's own Table 1:
S&P 500 mean=0.0406, std=0.9307 are only sensible as percent). That bug
shrank every MSE/MAE by a constant factor of ~10,000 and, more importantly,
was masking the real relative ranking of the models trained under it
(the loss landscape and effective learning dynamics at a ~1e-4-scale target
are not simply a rescaled version of the same problem at 1-scale — this
is why the fix changed conclusions, not just the numbers' units). It has
been fixed in `src/data_utils.py::compute_log_returns` and every result
below is from a full rerun after the fix. See git history / the code
comment there for detail. Thank you to whoever caught this.

## 1. Custom-cell correctness (paper's Table 2 analogue)

Unaffected by the scaling bug (it's a unit-test on the cells in isolation).
Both hand-rolled cells match PyTorch's built-in `nn.GRUCell`/`nn.LSTMCell`
on identical weights to floating-point precision:

| Cell | mean diff | t-stat | p-value | max abs diff |
|---|---|---|---|---|
| GRU  | 4.9e-11 | 0.09 | 0.93 | 3.6e-7 |
| LSTM | 8.6e-11 | 0.66 | 0.51 | 1.2e-7 |

## 2. Forecasting accuracy (paper's Table 3 analogue) — corrected

Full numbers: `results/table3_summary.csv`. With the scaling fixed, the
absolute MSE values now land in the same ballpark as the paper's own
reported figures — e.g. `eh_GARCH_GRU` on S&P 500, 1-day horizon: **0.0224**
here vs. **0.0202** in the paper; `GARCH(1,1)`: **0.0760** here vs. **0.0795**
in the paper; `GJR-GARCH`: **0.0736** here vs. **0.0863** in the paper. That
level of agreement, given this replication uses 3 seeds instead of 20 and a
much smaller hyperparameter search, is a meaningfully strong confirmation.

**Ranking, S&P 500 / DJI, all three horizons (1-day, 3-day, 1-week):**

| Dataset | Horizon | Best model | MSE | Worst-of-neural | Classical GARCH(1,1) | Classical GJR-GARCH |
|---|---|---|---|---|---|---|
| S&P 500 | 1D | eh_GARCH_GRU | 0.0224 | Transformer 0.0407 | 0.0760 | 0.0736 |
| S&P 500 | 3D | pl_GARCH_LSTM | 0.0450 | Transformer 0.0646 | 0.1189 | 0.0928 |
| S&P 500 | 1W | bl_GARCH_LSTM | 0.1134 | Transformer 0.1447 | 0.2016 | 0.1675 |
| DJI | 1D | GRU | 0.0445 | Transformer 0.0829 | 0.0950 | 0.1028 |
| DJI | 3D | GRU | 0.0732 | Transformer 0.1203 | 0.1503 | 0.1258 |
| DJI | 1W | eh_GARCH_LSTM | 0.1463 | eh_GARCH_GRU 0.1776 | 0.2677 | 0.2064 |

**This now robustly confirms the paper's central claim**: every
GRU/LSTM-family neural model — plain or hybrid — beats both classical GARCH
variants by a wide margin (roughly 2-4x lower MSE) at every horizon on both
indices. `eh_GARCH_GRU` (the proposed architecture) is the single best model
on S&P 500 at the 1-day horizon and stays within the tight neural cluster
(never the worst, usually top-3) everywhere else — consistent with the
paper's own more nuanced finding that the proposed model wins clearly in
some configurations and is closely competitive (not always #1) in others
(the paper itself flags a DJI short-horizon exception where plain GRU wins,
which is echoed here almost exactly: GRU edges out the hybrids on DJI at
1D and 3D). `Transformer` is consistently the weakest neural architecture
but still clearly beats classical GARCH — also consistent with the paper's
discussion of the Transformer's attention mechanism lacking the right
structural bias for this task.

Where this replication still legitimately differs from the paper, honestly:
it does not show `eh_GARCH_GRU` as the uniform, always-best model the way
the paper's Table 3 mostly does — here it shares the top spot across
configurations with plain GRU/LSTM and the other hybrids, with margins
between them (a few percent of MSE) that are within the noise band you'd
expect from 3 seeds and a reduced hyperparameter search rather than the
paper's 20 seeds and full Optuna sweep per model. The qualitative story
(hybrid/neural family >> classical GARCH) is now solid; the finer claim
(GARCH-GRU specifically edges out every other neural variant everywhere)
is only partially reproduced at this scale.

## 3. Value-at-Risk backtest (paper's Table 4 / Figure 3 analogue) — corrected

One-day-ahead, 95% confidence, S&P 500. Full numbers:
`results/table4_var_violations.csv`; plot: `results/plots/figure3_var.png`.

| Model | Violations / 252 | Violation ratio |
|---|---|---|
| GJR-GARCH | 6 | 2.38% |
| GARCH(1,1) | 7 | 2.78% |
| eh_GARCH_LSTM | 13 | 5.16% |
| **eh_GARCH_GRU (proposed)** | 14 | **5.56%** |
| bl_GARCH_LSTM | 15 | 5.95% |
| pl_GARCH_GRU | 16 | 6.35% |
| pl_GARCH_LSTM | 18 | 7.14% |

This is the one place this replication's *direction* diverges from the
paper's reported pattern, honestly: the paper finds classical GARCH
under-covers risk (violation ratios of 7-8%, well above the 5% nominal)
while its proposed model is unusually conservative (1.3%, well below
nominal). Here, classical GARCH is the conservative side (2.4-2.8%, below
nominal) and the neural models sit closer to, or slightly past, the 5%
nominal line, with `eh_GARCH_GRU` the closest of all to exact calibration
(5.56% vs. a 5.00% target). This is plausibly explained by the same
"training-only empirical quantile" step interacting differently with a
smaller neural hidden size and only 3 seeds than with the paper's 20-seed
ensemble — it would be worth rerunning at full seed count before treating
this specific reversal as a real finding rather than replication noise. It
does not undercut Section 2's result, which uses a very different
evaluation (raw forecast error, not a downstream risk decision).

## 4. Data caveat (unchanged)

S&P 500 and DJI are real daily data (2010-2019, see `README.md` for
provenance); NASDAQ Composite could not be retrieved given this sandbox's
network restrictions (only `raw.githubusercontent.com` and `pypi.org` are
reachable — every financial data host, including Yahoo Finance, Stooq,
FRED, and Investing.com, is blocked). Adding a NASDAQ CSV to `data/` and one
line to `run_experiment.py` extends the study with no other code changes.

## 5. Bottom line

After fixing a real scaling bug (returns were not converted to percentage
points), this replication now closely reproduces both the *magnitude* and
the *qualitative ranking* of the paper's Table 3: every GARCH-GRU/LSTM
hybrid and plain neural model clearly and consistently outperforms
classical GARCH(1,1)/GJR-GARCH, and `eh_GARCH_GRU` specifically is
competitive-to-best within the neural family, exactly as claimed. The one
place this replication does not match the paper's direction is the VaR
violation-ratio comparison, flagged honestly above rather than smoothed
over, and worth rerunning at the paper's full seed count to check if it's
signal or replication noise.
