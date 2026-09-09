# Expanded GARCH-GRU Pipeline — v2 Report (revised against the reference deck)

This revises the earlier v2 report after reading your reference deck
("Final_Presentation_GARCH_GRU", Mphasis/NextLabs internship project,
Satyam Anand, mentored by Samarth & Vinutha) in full and correcting the
implementation against its exact architecture, features, and VaR
methodology. All code is in `src/`; `run_experiment_v2.py` runs the full
pipeline end to end, `plot_v2.py` regenerates the figures.

## What changed since the first pass

Reading the deck surfaced several concrete corrections, applied and re-run:

1. **Base model input features corrected.** The deck's exact "direct
   sequence features" (Part 2/3, slides 17-18, 24-25) are
   `[log_return, cond_vol, std_resid]`, min-max scaled on **train data
   only** — not `[eps, VIX-z, ...]` as I'd assumed. The raw (unscaled)
   residual `eps_t = log_return_t - train_mean` drives the internal
   HAR/GARCH recursion on a separate path, mirroring the deck's dual-path
   design ("Input Features (scaled)" + "Raw History (unscaled):
   eps_prev, sigma2_prev").
2. **HAR equation had gamma/theta swapped.** The deck's slide 25 equation is
   `sigma2_t = omega + gamma_d*sig2_lag1 + gamma_w*sig2_sum5 + gamma_m*sig2_sum20
   + theta_d*eps2_lag1 + theta_w*eps2_sum5 + theta_m*eps2_sum20` — gamma
   scales the variance-history terms, theta scales the shock terms. My
   first draft had these reversed. Fixed in `src/har_cell.py`.
3. **Final model is 2 stacked layers, not 1.** Slide 49: "Layer 1 → Dropout
   → Layer 2". `src/multitask_model.py` now stacks two HAR-GARCH-GRU cells
   with dropout between them.
4. **Regime label is soft, not hard.** Slide 48: "y_regime:[N,3] soft
   labels". The classification auxiliary target is now the HMM's own
   probability vector at the target date, and the loss uses proper soft-
   target cross-entropy (`-mean(sum(target_probs * log_softmax(logits)))`),
   not argmax + standard `CrossEntropyLoss`.
5. **Credit spread gap fixed.** Swapped `BAMLH0A0HYM2` (2023-onward only,
   flagged last time) for the `BAA10Y` series you uploaded (full
   1986–2026 history). The regime HMM is now fit on 5 macro+volatility
   features — realized vol, VIX, BAA10Y credit spread, T10Y3M term spread,
   real Fed funds rate — matching the deck's slide 39 description of
   "regimes... built from real macroeconomic signals," rather than
   volatility alone.
6. **VaR methodology matched exactly.** Slide 30: `VaR_t = mu_train +
   q_alpha^train * sigma_hat_t`, with **model-specific** q_alpha (each
   model's own train-period standardized-residual quantile — slide 30:
   "lets each model's own residual shape correct for its own biases"). Added
   `src/var2.py` with Kupiec (POF), Christoffersen independence, Binomial,
   Pinball, and Lopez — the deck's exact four tests / two losses (slide 31),
   replacing the single violation-ratio check from the first pass.

## Results after the corrections

**Volatility forecasting (1-day horizon), OOS 2025-2026** — now compared
directly against the deck's own published numbers (its OOS period is
somewhat different — smaller N, different exact test window — so treat
this as a fidelity check, not a bit-for-bit reproduction):

| Model | My OOS MSE | My R² | My SMAPE | Deck's OOS MSE | Deck's R² | Deck's SMAPE |
|---|---|---|---|---|---|---|
| RiskMetrics EWMA | 0.235 | 0.313 | 39.1% | 0.274 | 0.253 | 41.9% |
| Classical GARCH(1,1) | 0.101 | 0.703 | 31.1% | 1.893* | −4.151* | 39.4%* |
| Time-Series Aggregated GARCH-GRU | 0.049 | 0.857 | 15.5% | 0.083 | 0.773 | 21.8% |
| Regime-Aware Multi-Task GARCH-GRU | 0.063 | 0.816 | 17.8% | 0.061 | 0.831 | 18.1% |

*The deck's "Rolling GARCH" (M2) number looks like a different
walk-forward/refit convention than my single-fit-then-forecast approach —
its own reported MSE (1.893) is far worse than either RiskMetrics or the
neural models, unlike a standard GARCH(1,1) baseline; not deeply
investigated given the scope, flagged rather than silently reconciled.

The **Regime-Aware Multi-Task GARCH-GRU is now within about 3% of the
deck's own MSE and R²** after the architecture corrections above — a good
fidelity signal that the corrected implementation matches the deck's
intended design, not just its shape.

**VaR backtest, OOS, α=0.05** (`results_v2/table_v2_var.csv`), deck's exact
methodology, model-specific quantile calibration:

| Model | Violations/421 | Violation ratio | Kupiec | Independence | Binomial |
|---|---|---|---|---|---|
| RiskMetrics EWMA | 25 | 5.94% | PASS (p=0.39) | PASS (p=0.67) | PASS (p=0.37) |
| Classical GARCH(1,1) | 26 | 6.18% | PASS (p=0.28) | PASS (p=0.58) | PASS (p=0.26) |
| Time-Series Aggregated GARCH-GRU | 18 | 4.28% | PASS (p=0.48) | PASS (p=0.79) | PASS (p=0.58) |
| Regime-Aware Multi-Task GARCH-GRU | 21 | 4.99% | PASS (p=0.99) | PASS (p=0.96) | PASS (p=1.00) |

All four models pass all three coverage tests — matching the deck's own
table 32/53, where every one of its five models also passes Kupiec and
independence. The Regime-Aware model's violation ratio (4.99%) is
essentially exact calibration against the 5% target, with the highest
Kupiec p-value of the group.

**Regime-classification accuracy** (auxiliary head vs. HMM's own hard label
at the target date): 85.6% (train), 82.8% (val), 91.4% (OOS).

**Figures**: `results_v2/figure_oos_forecasts.png` (all 4 models vs. realized
vol, 2025–2026, including the real volatility spike in April 2025) and
`results_v2/figure_regimes_changepoints.png` (full macro-driven regime
history + all three change-point detectors' flagged dates, 2013–2026).

## Still-open, honestly flagged

- **Single seed, not the deck's 5-seed ensemble**, and no Optuna retuning
  for the new stages (the deck's tuned hidden_size=24, lr=0.00488 vs. my
  fixed hidden_size=16, lr=2e-2) — numbers above are one representative
  run, not an ensemble mean.
- **Per-regime conditional breakdown tables not reproduced** (deck slides
  41-45: separate GMM-detected and HMM-detected regime rankings per model).
  This is a substantial additional analysis — every model's forecast would
  need to be re-scored separately within each detected regime, for both a
  GMM and an HMM regime labeling. Straightforward to add if you want it, but
  out of scope for this pass.
- **"Rolling GARCH" discrepancy** (see table footnote above) not
  investigated — my Classical GARCH(1,1) baseline is a standard
  fit-once-on-train, forecast-forward implementation; the deck's own
  "Rolling GARCH" MSE is far worse than that, suggesting a different
  (possibly walk-forward-refit) convention I haven't tried to reverse-
  engineer.
- **Composite rule-based / GMM regime detectors** (deck's other two of its
  three regime-detection methods, slide 40) are not implemented — only the
  HMM, which is the one actually feeding the multi-task model's live
  regime-probability features.
