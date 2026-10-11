# Results (n=300, seed = 54) 

## Dataset
Delayed (>12h) rate: **35.0%**; missing port-arrival ts: 8.7%, missing customs ts: 8.7% 

| lateness_h | mean | std | p90 |
|---|---|---|---|
| L0 | 3.88 | 7.63 | 9.76 |
| true_L1 | 5.13 | 9.11 | 14.09 |
| true_L2 | 15.85 | 18.96 | 37.89 |
| L3 | 16.33 | 22.05 | 46.10 |

Mean (positive) causal contribution by source, hours:

| source | hours |
|---|---|
| origin | 2.09 |
| weather | 3.36 |
| route_noise | 1.18 |
| congestion | 6.54 |
| processing | 2.08 |
| inspection | 1.76 |
| inland | 1.51 |

1. ETA accuracy by checkpoint (random 5-fold x 3)
MAE/RMSE in hours of final ETA error.

**CP0 departure**

| model | MAE | MAE_lo | MAE_hi | RMSE | bias |
|---|---|---|---|---|---|
| sched | 16.67 | 14.22 | 19.25 | 27.41 | -16.33 |
| carry | 14.52 | 12.49 | 16.83 | 24.00 | -12.45 |
| route_prior | 15.01 | 13.57 | 16.68 | 20.29 | 0.18 |
| ridge | 13.12 | 11.91 | 14.29 | 17.40 | -0.10 |
| gbm | 13.56 | 12.35 | 14.87 | 18.20 | 0.28 |
| chain | 11.50 | 10.34 | 12.75 | 16.06 | 0.64 |

**CP1 port arrival**

| model | MAE | MAE_lo | MAE_hi | RMSE | bias |
|---|---|---|---|---|---|
| sched | 16.67 | 14.22 | 19.25 | 27.41 | -16.33 |
| carry | 13.06 | 11.32 | 14.98 | 20.44 | -11.02 |
| route_prior | 15.01 | 13.57 | 16.68 | 20.29 | 0.18 |
| ridge | 9.26 | 8.25 | 10.23 | 12.66 | 0.04 |
| gbm | 8.88 | 7.90 | 9.81 | 12.80 | 0.34 |
| chain | 8.60 | 7.61 | 9.58 | 12.53 | 0.17 |

**CP2 customs cleared**

| model | MAE | MAE_lo | MAE_hi | RMSE | bias |
|---|---|---|---|---|---|
| sched | 16.67 | 14.22 | 19.25 | 27.41 | -16.33 |
| carry | 4.67 | 4.08 | 5.26 | 7.03 | -0.89 |
| route_prior | 15.01 | 13.57 | 16.68 | 20.29 | 0.18 |
| ridge | 4.13 | 3.63 | 4.68 | 6.57 | 0.04 |
| gbm | 3.81 | 3.30 | 4.30 | 6.19 | 0.08 |
| chain | 3.42 | 2.89 | 3.97 | 6.18 | -0.44 |

Paired MAE improvement over baseline (hours, 95% bootstrap CI; >0 = model better):

| cp | base | model | gain_h | lo | hi |
|---|---|---|---|---|---|
| 0 | carry | gbm | 0.96 | -0.76 | 2.86 |
| 1 | carry | gbm | 4.18 | 2.76 | 5.75 |
| 2 | carry | gbm | 0.86 | 0.50 | 1.20 |
| 0 | carry | chain | 3.02 | 1.28 | 4.97 |
| 1 | carry | chain | 4.46 | 3.09 | 5.98 |
| 2 | carry | chain | 1.25 | 0.94 | 1.56 |
| 0 | sched | chain | 5.17 | 3.25 | 7.37 |
| 1 | sched | chain | 8.07 | 5.94 | 10.31 |
| 2 | sched | chain | 13.25 | 10.98 | 15.74 |

2. Delay classification (P(delay>12h))

| cp | model | base_rate | PR_AUC | ROC_AUC | Brier | F1@0.5 |
|---|---|---|---|---|---|---|
| 0 | gbm_classifier | 0.35 | 0.54 | 0.67 | 0.21 | 0.37 |
| 0 | chain_monte_carlo | 0.35 | 0.72 | 0.78 | 0.17 | 0.64 |
| 0 | carry_forward_rule | 0.35 | 0.49 | 0.60 | n/a | n/a |
| 1 | gbm_classifier | 0.35 | 0.81 | 0.84 | 0.14 | 0.70 |
| 1 | chain_monte_carlo | 0.35 | 0.82 | 0.85 | 0.13 | 0.73 |
| 1 | carry_forward_rule | 0.35 | 0.69 | 0.75 | n/a | n/a |
| 2 | gbm_classifier | 0.35 | 0.94 | 0.95 | 0.05 | 0.91 |
| 2 | chain_monte_carlo | 0.35 | 0.94 | 0.95 | 0.05 | 0.91 |
| 2 | carry_forward_rule | 0.35 | 0.95 | 0.96 | n/a | n/a |

3. Prediction intervals (nominal 80%)

| cp | model | coverage_80 | mean_width_h |
|---|---|---|---|
| 0 | gbm_quantile | 0.60 | 27.77 |
| 0 | chain_monte_carlo | 0.79 | 34.60 |
| 1 | gbm_quantile | 0.61 | 19.84 |
| 1 | chain_monte_carlo | 0.81 | 25.36 |
| 2 | gbm_quantile | 0.64 | 7.59 |
| 2 | chain_monte_carlo | 0.75 | 8.60 |

4. Explanation validity vs generator ground truth
Spearman between explained stage contribution and exact Shapley causal contribution (across shipments). Inland (S3) is n/a: nothing about inland execution is observable before delivery, so the model (correctly) attributes 0 to it. The last row is top-1 stage agreement, not a correlation.

| cp | stage | spearman |
|---|---|---|
| 0 | Origin departure | 0.69 |
| 0 | Ocean transit | 0.29 |
| 0 | Port & customs | 0.30 |
| 0 | Inland delivery | n/a |
| 0 | top-stage agreement (delayed shipments) | 0.38 |
| 1 | Origin departure | 0.85 |
| 1 | Ocean transit | 0.61 |
| 1 | Port & customs | 0.70 |
| 1 | Inland delivery | n/a |
| 1 | top-stage agreement (delayed shipments) | 0.57 |
| 2 | Origin departure | 0.84 |
| 2 | Ocean transit | 0.64 |
| 2 | Port & customs | 0.88 |
| 2 | Inland delivery | n/a |
| 2 | top-stage agreement (delayed shipments) | 0.75 |

 Held-out-route generalisation (leave-one-route-out)

| model | 0 | 1 | 2 |
|---|---|---|---|
| carry | 14.52 | 13.09 | 4.64 |
| chain | 11.85 | 8.93 | 3.40 |
| gbm | 13.42 | 8.65 | 4.11 |
| ridge | 25.27 | 12.44 | 4.10 |
| sched | 16.67 | 16.67 | 16.67 |

 Forward-in-time split (expanding window, delivery embargo)

| model | 0 | 1 | 2 |
|---|---|---|---|
| carry | 10.92 | 10.76 | 4.63 |
| chain | 12.13 | 9.09 | 3.50 |
| gbm | 14.33 | 9.21 | 3.85 |
| ridge | 15.26 | 9.66 | 4.31 |
| sched | 12.93 | 12.93 | 12.93 |

Ablations (MAE, hours; random CV)

| config | gbm_cp0 | gbm_cp1 | gbm_cp2 | gbm_mean | ridge_mean |
|---|---|---|---|---|---|
| full | 13.68 | 8.96 | 3.78 | 8.80 | 8.83 |
| - interactions (queue/congestion x processing) | 13.75 | 9.08 | 3.80 | 8.88 | 9.12 |
| - propagation features (buffers, increments, risks) | 13.74 | 9.13 | 3.79 | 8.88 | 8.96 |
| - route priors | 13.32 | 8.79 | 3.75 | 8.62 | 8.76 |
| - missing-data model (LOCF instead) | 13.68 | 9.03 | 4.13 | 8.95 | 9.31 |
| - all four | 13.41 | 9.37 | 4.08 | 8.95 | 9.48 |

Robustness to extra missing events (GBM MAE, hours)

| extra_missing | cp | locf | model |
|---|---|---|---|
| 0.00 | 1 | 9.16 | 9.11 |
| 0.00 | 2 | 4.20 | 3.79 |
| 0.10 | 1 | 9.45 | 9.26 |
| 0.10 | 2 | 4.73 | 4.33 |
| 0.20 | 1 | 9.71 | 9.25 |
| 0.20 | 2 | 5.32 | 4.80 |
| 0.30 | 1 | 9.89 | 9.36 |
| 0.30 | 2 | 6.61 | 5.82 |
| 0.45 | 1 | 10.78 | 10.06 |
| 0.45 | 2 | 7.75 | 6.82 |

## 9. Worked example
```
Predicted final delay: +23.0 h  (P10 +13.7 / P90 +26.5 h; P(delay>12h)=75%)
  Origin departure  +13.4 h
  Ocean transit      -5.1 h
  Port & customs     +2.3 h
  Inland delivery    +0.0 h
  baseline/propagation  +12.4 h
Top model drivers: weekend/holiday timing (+2.0 h), carried delay & buffers (propagation) (+1.2 h), observed transit delay (-1.1 h)
Counterfactual: with normal congestion (0.65) delay would be +21.5 h (-1.5 h).
```
Ground-truth causal contributions (hours): {"true_phi_origin": 3.3, "true_phi_weather": 1.1, "true_phi_route_noise": 3.1, "true_phi_congestion": 12.7, "true_phi_processing": 10.3, "true_phi_inspection": 0.0, "true_phi_inland": -0.5}
