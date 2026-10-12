# Results - synthetic (n=300, seed=54)

## Dataset
Delayed (>12h) rate: **29.3%**; missing port-arrival ts: 9.3%, missing customs ts: 14.3%

| lateness_h | observed | mean | std | p90 |
|---|---|---|---|---|
| L0 | 300 | 3.83 | 6.19 | 11.05 |
| L1_obs | 272 | 3.80 | 6.71 | 9.83 |
| L2_obs | 257 | 11.58 | 14.60 | 31.26 |
| L3 | 300 | 12.44 | 18.36 | 37.49 |

Mean (positive) causal contribution by source, hours (synthetic ground truth):

| source | hours |
|---|---|
| origin | 1.36 |
| weather | 2.36 |
| route_noise | 0.91 |
| congestion | 5.08 |
| processing | 1.79 |
| inspection | 2.01 |
| inland | 1.35 |

## 1. ETA accuracy by checkpoint (random 5-fold x 3)
MAE/RMSE in hours of final ETA error.

**CP0 departure**

| model | MAE | MAE_lo | MAE_hi | RMSE | bias |
|---|---|---|---|---|---|
| sched | 12.73 | 10.75 | 14.96 | 22.16 | -12.44 |
| carry | 11.58 | 9.88 | 13.46 | 19.54 | -8.61 |
| route_prior | 13.24 | 12.01 | 14.64 | 17.63 | 0.08 |
| ridge | 11.76 | 10.53 | 13.06 | 16.40 | -0.14 |
| gbm | 12.51 | 11.25 | 13.83 | 17.51 | 0.02 |
| chain | 11.83 | 10.51 | 13.20 | 17.00 | -0.13 |
| chain_median | 10.82 | 9.30 | 12.44 | 17.40 | -3.41 |

**CP1 port arrival**

| model | MAE | MAE_lo | MAE_hi | RMSE | bias |
|---|---|---|---|---|---|
| sched | 12.73 | 10.75 | 14.96 | 22.16 | -12.44 |
| carry | 10.90 | 9.23 | 12.73 | 18.61 | -8.61 |
| route_prior | 13.24 | 12.01 | 14.64 | 17.63 | 0.08 |
| ridge | 9.21 | 8.14 | 10.19 | 13.12 | 0.07 |
| gbm | 8.95 | 7.88 | 9.94 | 13.11 | -0.06 |
| chain | 8.63 | 7.60 | 9.69 | 12.98 | -1.03 |
| chain_median | 7.74 | 6.56 | 8.94 | 13.46 | -3.87 |

**CP2 customs cleared**

| model | MAE | MAE_lo | MAE_hi | RMSE | bias |
|---|---|---|---|---|---|
| sched | 12.73 | 10.75 | 14.96 | 22.16 | -12.44 |
| carry | 4.67 | 4.07 | 5.36 | 7.39 | -0.64 |
| route_prior | 13.24 | 12.01 | 14.64 | 17.63 | 0.08 |
| ridge | 4.27 | 3.71 | 4.92 | 6.92 | 0.08 |
| gbm | 3.98 | 3.39 | 4.64 | 6.91 | 0.24 |
| chain | 3.43 | 2.82 | 4.09 | 6.63 | -1.01 |
| chain_median | 3.38 | 2.77 | 4.08 | 6.72 | -1.55 |

Paired MAE improvement over baseline (hours, 95% bootstrap CI; >0 = model better):

| cp | base | model | gain_h | lo | hi |
|---|---|---|---|---|---|
| 0 | carry | gbm | -0.93 | -2.11 | 0.34 |
| 1 | carry | gbm | 1.95 | 0.72 | 3.36 |
| 2 | carry | gbm | 0.68 | 0.27 | 1.12 |
| 0 | carry | chain | -0.25 | -1.55 | 1.07 |
| 1 | carry | chain | 2.27 | 1.24 | 3.44 |
| 2 | carry | chain | 1.24 | 0.89 | 1.56 |
| 0 | carry | chain_median | 0.76 | -0.39 | 1.93 |
| 1 | carry | chain_median | 3.16 | 2.27 | 4.21 |
| 2 | carry | chain_median | 1.29 | 0.94 | 1.60 |
| 0 | sched | chain_median | 1.91 | 0.66 | 3.27 |
| 1 | sched | chain_median | 4.99 | 3.80 | 6.43 |
| 2 | sched | chain_median | 9.35 | 7.59 | 11.28 |

## 2. Delay classification (P(delay>12h))

| cp | model | base_rate | PR_AUC | ROC_AUC | Brier | F1@0.5 |
|---|---|---|---|---|---|---|
| 0 | gbm_classifier | 0.29 | 0.45 | 0.70 | 0.19 | 0.28 |
| 0 | chain_monte_carlo | 0.29 | 0.51 | 0.71 | 0.19 | 0.48 |
| 0 | carry_forward_rule | 0.29 | 0.44 | 0.59 | n/a | n/a |
| 1 | gbm_classifier | 0.29 | 0.69 | 0.79 | 0.15 | 0.55 |
| 1 | chain_monte_carlo | 0.29 | 0.73 | 0.81 | 0.14 | 0.64 |
| 1 | carry_forward_rule | 0.29 | 0.54 | 0.69 | n/a | n/a |
| 2 | gbm_classifier | 0.29 | 0.93 | 0.96 | 0.05 | 0.88 |
| 2 | chain_monte_carlo | 0.29 | 0.92 | 0.94 | 0.05 | 0.89 |
| 2 | carry_forward_rule | 0.29 | 0.94 | 0.95 | n/a | n/a |

## 3. Prediction intervals (nominal 80%)

| cp | model | coverage_80 | mean_width_h |
|---|---|---|---|
| 0 | gbm_quantile | 0.62 | 24.15 |
| 0 | chain_monte_carlo | 0.81 | 35.51 |
| 1 | gbm_quantile | 0.62 | 19.59 |
| 1 | chain_monte_carlo | 0.80 | 24.35 |
| 2 | gbm_quantile | 0.68 | 8.30 |
| 2 | chain_monte_carlo | 0.73 | 7.61 |

## 4. Explanation vs generator ground truth (synthetic data only)
Spearman between each explained stage quantity and the simulator's exact Shapley attribution of the final delay to that stage (across shipments). The simulator's attribution is its own counterfactual, not real-world causality. Inland (S3) has no observable signal before delivery, so GBM/SHAP attributes 0 (n/a).

**4a. Chain-model stage breakdown (the explanation shown in the app headline)**

| cp | stage | spearman |
|---|---|---|
| 0 | Origin departure | 0.92 |
| 0 | Ocean transit | 0.10 |
| 0 | Port & customs | 0.42 |
| 0 | Inland delivery | 0.01 |
| 0 | top-stage agreement (delayed shipments) | 0.66 |
| 1 | Origin departure | 0.92 |
| 1 | Ocean transit | 0.57 |
| 1 | Port & customs | 0.60 |
| 1 | Inland delivery | 0.02 |
| 1 | top-stage agreement (delayed shipments) | 0.70 |
| 2 | Origin departure | 0.92 |
| 2 | Ocean transit | 0.57 |
| 2 | Port & customs | 0.87 |
| 2 | Inland delivery | 0.14 |
| 2 | top-stage agreement (delayed shipments) | 0.82 |

**4b. GBM comparator: observed stage delay + SHAP groups (diagnostics only)**

| cp | stage | spearman |
|---|---|---|
| 0 | Origin departure | 0.66 |
| 0 | Ocean transit | 0.19 |
| 0 | Port & customs | 0.33 |
| 0 | Inland delivery | n/a |
| 0 | top-stage agreement (delayed shipments) | 0.36 |
| 1 | Origin departure | 0.82 |
| 1 | Ocean transit | 0.55 |
| 1 | Port & customs | 0.60 |
| 1 | Inland delivery | n/a |
| 1 | top-stage agreement (delayed shipments) | 0.57 |
| 2 | Origin departure | 0.84 |
| 2 | Ocean transit | 0.58 |
| 2 | Port & customs | 0.85 |
| 2 | Inland delivery | n/a |
| 2 | top-stage agreement (delayed shipments) | 0.79 |

## 5. Held-out-route generalisation (leave-one-route-out)

| model | 0 | 1 | 2 |
|---|---|---|---|
| carry | 11.58 | 10.91 | 4.72 |
| chain | 11.29 | 8.15 | 3.41 |
| gbm | 12.88 | 9.05 | 3.79 |
| ridge | 24.11 | 9.70 | 4.40 |
| sched | 12.73 | 12.73 | 12.73 |

## 6. Forward-in-time split (expanding window, delivery embargo)

| model | 0 | 1 | 2 |
|---|---|---|---|
| carry | 11.23 | 10.83 | 4.65 |
| chain | 10.95 | 8.85 | 3.51 |
| gbm | 12.50 | 8.74 | 4.34 |
| ridge | 11.65 | 8.39 | 4.52 |
| sched | 12.47 | 12.47 | 12.47 |

## 7. Ablations (MAE, hours; random CV)

| config | gbm_cp0 | gbm_cp1 | gbm_cp2 | gbm_mean | ridge_mean |
|---|---|---|---|---|---|
| full | 12.57 | 8.85 | 3.95 | 8.46 | 8.41 |
| - interactions (queue/congestion x processing) | 12.51 | 8.98 | 3.88 | 8.46 | 8.88 |
| - propagation features (buffers, increments, risks) | 12.55 | 9.04 | 3.95 | 8.51 | 8.47 |
| - route priors | 12.60 | 8.78 | 3.90 | 8.43 | 8.40 |
| - missing-data model (LOCF instead) | 12.57 | 9.12 | 4.57 | 8.75 | 8.96 |
| - all four | 12.67 | 9.12 | 4.47 | 8.75 | 9.41 |

## 8. Robustness to extra missing events (GBM MAE, hours)

| extra_missing | cp | locf | model |
|---|---|---|---|
| 0.00 | 1 | 9.18 | 8.88 |
| 0.00 | 2 | 4.64 | 3.98 |
| 0.10 | 1 | 9.45 | 9.22 |
| 0.10 | 2 | 4.97 | 4.44 |
| 0.20 | 1 | 9.99 | 9.44 |
| 0.20 | 2 | 5.33 | 5.06 |
| 0.30 | 1 | 10.03 | 9.43 |
| 0.30 | 2 | 6.12 | 5.87 |
| 0.45 | 1 | 10.08 | 9.76 |
| 0.45 | 2 | 6.36 | 6.38 |

## 9. Worked example (checkpoint 1; shipment held out of training)
```
The median estimate is 0.5 h early against the scheduled delivery; 80% of simulated outcomes fall between -5.1 h and +13.5 h.
So far: -2.6 h accumulated (origin departure +0.0 h, ocean transit -2.6 h).
Most of the delay still to come is expected in port & customs (+3.2 h on average).
Schedule buffer recovered about 2.6 h during ocean transit.
Estimated probability of arriving more than 12 h late: 13% (share of simulated outcomes).
```
| stage | delta_h | status |
|---|---|---|
| Origin departure | 0.00 | observed |
| Ocean transit | -2.60 | observed |
| Port & customs | 3.24 | expected |
| Inland delivery | 1.16 | expected |

Actual final delay: +34.1 h
Simulator ground-truth attribution (hours; the simulator's own counterfactual, not real-world causality): {"true_phi_origin": 0.0, "true_phi_weather": 0.3, "true_phi_route_noise": -2.5, "true_phi_congestion": 0.9, "true_phi_processing": 1.3, "true_phi_inspection": 28.5, "true_phi_inland": 5.5}
