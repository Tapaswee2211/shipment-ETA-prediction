# Shipment ETA Intelligence

Predicts when a shipment will arrive, how likely it is to be late, and where the delay comes from, using **synthetic**
shipment journeys (origin departure → ocean transit → port & customs → inland delivery). The estimate is re-issued at
three checkpoints as the shipment progresses. A Streamlit app (`app/demo.py`) serves the predictions.

> All data are simulated. Results show that the method is sound on a world whose rules we wrote; they are **not** a claim
> about real-world accuracy, and this is a prototype, not a production system.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                             # Python 3.11 / 3.12
python -m src.generator                                     # (re)generate data/shipments_n300_s42.csv  (optional; the app does it on demand)
python -m pytest -q                                         # 67 tests, warnings are errors
python -m src.pipeline                                      # evaluation: outputs/results.md, outputs/summary.json, figures (~3 min)
streamlit run app/demo.py                                   # the app
```

Seed and size live in `src/config.py` (`SEED`, `N_SHIPMENTS`). Everything is deterministic for a given seed.
`python -m src.pipeline --fast` is a quick smoke run; `python -m src.multiseed` checks robustness across seeds.

## Architecture

```
src/generator.py   causal simulator: slack absorption, queueing nonlinearity, missed-slot knock-ons, AR(1) port congestion,
                   non-random missing timestamps, exact Shapley ground truth per delay source
src/features.py    checkpoint-safe features, stage-aware imputer, out-of-fold route priors (all fit/transform objects)
src/models.py      ETAPredictor per checkpoint: baselines, Ridge, LightGBM (+ quantile heads, classifier) and the chain model
src/inputs.py      app input handling: held-out lookup, "as-of" masking, new-shipment validation + feature-row construction
src/scenario.py    what-if inputs that are valid at each checkpoint, deterministic sweeps
src/explain.py     chain stage breakdown + plain-language text (headline), GBM/SHAP attribution (diagnostics)
src/evaluate.py    random / leave-one-route-out / forward-in-time splits, metrics, ablations, missing-data sweep
src/pipeline.py    runs the evaluation and writes results;  src/multiseed.py  seed robustness
src/adapters.py    experimental: map your own CSV onto the schema (CLI/pipeline only; not used by the app)
app/demo.py        Streamlit app
```

### How the prediction works

- **Lateness** `L_k` = actual − scheduled time at milestone k (0 departure, 1 port arrival, 2 customs cleared, 3 delivery).
  Final delay = `L_3`. All timestamps are UTC.
- **Checkpoint k** = what is known at milestone k. `CP0` departure, `CP1` port arrival, `CP2` customs cleared.
- **Target:** the models learn the _remaining_ delay `R_k = L_3 − L_k`.
- **ETA equation:** `predicted arrival = scheduled delivery + L_k + R̂_k`. Every prediction column returned by
  `ETAPredictor.predict()` (`delay_*`, `p10..p90`, `c10..c90`) is already `L_k + R̂_k`, the **total** delay versus schedule;
  the app adds it to the scheduled delivery exactly once (tested in `tests/test_semantics.py`).
- **Headline model, the chain model:** one LightGBM per stage predicts that stage's change in lateness from the state at the
  start of the stage (with monotone constraints: worse weather / a more congested port never reduces the delay added in that
  stage). The remaining stages are simulated 200 times, drawing noise from out-of-fold residuals. From the same simulation:
  - **point estimate = median** of the simulated final delay (lowest MAE of the variants tried; the mean is also reported),
  - **80% range** = 10th–90th percentile,
  - **P(late)** = share of simulated outcomes with final delay > **12 h** (`DELAY_THRESHOLD_H`).
- **Comparators:** trust-the-schedule, carry-the-current-delay-forward, route average, Ridge, gradient-boosted trees (GBM).
  The GBM also has quantile heads and a calibrated classifier, shown only as cross-checks.
- **Stage breakdown (the "why"):** stage k's value is `chain_L_k − chain_L_(k−1)`. Observed stages are measured; stages not yet
  reached are the simulation's _average_. The values sum exactly to the average delay; the headline uses the median, which is
  lower because delays are right-skewed. It is accounting of the model's expectation, not a causal claim.

## The app

```bash
streamlit run app/demo.py
```

**Existing shipment.** Enter a shipment ID (e.g. `SH0100`; unknown IDs show an error) and a checkpoint. The route, schedule and
the observations available _at that checkpoint_ are used; later milestones, realised conditions, the outcome and simulator ground
truth are blanked before the model sees the row. A model is trained **without that shipment** (cached per shipment and
checkpoint). Tick _Reveal the actual outcome_ to compare with what happened.

**New shipment.** Reference ID, origin → destination (the 10 lanes of the route table; transit time, buffers and port
parameters come from it), scheduled departure, checkpoint, and the delays/conditions known at that checkpoint. Input is
validated (ranges, duplicate IDs, unsupported lanes, physically impossible milestone orderings) and converted into one row in
exactly the training schema; the feature contract is checked before prediction. Lanes outside the route table are **rejected**
rather than guessed. Cargo attributes (weight, container type, carrier) are not model features. Tests show that rebuilding an
existing shipment as a "new" one gives an identical feature vector and prediction.

**What-if.** Only inputs that can still change the remaining stages are adjustable: at departure, weather and destination
congestion; at port arrival, destination congestion only (transit is over); at customs clearance none (only the inland leg is
left). Scenarios use common random numbers, so differences come from the inputs, not sampling noise.

**Technical diagnostics** (collapsed): output definitions, the exact input vector, SHAP of the GBM comparator, the simulator's
ground-truth attribution (existing shipments only), and the measured evaluation numbers from `outputs/summary.json`.

## Evaluation

Splits (everything learned — imputer, route priors, scaler, models, residual banks — is fitted on the training part only; every
shipment is one row and stays entirely in train or test; all checkpoints share the same split):

- `random`: 5-fold × 3 repeats; `route`: leave-one-route-out; `time`: expanding window with a delivery embargo (training
  shipments not yet delivered when the test period starts are dropped).
- Metrics: MAE/RMSE/bias of the final arrival time (hours), paired bootstrap CIs over shipments, PR-AUC/Brier for P(late),
  80%-interval coverage.

Measured results (`python -m src.pipeline`, seed 42, n=300; MAE in hours; schedule-only = 14.44):

| checkpoint          | carry delay forward | Ridge | GBM   | chain (mean) | **chain (median) = headline** |
| ------------------- | ------------------- | ----- | ----- | ------------ | ----------------------------- |
| CP0 departure       | 12.55               | 12.22 | 11.57 | 10.97        | **10.40**                     |
| CP1 port arrival    | 11.72               | 8.90  | 8.42  | 7.73         | **7.32**                      |
| CP2 customs cleared | 4.65                | 4.44  | 4.14  | 3.59         | **3.45**                      |

Paired MAE gain of the headline over carry-forward (95% CI): CP0 +2.15 h [0.77, 3.59], CP1 +4.40 h [3.30, 5.67], CP2 +1.20 h [0.87, 1.50].

Across 5 seeds (`python -m src.multiseed`, 5-fold CV; mean ± std): headline 10.28 ± 1.88 / 7.45 ± 1.15 / 3.53 ± 0.60 versus
carry-forward 12.78 ± 1.34 / 12.00 ± 1.42 / 4.80 ± 0.61; it is better in 15/15 seed×checkpoint cells. The forward-in-time
split gives the same 15/15 (and 3/3 on seed 54). On seed 54, forward in time at departure, the GBM comparator (12.50) is _worse_
than carry-forward (11.23) while the headline model (10.26) is not.

Other measured results (seed 42):

- **80% range coverage** (chain): 81% / 79% / 74% at CP0/CP1/CP2 (GBM quantile heads: 64% / 62% / 65%, so they are not shown).
- **P(delay > 12 h)**, base rate 30%: chain Brier 0.17 / 0.13 / 0.07, PR-AUC 0.65 / 0.76 / 0.90. At CP2 the carry-forward rule has
  the best PR-AUC (0.93): once customs has cleared, the delay is already visible.
- **Unseen lanes** (leave-one-route-out, chain mean): 10.74 / 7.29 / 3.45 h versus Ridge 17.98 / 12.34 / 4.34.
- **Explanation vs simulator ground truth** (Spearman, CP1; synthetic only): chain breakdown origin 0.97, transit 0.66, port 0.66,
  inland 0.03 (nothing observable predicts inland); at CP2 port 0.88. At CP0 transit and port are weak for both attribution methods (0.26–0.53), as expected before those stages happen.
- **Ablations:** removing engineered feature groups changes the GBM's mean MAE (over checkpoints) by at most ~0.2 h; the missing-data model matters at CP2
  (4.17 → 4.50 h without it); route priors do not help at n=300 (removing them slightly lowers MAE).

## Assumptions and limitations

- **Synthetic and circular.** The generator's rules are the ground truth and the model is evaluated on its output; weather and
  congestion are 0–1 indices supplied by the user (in practice they would come from feeds). Simulator Shapley attribution is the
  generator's own counterfactual; it says nothing about real-world causality.
- **Median trades bias for MAE:** its bias is −2.0 / −3.1 / −1.4 h (it is earlier than the average outcome because delays are
  skewed); the mean is unbiased but has higher MAE.
- **Interval coverage** at CP2 is 74% against a nominal 80%.
- **What-if curves** are monotone for congestion (worst adjacent dip 0.6 h over 40 shipments) but weather at departure can wiggle
  by up to ~2.5 h because later stage models take the carried delay as input.
- **Stage breakdown:** inland delay is essentially unpredictable; a missing milestone timestamp is imputed and labelled "estimated".
- **Worked example:** the one in `outputs/results.md` (6th delayed shipment, checkpoint 1) is a miss: it arrived 76 h late against
  a median estimate of 31 h, outside the 80% range. About one in five outcomes should fall outside.
- Fixed 4-milestone structure; new shipments restricted to the 10 route-table lanes; the holiday calendar covers 2025 only;
  the departure date and weather are the only calendar/seasonal inputs.

## Feature status

|                        |                                                                                                                                                                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implemented and tested | data generation (seeded), checkpoint features, chain/GBM/Ridge/baselines, existing-shipment held-out flow, new-shipment validation and feature construction, what-if, stage breakdown, evaluation pipeline, multi-seed check |
| Experimental           | `src/adapters.py` bring-your-own-data (`docs/BRING_YOUR_OWN_DATA.md`; CLI/pipeline only, tested on one foreign-format example); GBM classifier (cross-check only); SHAP diagnostics                                          |
| Not supported          | other lanes in the app, cargo attributes, live data feeds, more than 4 milestones, real-world validation                                                                                                                     |

## Tests

`PYTHONPATH="" python -m pytest -q` runs 67 tests with warnings treated as errors (`pytest.ini`): deterministic generation, stage/Shapley
identities, no future information (masked vs unmasked rows give identical features and predictions), prediction semantics
(observed delay added once, breakdown sums to the chain mean), batch-independence of predictions, split integrity, input
validation, new-shipment ≡ existing-shipment equivalence, what-if validity per checkpoint, and Streamlit `AppTest` smoke tests.

More: `docs/RESULTS_GUIDE.md` (reading the pipeline output), `docs/BRING_YOUR_OWN_DATA.md`.
