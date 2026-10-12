# Shipment ETA Intelligence

A machine learning prototype that predicts shipment arrival delays, estimates delay risk, and explains how delays accumulate across shipment stages.

The model updates its prediction at three checkpoints: departure, port arrival, and customs clearance. A Streamlit app provides the interface.

**Note:** The project uses synthetic data. Results demonstrate performance on the simulator, not real-world accuracy.

## Features

- Predicts final shipment delay at each checkpoint.
- Models delay propagation across transit, port, customs, and inland delivery.
- Estimates an 80% prediction interval and probability of being more than 12 hours late.
- Provides stage-wise delay explanations and what-if scenarios.
- Evaluates models using cross-validation, unseen-route testing, and time-based splits.

## Architecture

```text
Synthetic Data
      ↓
Feature Engineering
      ↓
ETA Prediction Model
      ↓
ETA, Delay Risk, Prediction Interval
      ↓
Streamlit App
```

The main model is a stage-wise LightGBM chain. It predicts delay increments for each stage and uses Monte Carlo simulation to estimate the remaining delay.

## Project Structure

| File | Purpose |
|---|---|
| `src/generator.py` | Generates synthetic shipment data |
| `src/features.py` | Feature engineering and missing-data handling |
| `src/models.py` | Baselines, Ridge, LightGBM, and chain model |
| `src/inputs.py` | Input validation and checkpoint handling |
| `src/scenario.py` | What-if scenarios |
| `src/explain.py` | Delay breakdown and SHAP diagnostics |
| `src/evaluate.py` | Model evaluation and robustness tests |
| `src/pipeline.py` | Runs experiments and generates results |
| `app/demo.py` | Streamlit application |

## Setup

Requires Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline
streamlit run app/demo.py
```

Optional commands:

```bash
python -m src.generator
python -m src.multiseed
```

## Results

Results from 300 synthetic shipments (seed 42), measured using MAE in hours:

| Checkpoint | Carry-forward | LightGBM chain (median) |
|---|---:|---:|
| Departure | 12.55 | **10.40** |
| Port arrival | 11.72 | **7.32** |
| Customs cleared | 4.65 | **3.45** |

The chain model achieved lower MAE than the carry-forward baseline at all three checkpoints. Its 80% prediction interval covered approximately 81%, 79%, and 74% of outcomes, respectively.

## Limitations

- Data and delay causes are simulated.
- Real-world accuracy has not been established.
- Predictions depend on the available route, weather, and congestion inputs.
- Inland delays are difficult to predict before the shipment reaches that stage.
- The app supports only the predefined routes.

## Outputs

The evaluation pipeline generates metrics, a summary JSON file, and figures under `outputs/`.
