# Machine and Influence Learning

Research-grade ML experiments spanning computer vision, NLP, ensembling, anomaly detection, and random seed sensitivity analysis.

## Problem

Build intuition for model behavior under varying data, hyperparameters, and random seeds while maintaining reusable training code.

## Approach

### Notebook pipeline

| Notebook | Focus |
|----------|-------|
| 00_pilot.ipynb | Experiment setup |
| 01_crossval.ipynb | Cross-validation strategy |
| 06_NLP.ipynb | Text classification |
| 11_RandomSeed_influence.ipynb | Seed sensitivity analysis |

### Structured module

```
Bi-LSTM/
  train.py      Training entry point
  engine.py     Training loop
  lstm.py       Model definition
```

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Train Bi-LSTM:

```bash
cd Bi-LSTM && python train.py
```

## Tech stack

Python 3, PyTorch, TensorFlow (notebook-specific), scikit-learn, Jupyter

## Design principles

- Fix random seeds and document them per experiment
- Separate data prep, training, and evaluation cells
- Prefer modular scripts over monolithic notebooks for production paths

## Limitations and next steps

- Unify Bi-LSTM on a single DL framework
- Add MLflow experiment tracking
- Move datasets to external storage with download scripts
