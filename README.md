# Teaching notebooks

Runnable notebooks for teaching meteorology, machine learning, and data science.

## XGBoost from first principles

[Open the notebook on GitHub](./xgboost_from_first_principles_meteorology.ipynb) or [launch it in Google Colab](https://colab.research.google.com/github/kieranmrhunt/teaching-notebooks/blob/main/xgboost_from_first_principles_meteorology.ipynb).

This guided notebook builds a compact XGBoost-like learner from the mathematics upwards, then uses simulated weather-station data to explore:

- gradients, Hessians, leaf scores, and split gain;
- regularisation, shrinkage, sampling, and histogram-like splits;
- learnt missing-value directions;
- regression, classification, and a custom asymmetric loss;
- monotonic, interaction, categorical, and cyclic constraints;
- feature importance, partial dependence, and SHAP values;
- time-aware validation and leakage.

The notebook is deliberately more readable than production XGBoost while retaining the core ideas. It was executed top-to-bottom with Python 3.11; outputs are retained and the meteorological dataset is generated within the notebook.

## Nonlinear stochastic ENSO oscillator

[Open the notebook on GitHub](./enso-oscillator.ipynb).

This notebook reproduces the Jin, Jin++, and Jin++ delayed-memory ENSO experiments, including the diagnostic scorecards, mechanism ablations, parameter sensitivities, and 50,000-year extreme-event controls. The frozen 1979--2025 observational input is stored in `data/`, and the reusable simulator and diagnostic code is stored in `src/`.

For an isolated ENSO environment, install `enso-oscillator-requirements.txt`, start Jupyter in the repository root, and run `enso-oscillator.ipynb` from top to bottom. Set `MODE = "quick"` for a short smoke test or `MODE = "full"` for the manuscript settings. The extreme-event section has separate run-length controls and uses 50,000 retained years per model by default.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Most of the notebook remains useful without the optional `xgboost` and `shap` comparisons, but the requirements file installs the complete teaching environment.

## Author

Kieran Hunt, Department of Meteorology, University of Reading.
