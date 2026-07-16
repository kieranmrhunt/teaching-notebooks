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
