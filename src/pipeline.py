"""End-to-end ENSO reproduction pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import generate_baselines
from .figures import generate_figures
from .metrics import evaluate_models, scale_model_output
from .model import default_parameters, simulate_jinpp, with_updates


@dataclass(frozen=True)
class ReproductionConfig:
    """Runtime and sampling choices for a complete reproduction."""

    mode: str = "full"
    reference_years: float = 1650.0
    reference_burn: float = 100.0
    ablation_years: float = 900.0
    ablation_burn: float = 100.0
    sensitivity_years: float = 700.0
    sensitivity_burn: float = 100.0
    perturbation_years: float = 12.0
    dt: float = 1.0 / 48.0
    reference_seed: int = 20260614
    experiment_seed: int = 20260615

    @classmethod
    def for_mode(cls, mode: str) -> "ReproductionConfig":
        if mode == "full":
            return cls(mode="full")
        if mode == "quick":
            return cls(
                mode="quick",
                reference_years=250.0,
                reference_burn=20.0,
                ablation_years=140.0,
                ablation_burn=20.0,
                sensitivity_years=120.0,
                sensitivity_burn=20.0,
                perturbation_years=8.0,
            )
        raise ValueError("mode must be 'quick' or 'full'")


def load_observations(path: str | Path) -> pd.DataFrame:
    """Load and validate the frozen monthly observational input."""
    path = Path(path)
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "nino34", "heat"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Observation file lacks columns: {sorted(missing)}")
    if frame[list(required - {"date"})].isna().any().any():
        raise ValueError("Observation target columns contain missing values.")
    expected = pd.date_range(frame["date"].iloc[0], frame["date"].iloc[-1], freq="MS")
    if not np.array_equal(frame["date"].to_numpy(), expected.to_numpy()):
        raise ValueError("Observation dates are not a complete monthly sequence.")
    return frame


def apply_ablation(parameters: dict[str, float], name: str) -> dict[str, float]:
    """Return a copy of ``parameters`` with one advertised mechanism removed."""
    p = dict(parameters)
    if name == "Full model":
        return p
    if name == "No delayed memory":
        p["memory_factor"] = 0.0
    elif name == "No seasonality":
        for key in (
            "RE_season",
            "RC_season",
            "alphaE_season",
            "alphaC_season",
            "q_seasonality",
            "e_seasonality",
            "conv_thresh_season",
        ):
            p[key] = 0.0
    elif name == "No wind bursts":
        p["lambda_q0"] = 0.0
        p["lambda_e0"] = 0.0
    elif name == "No SST quadratic asymmetry":
        p["quadE"] = 0.0
        p["quadC"] = 0.0
    elif name == "No zonal-current SST coupling":
        p["advE"] = 0.0
        p["advC"] = 0.0
    elif name == "Constant-amplitude SST noise":
        p["multi_noise"] = 0.0
    elif name == "No slow mean state":
        p["sigmaM"] = 0.0
        for key in ("RE_m", "RC_m", "conv_m", "lambda_q_m", "lambda_e_m"):
            p[key] = 0.0
    else:
        raise KeyError(name)
    return p


def perturb_parameter_group(
    parameters: dict[str, float], experiment: str, factor: float
) -> dict[str, float]:
    """Scale one named parameter group for common-seed trajectory tests."""
    p = dict(parameters)
    groups = {
        "Thermocline feedback": ("alphaE", "alphaC"),
        "Seasonal growth": ("RE_season", "RC_season"),
        "Wind-burst rate": ("lambda_q0", "lambda_e0"),
        "Delayed memory": ("memory_factor",),
    }
    for key in groups[experiment]:
        p[key] *= factor
    return p


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g")


def _manifest(root: Path, files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "file": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return pd.DataFrame(rows)


def run_reproduction(
    root: str | Path,
    *,
    config: ReproductionConfig | None = None,
    mode: str = "full",
) -> dict[str, object]:
    """Run every simulation, diagnostic, experiment, and figure from scratch."""
    root = Path(root).resolve()
    config = ReproductionConfig.for_mode(mode) if config is None else config
    data_dir = root / "data"
    output_data = root / "output" / "data"
    figure_dir = root / "output" / "figures"
    output_data.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    observations = load_observations(data_dir / "enso_observation_indices_1979_2025.csv")
    parameters = default_parameters()

    memory_off_raw = simulate_jinpp(
        with_updates(parameters, memory_factor=0.0),
        years=config.reference_years,
        burn=config.reference_burn,
        dt=config.dt,
        seed=config.reference_seed,
    )
    memory_on_raw = simulate_jinpp(
        with_updates(parameters, memory_factor=1.0),
        years=config.reference_years,
        burn=config.reference_burn,
        dt=config.dt,
        seed=config.reference_seed,
    )
    memory_off = scale_model_output(memory_off_raw, observations)
    memory_on = scale_model_output(memory_on_raw, observations)

    analysis_months = len(memory_on)
    baselines = generate_baselines(
        observations,
        months=analysis_months,
        seed=config.reference_seed + 100,
    )
    baselines["Linear Jin RO"] = scale_model_output(baselines["Linear Jin RO"], observations)
    models = {
        **baselines,
        "Jin++ (memory off)": memory_off,
        "Jin++ (memory on)": memory_on,
    }
    statistics, errors, scores = evaluate_models(observations, models)

    _write_csv(memory_off, output_data / "jinpp_memory_off_indices.csv")
    _write_csv(memory_on, output_data / "jinpp_memory_on_indices.csv")
    _write_csv(statistics, output_data / "model_statistics.csv")
    _write_csv(errors, output_data / "model_normalised_errors.csv")
    _write_csv(scores, output_data / "model_scores.csv")

    ablation_names = [
        "Full model",
        "No delayed memory",
        "No seasonality",
        "No wind bursts",
        "No SST quadratic asymmetry",
        "No zonal-current SST coupling",
        "Constant-amplitude SST noise",
        "No slow mean state",
    ]
    ablation_models: dict[str, pd.DataFrame] = {}
    for index, name in enumerate(ablation_names):
        raw = simulate_jinpp(
            apply_ablation(parameters, name),
            years=config.ablation_years,
            burn=config.ablation_burn,
            dt=config.dt,
            seed=config.experiment_seed,
        )
        ablation_models[name] = scale_model_output(raw, observations)
    ablation_stats_all, ablation_errors_all, ablation_scores_all = evaluate_models(
        observations, ablation_models
    )
    ablation_stats = ablation_stats_all[ablation_stats_all["model"] != "Observed"].reset_index(drop=True)
    ablation_errors = ablation_errors_all[ablation_errors_all["model"] != "Observed"].reset_index(drop=True)
    ablation_scores = ablation_scores_all[ablation_scores_all["model"] != "Observed"].reset_index(drop=True)
    _write_csv(ablation_stats, output_data / "ablation_statistics.csv")
    _write_csv(ablation_errors, output_data / "ablation_normalised_errors.csv")
    _write_csv(ablation_scores, output_data / "ablation_scores.csv")

    factors = np.linspace(0.0, 1.8, 10)
    sensitivity_models: dict[str, pd.DataFrame] = {}
    sensitivity_names = []
    for factor in factors:
        name = f"Memory factor {factor:.1f}"
        sensitivity_names.append(name)
        raw = simulate_jinpp(
            with_updates(parameters, memory_factor=float(factor)),
            years=config.sensitivity_years,
            burn=config.sensitivity_burn,
            dt=config.dt,
            seed=config.experiment_seed + 1,
        )
        sensitivity_models[name] = scale_model_output(raw, observations)
    sens_stats, sens_errors, sens_scores = evaluate_models(observations, sensitivity_models)
    obs_stats = statistics.set_index("model").loc["Observed"]
    sens_stats = sens_stats.set_index("model").loc[sensitivity_names]
    sens_scores = sens_scores.set_index("model").loc[sensitivity_names]
    sensitivity = pd.DataFrame(
        {
            "model": sensitivity_names,
            "memory_factor": factors,
            "core_rms_score": sens_scores["core_rms_score"].to_numpy(float),
            "recharge_rms_score": sens_scores["recharge_rms_score"].to_numpy(float),
            "L_to_L": sens_stats["L_to_L"].to_numpy(float),
            "L_to_L_error": np.abs(sens_stats["L_to_L"].to_numpy(float) - obs_stats["L_to_L"]),
            "ac12": sens_stats["ac12"].to_numpy(float),
            "ac12_error": np.abs(sens_stats["ac12"].to_numpy(float) - obs_stats["ac12"]),
            "spectrum_error": sens_stats["spectrum_error"].to_numpy(float),
            "period": sens_stats["period"].to_numpy(float),
        }
    )
    _write_csv(sensitivity, output_data / "memory_sensitivity.csv")

    perturbation_rows = []
    for experiment in (
        "Thermocline feedback",
        "Seasonal growth",
        "Wind-burst rate",
        "Delayed memory",
    ):
        for factor in (0.7, 1.0, 1.3):
            raw = simulate_jinpp(
                perturb_parameter_group(parameters, experiment, factor),
                years=config.perturbation_years,
                burn=0.0,
                dt=config.dt,
                seed=config.experiment_seed + 2,
            )
            scaled = scale_model_output(raw, observations)
            for month_index, value in enumerate(scaled["nino34"].to_numpy(float)):
                perturbation_rows.append(
                    {
                        "experiment": experiment,
                        "factor": factor,
                        "month_index": month_index,
                        "year": month_index / 12.0,
                        "nino34": value,
                    }
                )
    perturbations = pd.DataFrame(perturbation_rows)
    _write_csv(perturbations, output_data / "parameter_perturbation_trajectories.csv")

    figures = generate_figures(
        observations=observations,
        models=models,
        statistics=statistics,
        errors=errors,
        scores=scores,
        ablation_errors=ablation_errors,
        ablation_scores=ablation_scores,
        sensitivity=sensitivity,
        perturbations=perturbations,
        directory=figure_dir,
    )

    summary = {
        "config": asdict(config),
        "observation_months": len(observations),
        "reference_model_months": analysis_months,
        "figure_count": len(figures),
        "best_core_model": scores.loc[
            scores["model"] != "Observed"
        ].sort_values("core_rms_score").iloc[0]["model"],
        "scores": scores.to_dict(orient="records"),
    }
    summary_path = output_data / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generated = sorted(output_data.glob("*")) + sorted(figure_dir.glob("*"))
    manifest = _manifest(root, [path for path in generated if path.name != "reproduction_manifest.csv"])
    manifest_path = output_data / "reproduction_manifest.csv"
    _write_csv(manifest, manifest_path)
    return {
        "config": config,
        "observations": observations,
        "models": models,
        "statistics": statistics,
        "errors": errors,
        "scores": scores,
        "ablation_statistics": ablation_stats,
        "ablation_errors": ablation_errors,
        "ablation_scores": ablation_scores,
        "sensitivity": sensitivity,
        "perturbations": perturbations,
        "figures": figures,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
    }
