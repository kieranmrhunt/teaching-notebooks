"""Long-control ENSO event catalogues and retrospective precursor diagnostics.

Return periods here are exposure / separated warm-event counts, not monthly
tail probabilities. All temperature mappings are fitted on separate controls.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

from .baselines import simulate_linear_jin
from .metrics import separated_peaks, three_month_mean
from .model import simulate_jinpp, with_updates


MODEL_NAMES = ("Jin", "Jin++", "Jin++memory")
VARIABLE_LABELS = {
    "nino34_raw": r"Niño-3.4 SST ($T_N$)",
    "east_raw": r"Eastern SST ($T_E$)",
    "central_raw": r"Central SST ($T_C$)",
    "heat_raw": r"Heat-content proxy ($H_{\mathrm{diag}}$)",
    "recharge_raw": r"Equatorial recharge ($H$)",
    "zonal_current": r"Zonal current ($U$)",
    "kelvin": r"Fast wind filter ($K$)",
    "north_recharge": r"Northern recharge ($H_N$)",
    "south_recharge": r"Southern recharge ($H_S$)",
    "offequatorial_mean": r"Off-equatorial recharge ($H_o$)",
    "tau": r"Net wind forcing ($\tau$)",
    "q": r"Westerly burst state ($q$)",
    "e": r"Easterly burst state ($e$)",
    "burst_q_amplitude": r"Westerly monthly burst input ($\sum A_q\,\Delta N_q$)",
    "burst_e_amplitude": r"Easterly monthly burst input ($\sum A_e\,\Delta N_e$)",
    "m": r"Slow background state ($m$)",
    "memory_source": r"Memory source ($F_R$)",
    "memory_1": r"Memory filter ($R_1$)",
    "memory_2": r"Memory filter ($R_2$)",
    "memory_return": r"Delayed memory ($R_3$)",
    "kelvin_return": r"Filtered return ($R_K$)",
}


def simulate_control(model, parameters, *, years, burn, dt, seed, cache_dir=None):
    """Return retained whole years, with optional source-keyed numerical cache.

    ``years`` excludes burn-in (unlike simulate_jinpp). Each seed starts its
    own integration. The linear Jin model starts in its stationary law.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    if years < 1 or int(years) != years or burn < 0 or int(burn) != burn:
        raise ValueError("years and burn must be whole years; years must be positive.")
    source_dir = Path(__file__).parent
    metadata = {
        "schema": 1, "model": model, "parameters": dict(parameters),
        "retained_years": int(years), "burn_years": int(burn),
        "dt": float(dt), "seed": int(seed), "numpy": np.__version__,
        "sources": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                    for name in ("model.py", "baselines.py")},
    }
    encoded = json.dumps(metadata, sort_keys=True)
    key = hashlib.sha256(encoded.encode()).hexdigest()[:20]
    path = None if cache_dir is None else Path(cache_dir) / f"{model}_{key}.npz"
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["metadata"]) != encoded:
                raise ValueError(f"Cache metadata mismatch: {path}")
            frame = pd.DataFrame(saved["values"], columns=saved["columns"])
    else:
        if model == "Jin":
            frame = simulate_linear_jin(int(12 * (years + burn)), seed=seed)
            frame = frame.iloc[int(12 * burn):].reset_index(drop=True)
        else:
            p = with_updates(parameters, memory_factor=float(model == "Jin++memory"))
            frame = simulate_jinpp(p, years=years + burn, burn=burn, dt=dt, seed=seed)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, values=frame.to_numpy(),
                                columns=frame.columns.to_numpy(dtype=str), metadata=encoded)
            temporary.replace(path)
    if len(frame) != years * 12 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Invalid control output for {model}, seed {seed}")
    return frame


def temperature_mapping(calibration, observations):
    """Fit the project's mean/SD mapping once, on an independent control."""
    raw = calibration["nino34_raw"].to_numpy(float)
    observed = observations["nino34"].to_numpy(float)
    scale = observed.std() / raw.std()
    return {"offset": float(observed.mean() - raw.mean() * scale),
            "scale": float(scale), "calibration_raw_mean": float(raw.mean()),
            "calibration_raw_sd": float(raw.std())}


def event_catalogue(frame, mapping, *, seed, minimum_separation=18):
    """Use the manuscript's 0.75-SD, centred-three-month warm peak definition.

    A local peak needs two months on either side (smoothing plus neighbours),
    so the exposure omits the first and last two monthly centres of each run.
    """
    x = frame["nino34_raw"].to_numpy(float) * mapping["scale"] + mapping["offset"]
    peaks = separated_peaks(x, minimum_separation=minimum_separation)
    smoothed, _ = three_month_mean(x)
    return pd.DataFrame({
        "seed": int(seed), "month_index": peaks, "year": (peaks + 0.5) / 12,
        "calendar_month": peaks % 12 + 1, "peak_c": smoothed[peaks - 1],
    })


def empirical_return_curve(events, exposure_years):
    """Temperature versus mean event recurrence; tied levels count together."""
    levels, counts = np.unique(events["peak_c"].to_numpy(), return_counts=True)
    exceedances = np.cumsum(counts[::-1])[::-1]
    return pd.DataFrame({"level_c": levels, "exceedances": exceedances,
                         "return_years": exposure_years / exceedances})


def empirical_return_level(events, exposure_years, return_years):
    """Linearly interpolate the empirical recurrence curve; never extrapolate."""
    curve = empirical_return_curve(events, exposure_years)
    if curve.empty or return_years < curve.return_years.min() or return_years > curve.return_years.max():
        return np.nan  # Never extrapolate beyond the sampled event catalogue.
    return float(np.interp(return_years, curve.return_years, curve.level_c))


def recurrence_at_threshold(events, exposure_years, threshold_c, confidence=0.95):
    """Fixed-threshold recurrence and Poisson sampling interval.

    The interval is exact for a homogeneous Poisson count, an approximation
    for clustered ENSO. Zero events yield a one-sided lower recurrence bound.
    """
    count = int((events.peak_c >= threshold_c).sum())
    alpha = 1 - confidence
    if count:
        rate_low = chi2.ppf(alpha / 2, 2 * count) / (2 * exposure_years)
        rate_high = chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / (2 * exposure_years)
        return_low, return_high = 1 / rate_high, 1 / rate_low
    else:
        return_low, return_high = exposure_years / -np.log(alpha), np.inf
    return {"threshold_c": threshold_c, "events": count,
            "exposure_years": exposure_years,
            "return_years": exposure_years / count if count else np.nan,
            "return_low": return_low, "return_high": return_high,
            "interval": f"Poisson {confidence:.0%} two-sided" if count else f"Poisson {confidence:.0%} one-sided lower bound"}


def block_return_levels(events, run_months, return_periods, *, block_years=200, draws=400, seed=817):
    """Resample whole time-block event catalogues, keeping within-block clusters.

    The original events are detected once per independent run. No detection
    or smoothing crosses a run boundary. Each resampled block brings its own
    exposure, including the small edge correction. Intervals describe Monte
    Carlo sampling at fixed parameters/scaling, not structural uncertainty.
    """
    if block_years < 1 or draws < 1:
        raise ValueError("Need positive block_years and draws")
    blocks = []
    width = int(block_years * 12)
    for run_seed, months in run_months.items():
        run = events[events.seed == run_seed]
        for start in range(0, months, width):
            end = min(start + width, months)
            exposure = max(0, min(end, months - 2) - max(start, 2)) / 12
            if exposure:
                peaks = run.loc[(run.month_index >= start) & (run.month_index < end), "peak_c"].to_numpy()
                blocks.append((peaks, exposure))
    rng = np.random.default_rng(seed)
    samples = np.full((draws, len(return_periods)), np.nan)
    for i in range(draws):
        chosen = rng.integers(len(blocks), size=len(blocks))
        levels = np.concatenate([blocks[j][0] for j in chosen])
        exposure = sum(blocks[j][1] for j in chosen)
        sample = pd.DataFrame({"peak_c": levels})
        for k, period in enumerate(return_periods):
            samples[i, k] = empirical_return_level(sample, exposure, period)
    total_exposure = sum(b[1] for b in blocks)
    rows = []
    for k, period in enumerate(return_periods):
        valid = samples[:, k][np.isfinite(samples[:, k])]
        # Dropping many unsupported replicates would give a misleading,
        # conditional interval for an unresolved tail.
        resolved = len(valid) >= .95 * draws
        rows.append({"return_years": period,
                     "level_c": empirical_return_level(events, total_exposure, period),
                     "low_c": np.quantile(valid, .025) if resolved else np.nan,
                     "high_c": np.quantile(valid, .975) if resolved else np.nan,
                     "bootstrap_valid": len(valid), "bootstrap_draws": draws,
                     "blocks": len(blocks), "expected_exceedances": total_exposure / period})
    return pd.DataFrame(rows)


def seasonal_percentiles(runs, variables=None):
    """Pooled, same-calendar-month empirical midrank percentiles per model.

    Ties receive their midrank (important for zero-inflated burst counts).
    Constant variables are omitted instead of being labelled extreme.
    """
    if variables is None:
        variables = [v for v in VARIABLE_LABELS if v in next(iter(runs.values()))]
    lengths = [len(frame) for frame in runs.values()]
    combined = pd.concat([frame[variables] for frame in runs.values()], ignore_index=True)
    months = np.concatenate([np.arange(n) % 12 for n in lengths])
    active = [v for v in variables if combined[v].std(ddof=0) > 1e-12]
    omitted = [v for v in variables if v not in active]
    ranks = np.empty((len(combined), len(active)), dtype=np.float32)
    for month in range(12):
        select = months == month
        for col, variable in enumerate(active):
            ranks[select, col] = 100 * (rankdata(combined.loc[select, variable]) - .5) / select.sum()
    result = {}
    start = 0
    for (run_seed, _), length in zip(runs.items(), lengths):
        result[run_seed] = pd.DataFrame(ranks[start:start + length], columns=active)
        start += length
    return result, omitted


def event_windows(ranks, events, *, before=24, after=12):
    """Stack complete event windows; never wrap across independent runs."""
    variables = next(iter(ranks.values())).columns.tolist()
    lags = np.arange(-before, after + 1)
    windows, kept = [], []
    for row in events.itertuples():
        frame = ranks[row.seed]
        peak = int(row.month_index)
        if peak >= before and peak + after < len(frame):
            windows.append(frame.iloc[peak + lags].to_numpy())
            kept.append(row.Index)
    array = np.stack(windows) if windows else np.empty((0, len(lags), len(variables)))
    return array, lags, variables, events.loc[kept].copy()


def precursor_summary(monster_windows, ordinary_windows, lags, variables):
    """Tail occupancy per event/month, for predeclared time windows.

    Differences from ordinary warm events are retrospective associations.
    No scanning for the most favourable lag and no forecast/causal claims.
    """
    periods = {"12–7 months before": (-12, -7), "6–1 months before": (-6, -1),
               "Peak ±1 month": (-1, 1)}
    rows = []
    if not len(monster_windows):
        return pd.DataFrame()
    for period, (first, last) in periods.items():
        select = (lags >= first) & (lags <= last)
        for col, variable in enumerate(variables):
            values = monster_windows[:, select, col]
            ordinary = ordinary_windows[:, select, col]
            rows.append({
                "window": period, "variable": variable,
                "label": VARIABLE_LABELS.get(variable, variable),
                "events": len(monster_windows), "ordinary_events": len(ordinary_windows),
                "median_percentile": float(np.median(values)),
                "above_p95_pct": float(100 * np.mean(values >= 95)),
                "below_p05_pct": float(100 * np.mean(values <= 5)),
                "ordinary_above_p95_pct": float(100 * np.mean(ordinary >= 95)) if len(ordinary) else np.nan,
                "ordinary_below_p05_pct": float(100 * np.mean(ordinary <= 5)) if len(ordinary) else np.nan,
            })
    return pd.DataFrame(rows)
