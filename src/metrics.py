"""ENSO diagnostics with definitions shared by tables and figures."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew


TOLERANCES: dict[str, float] = {
    "skew": 0.30,
    "kurt": 0.50,
    "ac6": 0.12,
    "ac12": 0.12,
    "ac24": 0.15,
    "period": 1.00,
    "enso_power": 0.15,
    "spectrum_error": 0.45,
    "winter_peak": 0.15,
    "phase_lock": 0.15,
    "warm_cold": 0.30,
    "E_to_L": 0.15,
    "L_to_L": 0.15,
    "L_to_E": 0.15,
    "H_leads_T_6m": 0.15,
}

CORE_METRICS = [name for name in TOLERANCES if name != "H_leads_T_6m"]
ALL_METRICS = list(TOLERANCES)
PERIOD_NFFT = 2048


def _affine_match(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    target = np.asarray(target, float)
    return (
        (x - np.mean(x))
        / np.std(x, ddof=0)
        * np.std(target, ddof=0)
        + np.mean(target)
    )


def scale_model_output(frame: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Convert model units consistently to observed anomaly units."""
    result = frame.copy()
    obs_t = observations["nino34"].to_numpy(float)
    obs_h = observations["heat"].to_numpy(float)
    mappings = {
        "nino34_raw": ("nino34", obs_t),
        "east_raw": ("east_sst", obs_t),
        "central_raw": ("central_sst", obs_t),
        "heat_raw": ("heat", obs_h),
        "recharge_raw": ("recharge", obs_h),
    }
    for source, (target, reference) in mappings.items():
        if source in result:
            result[target] = _affine_match(result[source].to_numpy(float), reference)
    return result


def three_month_mean(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centred three-month means and their original monthly indices."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.array([], dtype=float), np.array([], dtype=int)
    return np.convolve(x, np.ones(3) / 3.0, mode="valid"), np.arange(1, len(x) - 1)


def separated_peaks(
    x: np.ndarray,
    *,
    threshold_std: float = 0.75,
    minimum_separation: int = 18,
) -> np.ndarray:
    """Find peaks after the manuscript's centred three-month smoothing."""
    centered = np.asarray(x, float) - np.mean(x)
    smooth, original_index = three_month_mean(centered)
    threshold = threshold_std * np.std(centered, ddof=0)
    candidates: list[int] = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] > threshold and smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]:
            candidate = int(original_index[i])
            if not candidates or candidate - candidates[-1] >= minimum_separation:
                candidates.append(candidate)
            elif smooth[i] > smooth[candidates[-1] - 1]:
                candidates[-1] = candidate
    return np.asarray(candidates, dtype=int)


def peak_months(x: np.ndarray) -> np.ndarray:
    return (separated_peaks(x) % 12 + 1).astype(int)


def spectrum_frequency(
    x: np.ndarray, *, nfft: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    frequency, density = welch(
        x - np.mean(x),
        fs=12.0,
        nperseg=min(256, len(x)),
        nfft=nfft,
        detrend="linear",
        scaling="density",
    )
    positive = frequency > 0
    return frequency[positive], density[positive]


def preferred_period(x: np.ndarray) -> float:
    """Locate the ENSO-band PSD peak on a finer zero-padded FFT grid.

    The 256-month Welch window is unchanged: zero-padding refines the sampled
    peak location but does not add independent spectral resolution.
    """
    frequency, density = spectrum_frequency(x, nfft=PERIOD_NFFT)
    ensoband = (frequency >= 1.0 / 8.0) & (frequency <= 1.0 / 2.0)
    dominant = np.flatnonzero(ensoband)[np.argmax(density[ensoband])]
    return float(1.0 / frequency[dominant])


def spectrum_period(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a correctly transformed density per year of period."""
    frequency, density_frequency = spectrum_frequency(x)
    period = 1.0 / frequency
    density_period = density_frequency / period**2
    order = np.argsort(period)
    return period[order], density_period[order]


def _autocorrelation(x: np.ndarray, lag: int) -> float:
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def _transition_probabilities(x: np.ndarray) -> tuple[float, float, float, float]:
    centered = np.asarray(x, float) - np.mean(x)
    threshold = 0.6 * np.std(centered, ddof=0)
    months = np.arange(len(centered)) % 12 + 1
    years = np.arange(len(centered)) // 12
    states: list[str] = []
    for year in range(1, int(np.max(years)) + 1):
        selection = ((years == year) & np.isin(months, [1, 2])) | (
            (years == year - 1) & (months == 12)
        )
        if np.sum(selection) != 3:
            continue
        value = float(np.mean(centered[selection]))
        states.append("E" if value > threshold else "L" if value < -threshold else "N")
    counts: dict[tuple[str, str], int] = {}
    for first, second in zip(states[:-1], states[1:]):
        counts[(first, second)] = counts.get((first, second), 0) + 1

    def probability(first: str, second: str) -> float:
        denominator = sum(v for (a, _), v in counts.items() if a == first)
        return counts.get((first, second), 0) / denominator if denominator else np.nan

    return (
        probability("E", "E"),
        probability("E", "L"),
        probability("L", "L"),
        probability("L", "E"),
    )


def lead_lag_correlation(
    heat: np.ndarray, temperature: np.ndarray, max_lag: int = 18
) -> tuple[np.ndarray, np.ndarray]:
    heat = np.asarray(heat, float)
    temperature = np.asarray(temperature, float)
    lags = np.arange(-max_lag, max_lag + 1)
    values = []
    for lag in lags:
        if lag < 0:
            value = np.corrcoef(heat[-lag:], temperature[:lag])[0, 1]
        elif lag > 0:
            value = np.corrcoef(heat[:-lag], temperature[lag:])[0, 1]
        else:
            value = np.corrcoef(heat, temperature)[0, 1]
        values.append(value)
    return lags, np.asarray(values)


def metric_vector(
    temperature: np.ndarray,
    heat: np.ndarray | None,
    *,
    reference_spectrum: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    temperature = np.asarray(temperature, float)
    centered = temperature - np.mean(temperature)
    frequency, density = spectrum_frequency(centered)
    ensoband = (frequency >= 1.0 / 8.0) & (frequency <= 1.0 / 2.0)
    interannual = (frequency >= 1.0 / 10.0) & (frequency <= 1.0)
    total_power = float(np.trapezoid(density, frequency))
    enso_power = float(np.trapezoid(density[ensoband], frequency[ensoband]) / total_power)

    ref_frequency, ref_density = reference_spectrum
    ref_select = (ref_frequency >= 1.0 / 10.0) & (ref_frequency <= 1.0)
    model_density = np.interp(ref_frequency[ref_select], frequency, density)
    model_density /= np.trapezoid(model_density, ref_frequency[ref_select])
    target_density = ref_density[ref_select].copy()
    target_density /= np.trapezoid(target_density, ref_frequency[ref_select])
    spectrum_error = float(
        np.sqrt(
            np.mean(
                (np.log(model_density + 1e-12) - np.log(target_density + 1e-12)) ** 2
            )
        )
    )

    warm_indices = separated_peaks(centered)
    cold_indices = separated_peaks(-centered)
    warm_months = warm_indices % 12 + 1
    phase_lock = (
        float(abs(np.mean(np.exp(2j * np.pi * (warm_months - 1) / 12.0))))
        if len(warm_months)
        else np.nan
    )
    smooth, smooth_indices = three_month_mean(centered)
    smooth_lookup = dict(zip(smooth_indices.tolist(), smooth.tolist()))
    warm_amplitude = np.mean([smooth_lookup[i] for i in warm_indices]) if len(warm_indices) else np.nan
    cold_amplitude = np.mean([-smooth_lookup[i] for i in cold_indices]) if len(cold_indices) else np.nan
    e_to_e, e_to_l, l_to_l, l_to_e = _transition_probabilities(centered)
    heat_lead = (
        float(np.corrcoef(np.asarray(heat, float)[:-6], centered[6:])[0, 1])
        if heat is not None
        else np.nan
    )
    years = len(centered) / 12.0
    return {
        "std": float(np.std(centered, ddof=0)),
        "skew": float(skew(centered, bias=True)),
        "kurt": float(kurtosis(centered, fisher=True, bias=True)),
        "ac6": _autocorrelation(centered, 6),
        "ac12": _autocorrelation(centered, 12),
        "ac24": _autocorrelation(centered, 24),
        "period": preferred_period(centered),
        "enso_power": enso_power,
        "spectrum_error": spectrum_error,
        "winter_peak": float(np.mean(np.isin(warm_months, [11, 12, 1]))),
        "phase_lock": phase_lock,
        "warm_cold": float(warm_amplitude / cold_amplitude),
        "E_to_E": e_to_e,
        "E_to_L": e_to_l,
        "L_to_L": l_to_l,
        "L_to_E": l_to_e,
        "H_leads_T_6m": heat_lead,
        "warm_peaks_per_century": float(len(warm_indices) / years * 100.0),
        "cold_peaks_per_century": float(len(cold_indices) / years * 100.0),
    }


def evaluate_models(
    observations: pd.DataFrame,
    models: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute moments, normalised errors, and transparent RMS scores."""
    reference = spectrum_frequency(observations["nino34"].to_numpy(float))
    ordered = {"Observed": observations, **models}
    rows = []
    for name, frame in ordered.items():
        rows.append(
            {
                "model": name,
                **metric_vector(
                    frame["nino34"].to_numpy(float),
                    frame["heat"].to_numpy(float) if "heat" in frame else None,
                    reference_spectrum=reference,
                ),
            }
        )
    statistics = pd.DataFrame(rows)
    observed = statistics.set_index("model").loc["Observed"]
    error_rows = []
    score_rows = []
    for _, row in statistics.iterrows():
        errors = {
            metric: abs(float(row[metric]) - float(observed[metric])) / tolerance
            if np.isfinite(row[metric])
            else np.nan
            for metric, tolerance in TOLERANCES.items()
        }
        core_values = np.asarray([errors[name] for name in CORE_METRICS], float)
        all_values = np.asarray([errors[name] for name in ALL_METRICS], float)
        core_score = float(np.sqrt(np.mean(core_values**2)))
        recharge_score = (
            float(np.sqrt(np.mean(all_values**2))) if np.all(np.isfinite(all_values)) else np.nan
        )
        error_rows.append({"model": row["model"], **errors})
        score_rows.append(
            {
                "model": row["model"],
                "core_rms_score": core_score,
                "recharge_rms_score": recharge_score,
            }
        )
    return statistics, pd.DataFrame(error_rows), pd.DataFrame(score_rows)
