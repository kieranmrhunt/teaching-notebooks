"""Corrected low-order Jin++ ENSO simulator.

The delayed pathway in this module is deliberately called a *wind-forced
memory proxy*.  It is a zero-dimensional cascade of first-order filters; it
does not calculate spatial wind-stress curl, Rossby-wave propagation, or
boundary reflection.

Time is measured in years.  The core nonlinear stochastic tendencies are
evaluated simultaneously from the state at the start of each step.  Linear
relaxation states use exponential updates, avoiding the shortened effective
timescales produced by forward Euler at the original 1/48-year time step.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np
import pandas as pd


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic response."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def default_parameters() -> dict[str, float]:
    """Return the single fixed parameter set used by every mechanism test.

    The values retain the exploratory calibration supplied with the reviewed
    prototype.  The delayed-memory model and the memory-off model differ only
    through ``memory_factor``; unrelated parameters are not retuned.
    """
    return {
        "RC0": -0.734556252924,
        "RC_m": 0.15,
        "RC_season": 1.4,
        "RE0": -0.882364148657,
        "RE_m": 0.25,
        "RE_season": 1.4,
        "advC": 0.52936129491,
        "advE": 0.18426594154,
        "alphaC": 1.31679743945,
        "alphaC_season": 0.35,
        "alphaE": 2.17236624671,
        "alphaE_season": 0.45,
        "chiU": 1.6,
        "conv_E": 0.55,
        "conv_H": 0.25,
        "conv_m": -0.05,
        "conv_thresh0": 0.143411873081,
        "conv_thresh_season": 0.0735466419107,
        "conv_width": 0.745711595068,
        "cubicC": 0.548225807988,
        "cubicE": 1.0476133581,
        "eC": 0.0697920960255,
        "eE": 0.0313423483297,
        "e_cap": 0.91700228504,
        "e_scale": 0.166975737571,
        "e_seasonality": 1.41499599454,
        "e_shape": 1.3,
        "etaC_K": 0.298416178634,
        "etaC_off": 0.1,
        "etaC_tau": 0.0987184331149,
        "etaC_RK": 0.0523,
        "etaE_K": 0.723367517778,
        "etaE_off": 0.15,
        "etaE_tau": 0.547055795805,
        "etaE_RK": 0.1157,
        "flux2C": 0.163310595022,
        "flux2E": 0.258115350211,
        "h0": 0.15,
        "h_scale": 1.01421788838,
        "heat_H": 0.640411191066,
        "heat_K": 1.21823836516,
        "heat_off": 0.461205643585,
        "heat_tau": 0.495528628737,
        "heat_R3": 1.7635,
        "lambda_e0": 0.937129016065,
        "lambda_e_H": 0.4,
        "lambda_e_TC": 0.843463065037,
        "lambda_e_TE": 0.7,
        "lambda_e_const": -1.59445184324,
        "lambda_e_m": 0.2,
        "lambda_q0": 3.30635872178,
        "lambda_q_H": 1.33044413018,
        "lambda_q_TC": 2.88538571339,
        "lambda_q_TE": 1.38796400105,
        "lambda_q_const": 0.00023198971764,
        "lambda_q_m": 0.8,
        "memory_e": 0.0683,
        "memory_e2": 0.1428,
        "memory_factor": 1.0,
        "memory_w": 0.7295,
        "memory_w2": 0.435,
        "mu_HT": 0.183693643417,
        "mu_e": 1.5402,
        "mu_e2": 0.4748,
        "mu_w": 2.72306181279,
        "mu_w2": 0.923497867137,
        "multi_noise": 0.449358798366,
        "nino_E": 0.55,
        "omega_R3": 0.3761,
        "omega_off": 0.07,
        "phase_conv_thresh": 0.25,
        "phase_e": 0.147003596195,
        "phase_growth": 0.778634008717,
        "phase_q": 0.780934363783,
        "phase_thermo": 0.877856322941,
        "qC": 0.613625444102,
        "qE": 0.599991461727,
        "q_cap": 1.71300561333,
        "q_scale": 2.00474885353,
        "q_seasonality": 6.5,
        "q_shape": 1.2,
        "quadC": 0.0895498239701,
        "quadE": 1.2482396895,
        "rH": 0.4025,
        "rN": 0.12,
        "rS": 0.12,
        "rU": 10.0,
        "rho_CE": 0.156537406961,
        "rho_EC": 0.177506053048,
        "sN": 0.2,
        "sR_N": 0.1175,
        "sR_S": 0.1274,
        "sS": 0.2,
        "sigmaC": 0.133121489152,
        "sigmaE": 0.250782747921,
        "sigmaH": 0.0898051587564,
        "sigmaM": 0.08,
        "sigmaN": 0.04,
        "sigmaS": 0.04,
        "sigmaU": 0.02,
        "tau_K": 0.08,
        "tau_R1": 0.2098,
        "tau_R2": 0.479,
        "tau_R3": 1.073,
        "tau_RK": 0.2158,
        "tau_TC": 0.65,
        "tau_TE": 0.35,
        "tau_cubic": 0.2,
        "tau_e": 0.06,
        "tau_linear": 0.617172250387,
        "tau_m": 8.0,
        "tau_q": 0.08,
        "tau_sst": 0.238652767271,
        "windC": 0.121128585507,
        "windE": 0.524211124304,
    }


def with_updates(
    parameters: Mapping[str, float] | None = None,
    **updates: float,
) -> dict[str, float]:
    """Return a copied parameter dictionary with explicit updates."""
    result = deepcopy(default_parameters() if parameters is None else dict(parameters))
    result.update(updates)
    return result


def _seasonal_closure(
    state: Mapping[str, float], t: float, p: Mapping[str, float]
) -> dict[str, float]:
    """Compute deterministic atmospheric closure and event rates."""
    te, tc, h, m = state["TE"], state["TC"], state["H"], state["m"]
    growth = np.cos(2.0 * np.pi * (t - p["phase_growth"]))
    thermo = np.cos(2.0 * np.pi * (t - p["phase_thermo"]))
    q_season = max(
        0.02,
        1.0 + p["q_seasonality"] * np.cos(2.0 * np.pi * (t - p["phase_q"])),
    )
    e_season = max(
        0.02,
        1.0 + p["e_seasonality"] * np.cos(2.0 * np.pi * (t - p["phase_e"])),
    )
    threshold = (
        p["conv_thresh0"]
        + p["conv_thresh_season"]
        * np.cos(2.0 * np.pi * (t - p["phase_conv_thresh"]))
        + p["conv_m"] * m
    )
    conv = sigmoid((tc + p["conv_E"] * te + p["conv_H"] * h - threshold) / p["conv_width"])
    conv0 = sigmoid(-threshold / p["conv_width"])
    c = float(conv - conv0)
    tau_det = (
        p["tau_linear"] * c
        - p["tau_cubic"] * c**3
        + p["tau_sst"] * (p["tau_TC"] * tc + p["tau_TE"] * te)
    )
    rate_q = p["lambda_q0"] * q_season * sigmoid(
        p["lambda_q_const"]
        + p["lambda_q_TC"] * tc
        + p["lambda_q_TE"] * te
        + p["lambda_q_H"] * h
        + p["lambda_q_m"] * m
    )
    rate_e = p["lambda_e0"] * e_season * sigmoid(
        p["lambda_e_const"]
        - p["lambda_e_TC"] * tc
        - p["lambda_e_TE"] * te
        - p["lambda_e_H"] * h
        - p["lambda_e_m"] * m
    )
    return {
        "growth": float(growth),
        "thermo": float(thermo),
        "tau_det": float(tau_det),
        "rate_q": float(max(rate_q, 0.0)),
        "rate_e": float(max(rate_e, 0.0)),
    }


def _advance_jump_state(
    value: float,
    decay_time: float,
    rate: float,
    shape: float,
    scale: float,
    cap: float,
    dt: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    """Advance a decaying compound-Poisson state exactly within one step.

    Jump times are uniform conditional on the Poisson count.  The returned
    average is the time-mean state over the interval and is used as forcing.
    """
    count = int(rng.poisson(rate * dt))
    decay = float(np.exp(-dt / decay_time))
    end_value = value * decay
    average = value * decay_time * (1.0 - decay) / dt
    amplitude_sum = 0.0
    if count:
        event_times = rng.uniform(0.0, dt, size=count)
        amplitudes = np.minimum(rng.gamma(shape, scale, size=count), cap)
        remaining = dt - event_times
        event_decay = np.exp(-remaining / decay_time)
        end_value += float(np.sum(amplitudes * event_decay))
        average += float(
            np.sum(amplitudes * decay_time * (1.0 - event_decay) / dt)
        )
        amplitude_sum = float(np.sum(amplitudes))
    return end_value, average, amplitude_sum, count


def _exact_relax(value: float, target: float, decay_time: float, dt: float) -> float:
    decay = np.exp(-dt / decay_time)
    return float(decay * value + (1.0 - decay) * target)


def _exact_ou(
    value: float,
    target: float,
    rate: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> float:
    decay = np.exp(-rate * dt)
    variance = sigma**2 * (1.0 - decay**2) / (2.0 * rate)
    return float(decay * value + (1.0 - decay) * target + np.sqrt(variance) * rng.normal())


def _memory_source(tau: float, p: Mapping[str, float]) -> float:
    westerly = max(tau, 0.0)
    easterly = max(-tau, 0.0)
    return float(
        p["memory_factor"]
        * (
            -p["memory_w"] * westerly
            - p["memory_w2"] * westerly**2
            + p["memory_e"] * easterly
            + p["memory_e2"] * easterly**2
        )
    )


def _instantaneous_diagnostics(
    state: Mapping[str, float], t: float, p: Mapping[str, float]
) -> dict[str, float]:
    closure = _seasonal_closure(state, t, p)
    tau = closure["tau_det"] + state["q"] - state["e"]
    offmean = 0.5 * (state["HN"] + state["HS"])
    nino34 = p["nino_E"] * state["TE"] + (1.0 - p["nino_E"]) * state["TC"]
    heat = (
        p["heat_H"] * state["H"]
        + p["heat_K"] * state["K"]
        + p["heat_tau"] * tau
        + p["heat_off"] * offmean
        + p["heat_R3"] * state["R3"]
    )
    return {
        "nino34_raw": float(nino34),
        "east_raw": float(state["TE"]),
        "central_raw": float(state["TC"]),
        "heat_raw": float(heat),
        "recharge_raw": float(state["H"]),
        "tau": float(tau),
        "memory_source": _memory_source(tau, p),
    }


def simulate_jinpp(
    parameters: Mapping[str, float] | None = None,
    *,
    years: float = 1650.0,
    burn: float = 100.0,
    dt: float = 1.0 / 48.0,
    seed: int = 20260614,
) -> pd.DataFrame:
    """Integrate the corrected Jin++ stochastic model and sample monthly.

    ``memory_factor=0`` gives the no-delayed-memory model using exactly the
    same remaining parameter values.  ``memory_factor=1`` gives the reference
    delayed-memory model.
    """
    p = default_parameters() if parameters is None else dict(parameters)
    if dt <= 0 or years <= 0 or burn < 0 or burn >= years:
        raise ValueError("Require dt > 0 and 0 <= burn < years.")
    steps_per_month = int(round((1.0 / 12.0) / dt))
    if not np.isclose(steps_per_month * dt, 1.0 / 12.0, atol=1e-12):
        raise ValueError("dt must divide one month exactly.")

    rng = np.random.default_rng(seed)
    n_steps = int(round(years / dt))
    sqrt_dt = np.sqrt(dt)
    state = {
        name: 0.0
        for name in (
            "TE",
            "TC",
            "H",
            "HN",
            "HS",
            "U",
            "K",
            "q",
            "e",
            "m",
            "R1",
            "R2",
            "R3",
            "RK",
        )
    }
    records: list[dict[str, float]] = []
    month_burst_q = 0.0
    month_burst_e = 0.0
    month_count_q = 0
    month_count_e = 0

    for step in range(n_steps):
        t0 = step * dt
        tm = t0 + 0.5 * dt
        old = state.copy()
        closure = _seasonal_closure(old, tm, p)

        q_new, q_mean, burst_q, count_q = _advance_jump_state(
            old["q"],
            p["tau_q"],
            closure["rate_q"],
            p["q_shape"],
            p["q_scale"],
            p["q_cap"],
            dt,
            rng,
        )
        e_new, e_mean, burst_e, count_e = _advance_jump_state(
            old["e"],
            p["tau_e"],
            closure["rate_e"],
            p["e_shape"],
            p["e_scale"],
            p["e_cap"],
            dt,
            rng,
        )
        month_burst_q += burst_q
        month_burst_e += burst_e
        month_count_q += count_q
        month_count_e += count_e

        tau = closure["tau_det"] + q_mean - e_mean
        westerly = max(tau, 0.0)
        easterly = max(-tau, 0.0)
        source = _memory_source(tau, p)
        offmean = 0.5 * (old["HN"] + old["HS"])
        h_e = (
            old["H"]
            + p["etaE_K"] * old["K"]
            + p["etaE_tau"] * tau
            + p["etaE_off"] * (offmean - old["H"])
            + p["etaE_RK"] * old["RK"]
        )
        h_c = (
            old["H"]
            + p["etaC_K"] * old["K"]
            + p["etaC_tau"] * tau
            + p["etaC_off"] * (offmean - old["H"])
            + p["etaC_RK"] * old["RK"]
        )
        phi_e = np.tanh((h_e + p["h0"]) / p["h_scale"]) - np.tanh(
            p["h0"] / p["h_scale"]
        )
        phi_c = np.tanh((h_c + p["h0"]) / p["h_scale"]) - np.tanh(
            p["h0"] / p["h_scale"]
        )
        re = p["RE0"] + p["RE_season"] * closure["growth"] + p["RE_m"] * old["m"]
        rc = p["RC0"] + p["RC_season"] * closure["growth"] + p["RC_m"] * old["m"]
        alpha_e = p["alphaE"] * (1.0 + p["alphaE_season"] * closure["thermo"])
        alpha_c = p["alphaC"] * (1.0 + p["alphaC_season"] * closure["thermo"])

        d_te = (
            re * old["TE"]
            + alpha_e * phi_e
            + p["rho_EC"] * (old["TC"] - old["TE"])
            + p["advE"] * old["U"] * (old["TC"] - old["TE"])
            + p["windE"] * old["K"]
            + p["qE"] * q_mean
            - p["eE"] * e_mean
            + p["quadE"] * old["TE"] ** 2
            - p["flux2E"] * abs(old["TE"]) * old["TE"]
            - p["cubicE"] * old["TE"] ** 3
        )
        d_tc = (
            rc * old["TC"]
            + alpha_c * phi_c
            + p["rho_CE"] * (old["TE"] - old["TC"])
            + p["advC"] * old["U"]
            + p["windC"] * old["K"]
            + p["qC"] * q_mean
            - p["eC"] * e_mean
            + p["quadC"] * old["TC"] ** 2
            - p["flux2C"] * abs(old["TC"]) * old["TC"]
            - p["cubicC"] * old["TC"] ** 3
        )
        d_h = (
            -p["rH"] * old["H"]
            - p["mu_w"] * westerly
            - p["mu_w2"] * westerly**2
            + p["mu_e"] * easterly
            + p["mu_e2"] * easterly**2
            - p["mu_HT"] * old["H"] * tau
            + p["omega_off"] * (offmean - old["H"])
            + p["omega_R3"] * old["R3"]
        )

        # Simultaneous Euler--Maruyama update for the nonlinear core.
        noise_scale = 1.0 + p["multi_noise"] * max(old["TC"], 0.0)
        state["TE"] = old["TE"] + d_te * dt + p["sigmaE"] * noise_scale * sqrt_dt * rng.normal()
        state["TC"] = old["TC"] + d_tc * dt + p["sigmaC"] * noise_scale * sqrt_dt * rng.normal()
        state["H"] = old["H"] + d_h * dt + p["sigmaH"] * sqrt_dt * rng.normal()

        # Exact/exponential updates for linear relaxation states, using old
        # upstream states so every tendency is evaluated at a common time.
        state["K"] = _exact_relax(old["K"], tau, p["tau_K"], dt)
        state["U"] = _exact_ou(
            old["U"],
            p["chiU"] * tau / p["rU"],
            p["rU"],
            p["sigmaU"],
            dt,
            rng,
        )
        state["HN"] = _exact_ou(
            old["HN"],
            old["H"] + (p["sN"] * tau + p["sR_N"] * source) / p["rN"],
            p["rN"],
            p["sigmaN"],
            dt,
            rng,
        )
        state["HS"] = _exact_ou(
            old["HS"],
            old["H"] + (p["sS"] * tau + p["sR_S"] * source) / p["rS"],
            p["rS"],
            p["sigmaS"],
            dt,
            rng,
        )
        state["R1"] = _exact_relax(old["R1"], source, p["tau_R1"], dt)
        state["R2"] = _exact_relax(old["R2"], old["R1"], p["tau_R2"], dt)
        state["R3"] = _exact_relax(old["R3"], old["R2"], p["tau_R3"], dt)
        state["RK"] = _exact_relax(old["RK"], old["R3"], p["tau_RK"], dt)
        state["m"] = _exact_ou(old["m"], 0.0, 1.0 / p["tau_m"], p["sigmaM"], dt, rng)
        state["q"] = q_new
        state["e"] = e_new

        if not np.all(np.isfinite(list(state.values()))):
            raise FloatingPointError(f"Non-finite state at step {step}.")

        t1 = (step + 1) * dt
        if (step + 1) % steps_per_month == 0:
            if t1 > burn + 1e-12:
                diag = _instantaneous_diagnostics(state, t1, p)
                records.append(
                    {
                        "month_index": len(records),
                        **diag,
                        "q": state["q"],
                        "e": state["e"],
                        "m": state["m"],
                        "zonal_current": state["U"],
                        "kelvin": state["K"],
                        "north_recharge": state["HN"],
                        "south_recharge": state["HS"],
                        "offequatorial_mean": 0.5 * (state["HN"] + state["HS"]),
                        "memory_1": state["R1"],
                        "memory_2": state["R2"],
                        "memory_return": state["R3"],
                        "kelvin_return": state["RK"],
                        "burst_q_amplitude": month_burst_q,
                        "burst_e_amplitude": month_burst_e,
                        "burst_q_count": month_count_q,
                        "burst_e_count": month_count_e,
                    }
                )
            month_burst_q = 0.0
            month_burst_e = 0.0
            month_count_q = 0
            month_count_e = 0

    return pd.DataFrame.from_records(records)
