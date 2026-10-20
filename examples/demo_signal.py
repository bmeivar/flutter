"""Synthetic flutter-approach signal for the README demo.

A single aeroelastic mode (4.4 Hz) is driven by turbulence (white noise):

    x'' + 2*zeta(t)*w*x' + w^2*x + beta*x^2*x' = sigma*noise(t)

The linear damping zeta(t) is constant during cruise, then decreases with
airspeed (time) and crosses zero at
``onset_time`` (Hopf bifurcation = flutter onset). The nonlinear damping term
(Van der Pol type) saturates the post-flutter response into a limit-cycle
oscillation. Sensor noise is added to the sampled response.
"""

import numpy as np


def flutter_approach_signal(
    duration=60.0,
    fs=100.0,
    freq=4.4,
    zeta_start=0.08,
    ramp_start=20.0,
    onset_time=45.0,
    lco_ratio=8.0,
    sensor_noise=0.2,
    substeps=20,
    seed=0,
):
    """
    :return: t [s], measured signal, linear damping ratio zeta(t)
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    dt = 1.0 / (fs * substeps)
    w = 2 * np.pi * freq
    t = np.arange(n) / fs
    # constant damping (cruise) until ramp_start, then decreasing with airspeed
    zeta = zeta_start * np.clip((onset_time - t) / (onset_time - ramp_start), None, 1.0)

    # nominal (t=0) stationary std for unit-intensity forcing
    nominal_std = 1.0 / np.sqrt(4 * zeta_start * w**3)
    # LCO amplitude at the end of the record ~ lco_ratio * nominal std
    a_lco = lco_ratio * nominal_std
    beta = 8 * abs(zeta[-1]) * w / a_lco**2

    x, v = 0.0, 0.0
    out = np.zeros(n)
    for k in range(n):
        z = zeta[k]
        for _ in range(substeps):
            acc = -2 * z * w * v - w**2 * x - beta * x**2 * v
            v += acc * dt + rng.normal() * np.sqrt(dt)
            x += v * dt
        out[k] = x
    out /= nominal_std
    out += sensor_noise * rng.normal(size=n)
    return t, out, zeta
