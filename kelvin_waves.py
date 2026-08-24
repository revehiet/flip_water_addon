"""Pure-numpy Kelvin ship-wake field (no Blender dependencies).

Shared by the Wake Deformer (mesh displacement) and the Wake Solver
(crest-driven foam particle emission).
"""

import numpy as np


def kelvin_field(x_behind, y_lat, t, amplitude, speed, wave_scale,
                 ray_count, wave_count, decay, wedge_angle, time_scale):
    """Height field of the Kelvin wake in boat-local coordinates.

    x_behind: distance behind the boat along its heading (positive = astern)
    y_lat:    lateral offset (positive = starboard)
    """
    g = 9.81
    k0 = wave_scale * g / max(speed * speed, 0.01)
    h = np.zeros_like(x_behind)
    for j in range(1, int(wave_count) + 1):
        k = k0 * float(j)
        a_j = amplitude / (float(j) ** 1.2)
        for i in range(int(ray_count)):
            th = -0.5 * np.pi + np.pi * i / max(int(ray_count) - 1, 1)
            d = x_behind * np.cos(th) + y_lat * np.sin(th)
            w = np.sqrt(g * k)
            h += a_j * np.cos(k * d - w * t * time_scale)

    # Wake exists only astern of the boat, fading with distance
    h *= np.exp(-x_behind / max(decay, 1e-3))
    h *= (x_behind > 0.0)

    # Near-field ramp: waves need ~half their wavelength to develop. Without
    # it every ray is in phase at the boat and the field spikes (the classic
    # Kelvin superposition singularity). Ramp over half the SHORTEST sampled
    # wavelength so high harmonics develop quickly.
    k_short = k0 * max(int(wave_count), 1)
    ramp_len = 0.5 * (2.0 * np.pi / max(k_short, 1e-6))
    h *= np.clip(x_behind / max(ramp_len, 1e-3), 0.0, 1.0)

    # Lateral taper beyond the Kelvin wedge (smooth falloff)
    wedge = np.tan(np.radians(wedge_angle))
    lat_limit = x_behind * wedge
    margin = 0.3 * lat_limit + 1e-6
    taper = np.clip(1.0 - (np.abs(y_lat) - lat_limit) / margin, 0.0, 1.0)
    h *= taper

    # Hard cap so no single vertex can blow up
    limit = amplitude * max(wave_count, 1) * 2.0
    return np.clip(h, -limit, limit)


def crest_mask_from_field(h, threshold):
    """Boolean mask of wave crests: local maxima (4-neighbourhood) of a
    sampled height field that exceed `threshold`."""
    ny, nx = h.shape
    mask = h >= threshold
    if not mask.any():
        return mask
    crest = mask.copy()
    crest[1:-1, 1:-1] &= (h[1:-1, 1:-1] >= h[:-2, 1:-1])   # above
    crest[1:-1, 1:-1] &= (h[1:-1, 1:-1] >= h[2:, 1:-1])    # below
    crest[1:-1, 1:-1] &= (h[1:-1, 1:-1] >= h[1:-1, :-2])   # left
    crest[1:-1, 1:-1] &= (h[1:-1, 1:-1] >= h[1:-1, 2:])    # right
    # Exclude flat-zero regions (plateaus of zeros all equal)
    crest[1:-1, 1:-1] &= (h[1:-1, 1:-1] > 0.0)
    return crest


def kelvin_crest_mask(x_behind, y_lat, t, amplitude, speed, wave_scale,
                      ray_count, wave_count, decay, wedge_angle, time_scale,
                      threshold):
    """Convenience: sample the wake field and return its crest mask."""
    h = kelvin_field(x_behind, y_lat, t, amplitude, speed, wave_scale,
                     ray_count, wave_count, decay, wedge_angle, time_scale)
    return crest_mask_from_field(h, threshold)
