"""Diagnostics and complex-field interpolation used by the power-law study."""

import csv
from pathlib import Path

import numpy as np


def refine_field(psi, size, xp=np):
    """Fourier-interpolate a periodic complex cube without renormalizing it.

    Keep the original signed FFT modes, including the negative Nyquist mode.
    The sampling origin stays at -D/2. This preserves the integrated norm even
    if the input norm is below one. It does not introduce new physical modes.
    """
    old = psi.shape[0]
    if psi.shape != (old,) * 3 or old % 2 or size % 2 or size < old:
        raise ValueError("Interpolation needs even cubic grids with new size >= old size.")
    offset = (size - old) // 2
    modes = xp.zeros((size,) * 3, dtype=xp.complex128)
    modes[offset:offset+old, offset:offset+old, offset:offset+old] = xp.fft.fftshift(xp.fft.fftn(psi))
    return xp.fft.ifftn(xp.fft.ifftshift(modes)) * (size / old)**3


def physical_scalars(psi, psi_k, potential, beta, dx, k, xp=np):
    """Integrated GPE energies and lab-frame momentum second moments.

    The real static potential must include the complete conservative potential.
    Absorber losses are tracked through the norm, not included in E_total.
    Slabs limit temporary memory; no normalization is performed here.
    """
    norm = epot = eint = 0.0
    moments = np.zeros(3)
    scale = dx**3 / psi.size
    for start in range(0, len(k), 16):
        end = min(start + 16, len(k))
        rho = xp.abs(psi[start:end])**2
        norm += float(xp.sum(rho)) * dx**3
        epot += float(xp.sum(rho * potential[start:end].real)) * dx**3
        eint += float(xp.sum(rho**2)) * dx**3 * float(beta) / 2
        occupation = xp.abs(psi_k[start:end])**2
        for axis, wave in enumerate((k[start:end, None, None], k[None, :, None], k[None, None, :])):
            moments[axis] += float(xp.sum(occupation * wave**2)) * scale
    kinetic = float(moments.sum())
    fractions = moments / kinetic if kinetic > 0 else np.full(3, np.nan)
    return dict(norm=norm, E_kin=kinetic, E_potential=epot, E_interaction=eint,
                E_total=kinetic + epot + eint,
                kx2_fraction=fractions[0], ky2_fraction=fractions[1], kz2_fraction=fractions[2],
                axis_anisotropy=float(np.ptp(fractions)))


def record(stage, step, time_offset=0.0, images=False, scalar_row=True):
    """Save small raw arrays, including a genuine step-zero sample."""
    import cupy as cp
    from .diagnostics import get_L
    from .nk_extraction import nk_extraction

    folder = stage.monitor_dir
    folder.mkdir(parents=True, exist_ok=True)
    dx = float(stage.x_array[1] - stage.x_array[0])
    psi_k = cp.fft.fftn(stage.psi)
    spectrum = {key: cp.asnumpy(value) for key, value in
                nk_extraction(psi_k, stage.x_array, stage.k_array).items()}
    local_time = step * stage.dt
    metadata = dict(step=step, time=local_time + time_offset, stage_time=local_time)
    np.savez_compressed(folder / f"nk_{step:08d}.npz", **metadata, **spectrum)
    scalars = physical_scalars(stage.psi, psi_k, stage.staticV, stage.beta,
                               dx, stage.k_array, xp=cp)
    scalars.update(N_atoms=stage.N_atoms * scalars["norm"],
                   Lz=float(get_L(stage.psi, stage.x_array, stage.k_array)),
                   kp=float(spectrum["k"][np.argmax(spectrum["Nk"])]))
    volume = np.pi * stage.box_params["R"]**2 * stage.box_params["L"]
    scalars["kxi_nominal"] = np.sqrt(max(0.0, float(stage.beta) * scalars["norm"] / volume))
    row = dict(**metadata, **scalars)
    path = folder / "scalars.csv"
    fresh = not path.exists()
    if scalar_row:
        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if fresh:
                writer.writeheader()
            writer.writerow(row)

    if images:
        # Use the existing file conventions so these arrays remain reusable.
        x = cp.asnumpy(stage.x_array)
        k = cp.asnumpy(cp.fft.fftshift(stage.k_array))
        dk = abs(float(stage.k_array[1] - stage.k_array[0]))
        scale = dx**3 / (stage.psi.size * dk**3)
        rho = cp.abs(stage.psi)**2
        occupation = cp.abs(psi_k)**2
        middle = stage.N_grids // 2
        arrays = {
            "density_xy": dict(values=cp.sum(rho, axis=2)*dx, x=x, y=x),
            "density_slice_xy": dict(values=rho[:, :, middle], x=x, y=x),
            "phase_xy": dict(values=cp.where(rho[:, :, middle] > rho[:, :, middle].max()*1e-6,
                                             cp.angle(stage.psi[:, :, middle]), cp.nan), x=x, y=x),
            "TOF_xy": dict(values=cp.fft.fftshift(cp.sum(occupation, axis=2))*scale*dk, kx=k, ky=k),
            "TOF_yz": dict(values=cp.fft.fftshift(cp.sum(occupation, axis=0))*scale*dk, ky=k, kz=k),
        }
        for name, values in arrays.items():
            np.savez_compressed(folder / f"{name}_{step:08d}.npz", **metadata,
                                **{key: cp.asnumpy(value) for key, value in values.items()})
    return row
