"""Slab-wise wavefunction snapshots, with small companion metadata files."""

import json
from pathlib import Path

import cupy as cp
import numpy as np


# Number of x planes per CPU/GPU transfer (256 MiB for a 1024³ complex128 grid).
SLAB_SIZE = 16


def save_wavefunction(path, stage, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = np.lib.format.open_memmap(path, mode="w+", dtype=np.complex128,
                                     shape=stage.psi.shape)
    for start in range(0, stage.N_grids, SLAB_SIZE):
        saved[start:start + SLAB_SIZE] = cp.asnumpy(stage.psi[start:start + SLAB_SIZE])
    saved.flush()
    del saved
    np.savez(path.parent / "grid.npz", x_array=cp.asnumpy(stage.x_array),
             k_array=cp.asnumpy(stage.k_array))
    metadata = dict(step=step, time=step * stage.dt, evolution=stage.type,
                    N_atom=stage.N_atoms, simulation_range=stage.simulation_range,
                    N_grids=stage.N_grids, stage=stage.config)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_wavefunction(path, stage):
    """Load a reusable state without changing its norm or carrying over its clock."""
    path = Path(path)
    saved = np.load(path, mmap_mode="r", allow_pickle=False)
    if saved.shape != (stage.N_grids,) * 3 or saved.dtype != np.complex128:
        raise ValueError("The loaded state must be a complex128 cube matching N_grids.")
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["simulation_range"] != stage.simulation_range:
            raise ValueError("The loaded state has a different simulation range.")
        if metadata["N_atom"] != stage.N_atoms:
            raise ValueError("The loaded state uses a different reference N_atom.")
    # Older exports keep the axes beside each snapshot rather than per directory.
    grid_path = path.with_name(path.stem + "_grid.npz")
    if not grid_path.exists():
        grid_path = path.parent / "grid.npz"
    if grid_path.exists():
        with np.load(grid_path, allow_pickle=False) as grid:
            for name in ("x_array", "k_array"):
                expected = cp.asnumpy(getattr(stage, name))
                if grid[name].shape != expected.shape or not np.allclose(grid[name], expected):
                    raise ValueError("The loaded state's grid does not match this simulation.")
    psi = cp.empty(saved.shape, dtype=cp.complex128)
    for start in range(0, stage.N_grids, SLAB_SIZE):
        # Finish reading the host slab before releasing its memory mapping.
        psi[start:start + SLAB_SIZE] = cp.asarray(saved[start:start + SLAB_SIZE], blocking=True)
    return psi
