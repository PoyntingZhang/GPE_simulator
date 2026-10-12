# Run a staged simulation

For named, portable run folders and the upload/submit workflow, see
[the repository README](../README.md) and [HPC workflow](../docs/HPC_WORKFLOW.md).

From the repository root, install the shared Python dependencies:

```sh
python -m pip install -r requirements.txt
```

For simulation, also install a CuPy build matching the GPU machine's CUDA setup.
For example, with an existing CUDA 12.x toolkit, use `python -m pip install cupy-cuda12x`.
Choose the appropriate package and Python version using the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
The local editor and YAML validation also work on machines without CUDA.

Validate YAML settings, then run on the GPU machine:

```sh
python -m gpe.main experiment.yaml --check
python -m gpe.main experiment.yaml
```

`--check` validates settings only; it does not load snapshots, allocate GPU arrays,
or compile CUDA kernels. `--help` lists the CLI options.

Create the YAML with `python -m gui_periphery.editor.app`, or adapt
`config/examples/kick_and_hold.yaml`. The example is illustrative; choose
its physics, grid size, timestep, and durations for your experiment.

`config/yaml_parser.py` exposes `parse_yaml(path)`, returning a validated dictionary
without importing CuPy. `gpe/stage_factory.py` exposes `make_stages(path)`, returning
the ordered list of stage objects. It translates validated potential expressions into CUDA syntax
and shares the one-dimensional axes. Large arrays and kernel objects are prepared
only as each stage begins, so parsing many stages does not allocate many 3D states.
The loop in `gpe/main.py` is:

```python
from gpe.stage_factory import make_stages

stages = make_stages("experiment.yaml")
psi = None
for current in stages:
    if current.initial_state["type"] != "continue":
        psi = None
    current.prepare(psi)
    psi = None  # Release the caller's reference to the stage's initial array.
    psi = launch_stage(current)
    current.release()  # Drop stage-owned arrays and kernels; keep returned psi.
```

A Gaussian is normalized on creation. `continue` takes the preceding wavefunction
directly; `load` reads a complex128 NPY cube. Neither renormalizes the state.
Imaginary propagation normalizes after each step. Real propagation retains losses.
Continued stages must keep the same reference `N_atom`; change `a` to adjust the
interaction. A generated state may use a different reference atom number.
Every stage starts at t=0, including a stage loaded from a snapshot. Dynamic
potentials use the existing midpoint-time propagator. Only uncorrelated disorder
(`correlation_length: 0`) is currently supported.

## Output folders and intervals

Defaults are `data/<run>/stage_01_Name/` for wavefunctions and
`monitor/<run>/stage_01_Name/` for diagnostics, beside `gpe`. The same timestamp
identifies all stages of one run. Relative output roots are relative to the
repository root. `save.directory` optionally changes a stage's subfolder in both
roots; an absolute override writes wavefunctions under `<override>/<run>` and
keeps monitoring under its normal root. Duplicate stage output folders are rejected.
Relative overrides must remain inside the timestamped run folder.
Folders are created when first used.

All intervals count completed steps within each stage; zero disables. Schedules
are independent. `save_initial` and `save_final` additionally request wavefunction
snapshots, without forcing diagnostic output or duplicating a periodic snapshot.

| Setting | Output |
| --- | --- |
| `save.psi_interval`, `save_initial`, `save_final` | `psi_XXXXXXXX.npy`, matching JSON metadata, and shared `grid.npz` in the data folder |
| `save.scalar_interval` | Append `step,time,norm,N_atoms,E_kin,Lz` to `scalars.csv` in the monitor folder |
| `monitor.print_scalar_interval` | Print step, stage time, norm, kinetic energy and Lz |
| `save.nk_interval` or `monitor.show_nk_interval` | `nk_XXXXXXXX.npz` |
| Other `monitor.show_*_interval` settings | TOF projections/slices, density projections/slices, or phase as small NPZ arrays |

Monitoring files contain step, stage-local time, values and relevant axes. They
are saved without opening plot windows. The old standalone plotting helpers in
`gpe/io/monitor.py` remain available; use the separate plotting command below to
view diagnostics after a run.

Wavefunctions are copied in slabs to avoid an extra full host-memory copy.
`SLAB_SIZE = 16` in `gpe/io/state_io.py` names the number of x planes per transfer;
every plane, including a shorter final slab, is saved. The
JSON records the stage configuration, saved step/time, grid size/range and reference
atom number. To reuse a state, set `initial_state.type: load` and `path` to its NPY
file; relative paths resolve against the YAML file's folder. Keep the JSON and
`grid.npz` alongside it for consistency checks. Older `<snapshot>_grid.npz`
companions are also recognized. Bare NPY files can be loaded, but
only their shape and dtype can be checked. Loading never interpolates or rescales.
These are snapshots: the recorded step/time are informational, not restart state.

## View saved diagnostics

Generate plots after a run, on either the GPU machine or a CPU-only computer:

```sh
python -m pip install matplotlib
python -m gpe.plot_results "first launch/monitor" --latest
```

Replace the path with your monitor root. Alternatively pass one timestamped run
folder and omit `--latest`. The command prints which run it selected and creates
`plots/` inside that run: each stage gets scalar-history and diagnostic PNGs when
the corresponding files exist. It also creates `plots/index.html` and a
`plots.zip` containing the complete gallery. Download and extract the ZIP, then
open `index.html` in a browser, or open individual PNGs in Colab's Files panel.
No CuPy, GPU, wavefunction transfer or simulation rerun is needed for these plots.

To zoom all TOF images and slices to dimensionless momentum [-5, 5] on both axes:

```sh
python -m gpe.plot_results "first launch/monitor" --latest --tof-limit 5
```

This changes the plotted axes only; saved arrays retain the full momentum range.
Real-space images and the radial momentum spectrum keep their existing axes.

Momentum spectrum n(k) plots use log-log axes with an upper k limit of 5 by
default; use `--k-max` to change it. Zero k and nonpositive n(k) are omitted;
the lower k limit is the first populated positive-k bin in the selected range.
Shell populations Nk use linear axes from k=0 to the chosen upper limit.
The saved spectrum arrays retain every bin, including k=0.

To export a separate log-log PNG for every saved spectrum:

```sh
python -m gpe.plot_results "first launch/monitor" --latest --tof-limit 5 --all-spectra
```

These images are saved under `plots/<stage>/spectra/nk_XXXXXXXX.png` and included
in `plots.zip`. Expand **Momentum spectrum history** in the HTML gallery to
browse them. Figure rendering happens after the simulation. Sampling times come
from the saved NPZ files: at dt=0.01, `save.nk_interval: 1000` saves every 10
dimensionless time units. Set `monitor.show_nk_interval` to 0 or the same cadence
to avoid additional spectrum outputs. A 100,000-step run then has 100 samples,
at t=10, 20, ..., 1000; enabling `save_final` does not force a spectrum save.

Each diagnostic uses its latest saved sample, labelled with its step and
stage-local time. These samples may precede the final wavefunction or differ
between quantities. Stages with image monitoring disabled have no diagnostic
image plot; their saved scalar history is still plotted. Image color scales are
independent across panels; phase uses [-pi, pi], and masked phase remains blank.
Incomplete outer momentum shells are distinguished in the spectrum plot.
Rerunning the plotting command replaces the generated plots, leaving numerical
simulation outputs intact.

## Diagnostic conventions

`norm` is the wavefunction norm, and `N_atoms = cloud.N_atom * norm`. `E_kin` is
integrated dimensionless kinetic energy, and `Lz` is integrated angular momentum;
neither is divided by norm. The reference atom number is not the remaining count.

`nk` is the shell average of `|phi(k)|²`, where
`phi = dx³ fftn(psi)/(2 pi)^(3/2)`. Spectra retain mode `counts`, bin `edges`, shell
populations `Nk`, and `complete_shells`. `sum(Nk)` equals norm. Outer shells
intersect the Fourier cube boundary and are marked incomplete.

TOF arrays integrate over kz or kx, or use the actual zero-frequency plane.
Density arrays integrate over z or use index `N_grids//2`; that plane is z=0 for
even grids, and the saved z coordinate identifies it for odd grids. Phase is NaN
where density is at most 1e-12 of the slice maximum. Momentum diagnostics due at
the same step share one FFT of the completed state; Lz uses its separate slab FFTs.

## Checks

CPU configuration and workflow tests (NumPy and PyYAML required):

```sh
python -m unittest discover -s tests -p test_configuration.py -v
python -m unittest discover -s tests -p test_expressions.py -v
python -m unittest discover -s tests -p test_workflow.py -v
```

The workflow tests substitute NumPy arrays and mocked GPU kernels; they check
configuration, scheduling and file I/O, not CUDA execution or numerical accuracy.
Before a full simulation, run the small analytical GPU checks:

```sh
python -m unittest discover -s tests -p test_propagators.py -v
```

These exercise the generated spatial formulas, real-time phase (with static and
dynamic potentials and cutoff), imaginary-time decay and normalization, absorber
loss, cutoff loss, GPU snapshot roundtrips, and a two-stage run with saved diagnostics.
A skipped result means no GPU
verification occurred. Passing these checks does not establish convergence for
a particular nonlinear experiment; choose its timestep and grid accordingly.

On a CUDA machine with CuPy, NumPy, Matplotlib and pytest installed, the existing
diagnostic tests can also be run with `python -m pytest -q tests/test_diagnostics.py`.

## Benchmark the propagator variants

From the repository root on a CUDA machine:

```sh
python -u -m gpe.benchmark --grids 64 128 256 --steps 50 --warmup 5 --repeats 3
```

This compares ten cases: imaginary time with/without a static external potential,
and all eight real-time combinations of static potential, dynamic potential and
momentum cutoff. It calls the existing `launch_stage` loop with every diagnostic
and snapshot disabled. The propagators are unchanged. Each repeat starts from the
same Gaussian and stage time zero. The benchmark uses complex128, nonzero
interactions, and specified representative potential expressions; box, absorber
and disorder are off. The generated YAML files record every setting.
The Gaussian is regenerated outside the timer before each repeat, after releasing
the old state; no extra full GPU copy is retained for resets.

The primary result is synchronized wall time per step, with median and min/max
over repeated batches; CUDA-event timings are also recorded. Setup and the first
step are reported separately. Those initial timings share existing CuPy caches
and are not controlled cold-start measurements. Cases are shuffled using `--seed`
(default 42). FFT plans and unused memory pools are cleared between grid sizes.
Other workloads on the GPU can affect the measurements, so run the benchmark
without a simultaneous simulation. See the
[CuPy timing guidance](https://docs.cupy.dev/en/stable/user_guide/performance.html).

Reports are written after each completed case to a new
`benchmark_results/<UTC timestamp>/` folder (override the parent with `--output`):
`report.md` for reading, `summary.csv` for analysis, `results.json` for individual
repeat timings, hardware details, execution order and source hashes, and
`cupy_config.txt` for CUDA library versions. No wavefunctions are saved.
The report marks partial results if execution stops before all cases finish.

A short check of the benchmark itself can use
`--grids 8 --steps 2 --warmup 1 --repeats 2`; these tiny grids are not representative
performance measurements. CPU tests of the benchmark's configuration, timing
boundaries and reporting run with:

```sh
python -m unittest discover -s tests -p test_benchmark.py -v
```

For a 1024³ trial on an otherwise idle 80 GB A100, start with:

```sh
python -u -m gpe.benchmark --grids 1024 --steps 10 --warmup 3 --repeats 3
```

Each complex128 cube is 16 GiB. The current real-time propagator can retain four
such arrays simultaneously (input, potential, Fourier state and inverse-FFT
output), before FFT workspace and other allocations. A 40 GB GPU cannot hold
this workload. Actual fit on 80 GB depends on available memory and FFT workspace;
the command is a trial, not a guarantee. It writes a separate timestamped report.
