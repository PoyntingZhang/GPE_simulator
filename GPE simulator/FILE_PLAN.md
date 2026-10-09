# GPE solver: current files and implementation plan

Updated 2026-10-09. This document is the practical writing order for the user and future agents. [HANDOFF.md](HANDOFF.md) records the numerical conventions, discussion, current review findings, and unresolved scientific choices. The user writes the implementation; agents review, explain, and update the requested documentation unless explicitly asked to change source code.

The project root is `/Users/zhanghaoyu/Documents/GPE_simulator`. Python source is in its `gpe/` directory. This document, the handoff, and the existing README are in the **documentation subdirectory** `GPE simulator/`; that directory is not the Python project root.

**No exports are wanted at the current stage.** Disable the existing export call while preparing the first runnable simulation. Do not enable checkpoint, spectrum, image, or final-state file writes automatically. Saving becomes an explicit opt-in feature in a later milestone.

## Current file map

These files already exist; this is not a scaffold to recreate or rename.

```text
GPE_simulator/
├── gpe/
│   ├── config.py
│   ├── grids.py
│   ├── helper.py
│   ├── potentials.py
│   ├── initial_states.py
│   ├── kernels.py
│   ├── dynamic_potentials.cuh
│   ├── propagation.py
│   ├── imaginary_run.py
│   ├── monitor.py
│   ├── export.py
│   ├── import.py
│   └── run.md
└── GPE simulator/
    ├── README.md
    ├── HANDOFF.md
    ├── FILE_PLAN.md
    └── Welcome.md
```

| Existing file | Current role and next responsibility |
|---|---|
| `gpe/config.py` | Python parameters for the current experiment. Keep this simple during initial development. Resolve the duplicate sources of interaction strength: the configured `beta` and the value calculated from atom number and scattering length. Add explicit stage and diagnostic settings when their behavior is defined. |
| `gpe/grids.py` | Constructs the one-dimensional position and unshifted angular-wavevector axes. Retain the existing plural filename. Keep shape, spacing, and units consistent with propagation and diagnostics. |
| `gpe/helper.py` | Contains the fused squared-modulus reduction, in-place normalization, scaled Fourier helpers for analysis, and interaction conversion. `get_beta` now returns its result. Keep physical normalization and units documented; propagation uses its own ordinary FFT/IFFT pair. |
| `gpe/potentials.py` | Builds the real box trap, positive absorber strength, combined complex static potential, and optional disorder. Imaginary-time preparation receives the real potential. Real-time evolution may receive the trap minus imaginary absorption. |
| `gpe/initial_states.py` | Constructs and normalizes a Gaussian. The package-relative import and complex128 Gaussian construction have been corrected. Add other initial-state constructors only as needed. |
| `gpe/kernels.py` | Contains embedded CuPy `RawKernel` definitions for local and kinetic steps, including imaginary-time, real-time, dynamic, and masked variants. The present kinetic kernels calculate the phase from the one-dimensional `k_array`; precomputed one-dimensional propagators are a possible later optimization. |
| `gpe/dynamic_potentials.cuh` | CUDA device helpers for analytic drives. Ensure compilation can find this header independently of the process working directory before using the dynamic kernel. Python functions do not execute inside CUDA simply by being passed as arguments. |
| `gpe/propagation.py` | Implements one complete imaginary-time, real-time, or masked step, launches kernels, and returns the current wavefunction after the inverse FFT. These functions are the reusable numerical operations for future stages. Normalization belongs to imaginary-time preparation, not absorbing real-time evolution. |
| `gpe/imaginary_run.py` | Current initialization and imaginary-time loop. Still needs the immediate driver fixes below, followed by a callable evolution function and a guarded `main()` entry point. |
| `gpe/monitor.py` | Placeholder. Will coordinate requested live progress information, scalar histories, and selected plots without requiring saved files. Numerical calculations may move into a separate observables module when that improves clarity. |
| `gpe/export.py` | Placeholder. Leave inactive now. Later saves only explicitly requested results, with independent schedules for small diagnostics and large states. |
| `gpe/import.py` | Reserved for future state loading. `import` is a Python keyword, so ordinary import statements cannot conveniently use this module name. A future `load_state.py` is proposed; no rename is required for the current smoke run. |
| `gpe/run.md` | Earlier workflow notes. Treat these as design notes, not a runnable entry point or an implemented YAML interface. Update or consolidate them after the working interface is settled. |
| `GPE simulator/README.md` | Existing scientific/project description. Update with verified installation and launch instructions as the runnable workflow develops. |
| `GPE simulator/HANDOFF.md` | Full context for another agent: user preferences, decisions, numerical details, caveats, and remaining questions. |
| `GPE simulator/FILE_PLAN.md` | This implementation map and sequence. |

The code has been reviewed locally, but CUDA compilation and execution still need a supported GPU environment. A Python syntax check does not validate CUDA strings or GPU behavior.

## Milestone 1: finish a small imaginary-time smoke run

**Prerequisite:** the current kernels, helpers, grids, initial state, potential, and single-step wrapper are present. Preserve the accepted approximate imaginary-time update and normalization after every complete step; redesigning the ground-state algorithm is deferred.

1. In `imaginary_run.py`, skip the export branch entirely. The current `i % export_step == 0` is true at `i = 0`, so a large interval does not disable exports. Do not add a final save either.
2. Skip the unimplemented monitor call until it is deliberately connected. The existing `export` and `monitor` names are not imported, so the saved loop currently fails after its first step. Importing empty placeholders would avoid that exception but would not implement monitoring.
3. Select one interaction-strength definition and calculate it once before the loop. Check the resulting value and units against the intended experiment.
4. Use completed-step numbering for schedules: after a call to `imaginary_step`, the completed step is `i + 1`. Define separately whether an initial-state observation at step zero is wanted.
5. Add a `main()` guard so importing the module does not allocate the simulation and start a long run. Keep launch from the project root as `python -m gpe.imaginary_run`.
6. Use a small grid and a short step count for the first GPU execution. Check the initial state and trap, launch both kernels, perform the FFT pair, and confirm finite values and the intended normalization.

**Done when:** a short GPU job reaches its final iteration without exporting any simulation data; the state has the required shape/dtype and finite norm, and importing the driver does not launch the job. Record the actual GPU/software environment and any compilation fixes. Normalization alone is not evidence of ground-state convergence.

## Milestone 2: make stages callable and composable

**Prerequisite:** the small imaginary-time run works. Keep the structure lightweight: functions and a small explicit state/configuration container are sufficient; a class framework is not required.

- Separate initialization from evolution. Initialization supplies `psi`, axes, potential, and resolved physical parameters. A stage takes the current state plus its settings, advances it, and returns the updated state.
- Keep single-step operations in `propagation.py`. Let stage functions own iteration counts, optional monitoring, and stage completion. Entry-point scripts call these functions rather than duplicating the stepping mathematics.
- Pass the returned wavefunction onward. Out-of-place inverse FFTs can create a new array, so a stage cannot assume that every operation preserves the original Python array identity. Do not create a copy merely to pass the state into another stage.
- Track stage identity, completed steps, and time explicitly. Imaginary-time iterations and their fictitious time do not advance the physical real-time clock.
- Define transitions explicitly: which trap, absorber, interaction, drive, or cutoff changes; whether a drive continues its existing phase or restarts; and what physical time the next stage begins at. Do not silently rebuild a Gaussian or reset the drive when composing stages.

**Done when:** preparation can return a state directly to another callable stage without export/reload, and a two-stage run has explicit parameters and clock behavior. The user can still launch a single preparation stage independently.

## Milestone 3: connect real-time, driven, and masked evolution

**Prerequisite:** reusable state and stage interfaces are established. Start from the already-written static and masked step functions.

| Work | Responsibility and completion check |
|---|---|
| Static real-time stage | Use the current state with a deliberately constructed complex static potential. Confirm conservative norm behavior without loss and decreasing norm with the absorber. Do not renormalize after real-time steps. |
| Driven stage | Connect the dynamic kernel and header search path; specify time sampling, drive parameters, and the starting phase. Check a short driven run and continuation across a stage boundary. |
| Masked stage | Use the existing kinetic cutoff operation with a clear cutoff convention. Document that the final nonlinear real-space step may generate modes above the cutoff again; the present algorithm filters at the kinetic step. |
| Stage transitions | Make trap/interaction changes and drive-phase resets explicit settings. Preserve the state and clock for a continuation unless the protocol intentionally changes them. |

Proposed entry-point names are `real_run.py`, `masked_run.py`, and eventually a small `protocol.py` or `run.py`. **These are future options, not files that already exist or a requirement to create one script per possible combination.** Prefer shared stage functions to copied simulation loops.

**Done when:** small runs exercise the intended stage types and a preparation-to-real-time sequence, with the correct state, normalization policy, and physical-time/drive-phase handling.

## Milestone 4: add optional monitoring and diagnostics

**Prerequisite:** a stage can advance without diagnostics. Each diagnostic must be independently enabled and scheduled. Computing, displaying, and saving a result are separate choices; enabling a live plot must not silently enable exports.

Start with inexpensive scalar progress, then add the requested spectra and images. Reuse calculations when multiple enabled items are due together, without keeping extra full-grid arrays permanently alive.

| Requested result | Implementation responsibility and definition to settle |
|---|---|
| `N` | GPU reduction for integrated density/norm. Distinguish the wavefunction's unit normalization from physical atom number. During normalized imaginary time, use this as a sanity check, not a convergence measure. |
| `E_kin` | Kinetic-energy diagnostic, with the units, normalization, and whether it is total or per particle stated explicitly. The user wants a quantity useful for comparison with TOF; distinguish the three-dimensional simulated kinetic energy from any projected experimental estimator. |
| `L` | Requested scalar whose meaning is still undecided. Confirm the intended observable, components, and units before implementing it. Do not silently interpret it as a particular angular momentum or length. |
| Isotropic `n_k` | Reduce momentum-space occupation into radial bins on the GPU and return a small one-dimensional result. Fix bin edges, normalization, and whether the output is a shell average or shell-integrated population. See the handoff for conventions to resolve. |
| `n_k` plot | Plot the agreed radial spectrum on log–log axes. Handle the zero bin and nonpositive values deliberately. |
| `k² n_k` plot | Plot on linear axes using the same defined `n_k`. The interpretation depends on the radial-spectrum convention; do not multiply a shell-integrated population by `k²` without revisiting that meaning. |
| Momentum image integrated over `k_z` | Calculate the two-dimensional projection of momentum density. This is the requested TOF-like momentum view; any conversion to detector position requires stated expansion-time and physical assumptions. It is distinct from a real-space density column. |
| Density at `z = 0` | Extract the central real-space density slice, with a deliberate choice if the grid does not contain zero. Transfer only the two-dimensional result for display. |
| Phase at `z = 0` | Extract the corresponding phase slice. Decide how to hide or flag phase in nearly empty regions. |
| Column/slab-integrated real-space density | Integrate density along a selected axis, optionally over a finite slab. Projection axis, slab location, and thickness remain open choices. Include the grid-spacing factor. |

For preparation convergence, choose and document more than a normalized norm: energy change, a stationary-equation residual, or a suitable change-of-state measure can be added incrementally. Avoid an extra retained full wavefunction merely to make a cheap-looking progress display. An approximate imaginary-time scheme also needs a later timestep-refinement check; a stable iteration alone is not a guarantee of an accurate ground state.

Keep `monitor.py` as the coordinator initially. A proposed `observables.py` can hold numerical reductions when the file becomes substantial, and plotting can be separated if needed. These splits are organizational choices, not prerequisites for the first scalar monitor.

**Done when:** each selected item can be enabled or disabled and run at its own cadence; the displayed results have documented normalization/units; monitoring can be disabled completely; and a representative run remains within the memory budget. A small exported radial spectrum is cheap to store, but its calculation can still require an FFT and substantial temporary GPU memory.

## Milestone 5: add opt-in saving and state reuse

**Prerequisite:** the requested diagnostics and their definitions work. There are no exports before this milestone unless the user explicitly changes that requirement.

Use `export.py` for saving and a proposed `load_state.py` for loading. Keep three purposes distinct:

| Purpose | Content and completion check |
|---|---|
| Small diagnostic export | Selected scalars, radial spectra, or two-dimensional images plus their coordinates, units, time, and settings. Every item has an independent opt-in cadence. |
| Reusable prepared state | Wavefunction plus grid, normalization, units, and preparation metadata sufficient to use it as an initial condition in another experiment. Loading checks compatibility before allocating/evolving. |
| Restart of the same run | Also includes physical time, stage/step, resolved parameters, drive phase/origin, protocol position, and any random-process state needed for continuation. Verify against an uninterrupted small run within a stated numerical tolerance; do not promise bitwise identity across different environments. |

A saved wavefunction by itself is not necessarily a complete restart. Decide whether a final save is wanted independently of periodic saving; **the current answer is no**. Large state saving should stage data deliberately and avoid multiple full CPU/GPU copies. A complex128 `1024³` state alone is 16 GiB before metadata.

**Done when:** only explicitly selected outputs are written, saved states load with compatibility checks, and any claimed restart functionality passes the uninterrupted-versus-resumed comparison.

## Milestone 6: package, validate, and measure on HPC

Small GPU tests should happen as soon as their modules are ready, not after every feature above is complete. This milestone collects the reproducible cluster workflow and performance evidence.

| Proposed support file | Purpose |
|---|---|
| `gpe/__init__.py` | Optional minimal explicit package file. Keep importing the package free of simulation work. Its current absence does not prevent Python namespace-package launch. |
| `pyproject.toml`, `.gitignore` | Package/dependency metadata and exclusion of generated data, environments, caches, and job logs. |
| `hpc/wilkes3.sbatch` | Requests the actual allocated CSD3 account/partition/GPU resources, sets up the environment and working directory, and launches the chosen module. Verify current cluster details rather than assuming them. |
| `benchmarks/benchmark_gpu.py` | Records GPU, available memory, precision/software versions, initialization, first compilation/planning, warmed step timings, and peak memory. Separate FFT and local-kernel costs when useful. |
| `tests/test_numerics.py` | Small meaningful checks of phases, FFT normalization, norm/loss behavior, known limiting cases, and timestep refinement. Distinguish CPU-only checks from GPU execution. |
| `tests/test_restart.py` | Needed when restart is implemented, not a prerequisite for today's export-free smoke run. |
| `analysis/inspect_run.py` | Optional offline analysis once saved results exist. |
| `configs/*.yaml` and a CLI | Optional later interface once the configuration schema is stable. Keep Python configuration now; YAML is not required by HPC. |

Start small, then increase the resolution while measuring available headroom. Keep out-of-place CuPy FFTs initially. The current wrappers can temporarily hold **three** complex wavefunctions during inverse-FFT reassignment, in addition to the static potential, FFT workspace/cache, allocator reservations, and diagnostic workspaces. Do not budget only two permanent wavefunction arrays. At `1024³`, three complex128 states alone occupy 48 GiB; the real imaginary-time potential adds 8 GiB, or a complex real-time potential adds 16 GiB, before the other costs.

Benchmark the actual enabled diagnostics and, when saving is later enabled, its transfer/staging costs. A propagation-only fit is not enough to establish that the intended production job fits. Investigate buffer lifetime, FFT-plan reuse, fused operations, and eventually in-place FFT only where measurement justifies them. Multi-GPU decomposition is a separate future project if single-GPU limits require it.

**Done when:** short Slurm jobs are reproducible, numerical checks pass for the intended modes, representative timings and memory peaks are recorded, and larger target resolutions—including `1024³`—have measured evidence rather than a fit assumption.

## Boundaries to preserve throughout

- All Python modules may allocate arrays when that is their responsibility. Array creation is not restricted to `kernels.py`; make large allocations and their lifetimes visible.
- Keep the evolving state on the GPU between steps and stages. Routine diagnostics transfer scalars, one-dimensional reductions, or two-dimensional views, not the full state.
- Retain one-dimensional coordinate/frequency axes. Avoid permanent full `X/Y/Z`, `K_x/K_y/K_z`, total-potential, or density grids solely to support elementwise updates.
- Real-time loss must remain loss: never normalize it away. Imaginary-time normalization is deliberate preparation behavior.
- The current approximate imaginary-time method is accepted for initial preparation. Its refinement or replacement with a GPELab-style method is a later scientific/numerical task.
- Keep the actual implementation and documentation synchronized. Mark proposed filenames and untested behavior honestly; do not describe a planned CLI, monitor, export, restart, or buffer-reuse mechanism as already implemented.
