# GPE software handoff 3 — 12 October 2026

Read this for the current software state. [HANDOFF.md](HANDOFF.md) and
[HANDOFF_2.md](HANDOFF_2.md) are preserved historical records. Their statements
about missing YAML integration, hardcoded drives, placeholder savers, and pending
kernel construction no longer describe the implementation. The current source
and the user's latest instructions take precedence over all handoffs.

## Scope and working agreement

- The software now runs a complete YAML-defined sequence: preparation, real-time
  drives, and subsequent relaxation. The user has exercised it on Colab GPUs.
- Keep implementation readable and changes bounded. Discuss major changes first.
- The user prefers command-line simulation and separate notebook cells for
  analysis. Save numerical spectra during a run; render figures afterwards.
- This task only adds this handoff and updates `.gitignore`. Do not stage,
  commit, push, untrack files, or alter either original handoff.
- The user explicitly stopped recovery/restart work after the Colab disconnect.
  Do not restart or launch simulations without a new instruction to do so.
- The user chose reusable snapshots only. Exact interrupted-job resume is not
  implemented or currently requested.

Repository root on the Mac is the synced Google Drive `My Drive/GPE_simulator`
directory. `GPE simulator/` (with a space) contains documentation and an Obsidian
vault; it is not the Python package or command working directory.

The Colab working copy is `/content/GPE_simulator`; its persistent Drive copy is
`/content/drive/MyDrive/GPE_simulator`. These are separate copies: changing a YAML
or source file in Drive does not update the runtime copy automatically. The agent
has filesystem access to the Mac repository, not a live connection to Colab.

## Current software map

| File or directory | Responsibility |
| --- | --- |
| `config_editor/app.py`, `static/` | Local browser YAML constructor/importer; Python HTTP server and vanilla JavaScript. |
| `config_editor/configuration.py` | Schema-2 defaults, validation, normalization, YAML import/export. |
| `config_editor/expressions.py` | Restricted expression parser and mathematical preview. |
| `gpe/expressions.py` | Translate validated expressions to CUDA syntax. |
| `gpe/main.py` | CLI validation and ordered stage execution. |
| `gpe/yaml_parser.py` | Build stage configurations, shared 1D axes, and output paths. |
| `gpe/stage.py` | Lowercase `stage` class; allocate state/potential and prepare propagator on demand. |
| `gpe/make_propagator.py` | Factories returning imaginary/static-real/dynamic-real stepping functions. |
| `gpe/make_kernel.py`, `kernels.py` | Generated spatial kernels and ordinary/masked kinetic kernels. |
| `gpe/launch.py` | Execute a prepared stage and schedule output after completed steps. |
| `gpe/save.py`, `state_io.py` | Headless diagnostic writer and slab-wise wavefunction snapshots/loading. |
| `gpe/grids.py`, `initial_states.py`, `potentials.py`, `helper.py` | Axes, Gaussian state, trap/absorber/disorder, norm and other helpers. |
| `gpe/nk_extraction.py` | Slab-wise GPU shell histograms with counts and completeness flags. |
| `gpe/plot_results.py` | CPU-only diagnostic gallery and optional spectrum-history PNG export. |
| `gpe/benchmark.py` | Ten propagator variants, synchronized timings and timestamped reports. |
| `gpe/power_law_study.py` | Specific twelve-case control-study orchestration, budgeting, and reduced-data archive. |
| `gpe/study_diagnostics.py` | Study total-energy/anisotropy diagnostics and spectral field refinement. |
| `gpe/analyze_power_law.py` | CPU-only study figures, fits, and HTML/CSV report generation. |
| `tests/`, `config_editor/tests/` | Workflow, expressions, GPU propagators/diagnostics, benchmarks, and study checks. |

`gpe/config.py` supplies editor defaults; changing it does not replace the settings
in an existing YAML file. `export.py` and `monitor.py` retain earlier APIs, but the
current CLI output path is `launch.py` → `save.py` → `state_io.py`. Do not delete
older APIs or notebooks without checking their callers.

The old `propagation.py`, `imaginary_run.py`, `dynamic_potentials.cuh`, and
`static_potentials.cuh` are absent. Do not restore their old hardcoded potential
logic based on the earlier handoffs.

## Stage lifecycle and kernel construction

1. `python -m gpe.main CONFIG.yaml --check` validates through the configuration
   module without importing CuPy. This checks settings, not snapshot existence,
   GPU memory, or CUDA compilation.
2. Normal execution calls `parse_yaml(path)`. This imports GPU modules and creates
   shared 1D axes, but defers each stage's large 3D allocations.
3. `stage.prepare(previous_psi)` selects Gaussian/load/continue, builds the box
   and optional absorber/disorder, and invokes the propagator factory once.
4. The factory constructs/selects its spatial and momentum kernel objects before
   defining the inner stepping function. Those objects are captured by the
   closure and reused at every step. They are not globals and are not rebuilt
   in the inner loop. CuPy compilation itself can be lazy and cached.
5. `launch_stage` assigns the returned field on every step:
   `stage.psi = stage.propagator(stage)` or
   `stage.psi = stage.propagator(stage, i * stage.dt)` for a dynamic drive.
6. The main loop retains only the final field for the next continuing stage and
   calls `stage.release()` to clear stage-owned state, potential, and closure.

A function assigned to an instance attribute is legal here: it does not become
a bound class method. The explicit `stage.propagator(stage)` argument is
intentional. Changing expressions or the cutoff after factory construction does
not rebuild a kernel; rebuild the stage/factory when those settings change.

Propagation is local half-step → FFT → kinetic full-step → IFFT → local half-step.
The first local operation mutates the input, while the IFFT returns another array.
Always use the returned field; avoid retaining obsolete full-volume references.
Both dynamic local half-steps use the same midpoint `t + dt/2`, each with its
current density. Every ordinary stage starts its drive clock at zero, including
loaded and continued stages.

Imaginary time supports optional static expressions, normalizes after every full
step, and uses a float64 stored potential. Real time supports independently
enabled static and dynamic expressions, uses complex128 `U - i*Gamma`, and never
renormalizes. A real-time momentum cutoff is optional. The ten benchmark variants
are two imaginary cases and eight real cases (static/dynamic/cutoff combinations).

## YAML and editor behavior

The canonical schema is in `config_editor/configuration.py`; see also
[config_editor/README.md](../../gui_periphery/editor/README.md). The shared simulation block
sets `N_grids`, `simulation_range`, `data_root`, and `monitor_root`. Stages have
their own evolution, initial state, cloud, potentials, box, absorber, disorder,
cutoff, save, and monitor settings. Defaults fill omissions; settings do not
implicitly inherit from the previous stage.

- `gaussian` creates a normalized field. `load` reads an existing complex128 NPY
  cube. `continue` passes the previous final field directly without disk I/O.
  Loading/continuing does not normalize. The first stage cannot continue.
- `cloud.N_atom` is the reference atom number and must remain unchanged across
  continued stages. Change `cloud.a` to change the interaction.
- Static expressions accept `x,y,z`; dynamic expressions also accept `t`.
  Coefficients are inline, with supported arithmetic/functions checked by the
  expression parser. There is no unrestricted Python `eval` or arbitrary CUDA
  input. Generated kernels recover coordinates from the thread's grid indices.
- Disabling a potential clears its expression to `"0"`. Imaginary stages reject
  enabled dynamic potentials, absorbers, and momentum cutoffs.
- Only uncorrelated disorder is supported (`correlation_length: 0`); its amplitude
  is a fraction of the enabled box's trap depth. A fixed seed reproduces a
  realization for the same grid/generation method.
- Legacy schema-2 `simulation.output_root` maps to separate data/monitor roots.
  The old `checkpoint_interval` is accepted as a snapshot cadence when unambiguous;
  it never means resumable execution. Schema-1 migration is not implemented.
- The editor's local save destination is separate from simulation output paths.
  Reopening a browser-selected YAML does not reveal its original absolute path;
  verify the destination filename before overwriting. A previously reported
  unchanged wall thickness was ultimately a wrong-file/name issue.

Relative simulation output roots resolve against the repository root. Relative
loaded-state paths resolve against the YAML file's directory. Stages share one
UTC timestamp per run and normally have separate `stage_01_Name` subdirectories.
See [gpe/run.md](../../gpe/run.md) for `save.directory` overrides and path checks.

## Outputs and numerical conventions

Default directories are `data/<run>/<stage>/` and `monitor/<run>/<stage>/`.

| Output | Contents |
| --- | --- |
| `psi_XXXXXXXX.npy` | Complex128 wavefunction cube. |
| Matching `psi_XXXXXXXX.json`, `grid.npz` | Stage settings, saved step/time, reference atom number, dimensions, axes. |
| `scalars.csv` | Ordinary runner: step, stage-local time, norm, remaining atom count, kinetic energy, Lz. |
| `nk_XXXXXXXX.npz` | `step,time,k,nk,Nk,counts,edges,complete_shells`. |
| TOF/density/phase NPZ files | Small numerical image arrays with axes and time labels. |

Intervals are independent counts of completed steps; zero disables. Initial/final
save flags request wavefunction snapshots only, not automatic diagnostic samples.
`monitor.show_*` fields in the CLI write numerical files rather than open windows.
Coincident momentum diagnostics share a completed-state FFT; Lz has its own work.
Frequent scalar printing also computes diagnostics and can cost more than text I/O.
At `dt=0.01`, saving every 1000 steps means every 10 dimensionless time units.

`state_io.SLAB_SIZE = 16` is the number of x planes per host/GPU transfer, not a
grid size, sample count, or physical parameter. All planes, including a shorter
last slab, are saved. Loading checks shape/dtype and available metadata/axes;
ordinary loading does not interpolate. A loaded snapshot starts a new stage at
t=0; saved step/time do not restore a stage's progress or drive phase.

The equation is `i*dpsi/dt = (-laplacian + V + beta*abs(psi)**2)*psi`, with kinetic
coefficient 1, not 1/2. Length unit is 1 micrometre; `a` is in Bohr radii and
`beta = 8*pi*N_ref*a/l`. Norm `Q` initially equals 1, and remaining atom number is
`N_ref*Q`. Do not reduce beta again as norm is lost. Physical seconds require the
atomic mass: `t_unit = 2*m*l**2/hbar`.

The FFT grid is cubic, periodic, unshifted internally, complex128 for psi, and
float64 for axes. `dx=D/N`, `dk=2*pi/D`; arrays are C-order `(x,y,z)`. Spectral
diagnostics use `phi = dx**3*fftn(psi)/(2*pi)**1.5`. `nk` is a shell average of
`abs(phi)**2`; `Nk` is discrete population per shell and `sum(Nk)` equals norm.
They are not interchangeable. Ordinary `E_kin` and `Lz` are integrated quantities,
not per-particle values. Study `E_total` is a separate diagnostic.

The saved spectra retain the full radial grid, including k=0 and incomplete outer
shells. Log plots must omit nonpositive points; mark incomplete shells separately.
Display limits do not change saved data. The plotter defaults to `--k-max 5`;
analysis code also filters spectra to k<=5. To inspect the full spectrum, reload
the NPZ without that filter as well as removing the axis limit. A notebook slider
alone cannot recover points already discarded by its loader. For the current
256-grid, D=70 study, saved positive k extends approximately 0.0898–19.93, while
complete positive shells extend to about 11.40.

Accepted numerical limitations remain: imaginary local exponentials freeze the
nonlinear density within each half-step; real-time nonlinear phase in the absorber
uses its initial local density. Do not label the full nonlinear method exact or
claim verified convergence from splitting order alone. A momentum cutoff acts
during the kinetic step; the last local step can regenerate high-k components.
Gaussian trap walls are finite barriers, not a hard-wall ground-state guarantee.

## Commands and verification history

Run package commands from the repository root. The editor documents Python 3.10+
and PyYAML. Root `requirements.txt` supplies NumPy and editor dependencies. Install
a CuPy distribution matching the GPU environment separately; Matplotlib is needed
for plotting/analysis and pytest for the older diagnostics test module.

```sh
python -m pip install -r requirements.txt
python config_editor/app.py --no-browser
# Editor URL: http://127.0.0.1:8765

python -m gpe.main experiment.yaml --check
# On an authorized GPU run:
python -u -m gpe.main experiment.yaml

python -m gpe.plot_results monitor --latest --tof-limit 5 --k-max 5
# Add --all-spectra only when a separate PNG for every saved spectrum is wanted.
```

CPU checks and separate GPU checks are documented in [gpe/run.md](../../gpe/run.md):

```sh
python -m unittest discover -s config_editor/tests -v
python -m unittest discover -s tests -p test_workflow.py -v
python -m unittest discover -s tests -p test_benchmark.py -v
python -m unittest discover -s tests -p test_power_law_study.py -v
# The study suite includes two GPU checks, skipped when CUDA is unavailable.
python -m unittest discover -s tests -p test_propagators.py -v
python -m pytest -q tests/test_diagnostics.py
```

Previously recorded validation: 27 editor, 14 workflow, 5 benchmark, and 8 CPU
study tests passed locally. The user supplied a Colab run with all 7 propagator
tests passing and later all 10 study tests passing, including its 2 GPU tests.
GPU propagator checks cover analytic phase/decay, loss, spatial expressions,
cutoff, snapshots, and a small multi-stage run. CPU workflow tests mock GPU kernels;
they do not establish CUDA correctness. These suites were not rerun merely to
write this handoff, and no simulation was launched for this task.

User-reported A100 80 GB timings for ten 1024-cubed variants were about 232–243
ms/real-time step and 261 ms/imaginary step, with outputs disabled. A study probe
at 256 cubed reported about 3.56 ms/step. These are propagation measurements, not
full end-to-end throughput with diagnostics/Drive copies. The benchmark tool
records warm-up, setup/first step, synchronized repeated timings, and metadata.
One complex128 1024-cubed array is 16 GiB; several arrays plus FFT workspace coexist.
No CSD3 deployment or production-grid convergence has been demonstrated here.

## Study software and saved partial run

The three study modules are software and remain visible to Git. The study runner
is deliberately tied to the approved seven-stage preparation, specific source
stage names, and 256-cubed initial snapshots. It is not a generic replacement for
`gpe.main`. Its default experiment YAML and one-off shell launcher are local run
inputs, excluded by `.gitignore`; a fresh software checkout alone cannot recreate
that experiment or its starting fields.

The study records spectra/scalars every 10 time units and images/full snapshots
every 500, with initial/final acquisition. It records configurations, copied
Python source, source hashes, throughput, and per-case status. There is a wall-time
budget with reporting reserve, not a resumable scheduler. Timing probes restore
the starting wavefunction even though propagators partly mutate it.

`review.zip` is a reduced archive copied to Drive after completed cases and in
the normal exception/finalization path. It includes spectra, scalar CSVs, configs,
source, and available analysis. It excludes full 3D wavefunctions and raw 2D image
arrays. Abrupt runtime destruction can bypass the final handler; the last copied
archive is not evidence of all cases completing. Full states need separate
persistent storage if future continuation is required. No recovery mechanism was
implemented after the user declined restarting.

The retained local study is `study_results/20261012T023004532610Z/`, sourced from
the completed stronger-drive run `20261012T014446961921Z`. It contains eight new
completed cases plus the existing reference. The three new preparation seeds
43–45 and the clean 512-cubed comparison are absent. The original manifest's
`running` status is stale. Last copied data were at 05:26 London time on 12 October
2026, after roughly 1 h 56 min; this was not a completed five-hour study.

Original `review.zip` and `manifest.json` were preserved. Locally recovered
`recovered_review/analysis/index.html`, `RECOVERY.txt`, and `partial_report.zip`
identify the partial status. To inspect/rebuild analysis from an extracted archive:

```sh
python -m gpe.analyze_power_law PATH_TO_EXTRACTED_REVIEW
```

The generic analyzer does not know that a stale archived `running` flag means a
lost Colab runtime. Regenerating the recovered report can replace its manually
added recovery banner; preserve that context explicitly. Missing image arrays
cannot be reconstructed from a radial spectrum.

Scientific context for the next session: the user is investigating an observed
approximately `nk ~ k^-3.3` segment, different from the 2.4 comparison exponent.
Do not dismiss that segment merely because other k ranges or later snapshots
curve. At t=550 the sampled k=0.359–0.808 interval has alpha=3.314 and log RMS=0.051
(six bins). The physical interpretation remains open. The report's broad 0.1–2
fit and its restricted `max(kp,kxi)`–initial-kp fit are distinct diagnostics.
`kxi` uses nominal cylindrical density; the shell-population peak is discrete,
not a precise condensation-time measurement. Fit residuals are unweighted natural
logs, not statistically calibrated chi-square. Six-bin minimums and seven-bin
local slopes are analysis choices, not physical tests of WWT validity.

The latest interactive time sliders were supplied in chat/notebook cells, not
added to the software. They span t=200–1000 and can show the full saved nk range.
An exploratory variant allows three bins; the report still requires six. Do not
assume these sliders or their modified fitting rules are already packaged.

## Version-control boundary at this handoff

Before this task, local HEAD was `cdbd1ef` ("Complete staged simulation workflow
and diagnostic tooling"), and status showed `main...origin/main` without divergence.
This only describes the local tracking reference; no remote fetch was performed.

Existing modified software files are `config_editor/examples/kick_and_hold.yaml`,
`gpe/plot_results.py`, and `gpe/run.md`. New software files not yet tracked are
`gpe/power_law_study.py`, `gpe/study_diagnostics.py`, `gpe/analyze_power_law.py`, and
`tests/test_power_law_study.py`. This task additionally changes `.gitignore` and
adds this handoff. None of those changes has been staged or committed by this task.

The ignore rules cover standard output trees, `first launch/`, local `runs/` and
`experiments/`, root-level experiment YAMLs/notebooks, current one-off launchers
and study instructions, NumPy data, snapshot JSON companions, logs, and result ZIPs.
Keep new run material in those output trees; custom output roots outside them
need an appropriate rule. Packaged YAML examples, reusable Python modules, tests,
and documentation remain eligible for version control. There is no blanket rule
excluding all YAML, shell scripts, images, or CSVs throughout the software tree.

`.gitignore` does not remove already tracked files. Four historical root notebooks
and some `.DS_Store` files are already tracked; they remain tracked. Untracking
them would change the index and was intentionally not done. Ignoring run files
does not delete them, back them up, or make Drive/runtime copies identical.
