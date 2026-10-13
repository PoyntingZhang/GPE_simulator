# GPE solver agent handoff 2

Updated 10 October 2026, after the user accepted the revised configuration editor
as good enough for now. This is a continuation of [HANDOFF.md](HANDOFF.md), not a
replacement. The original file has been left unchanged during this handoff update.

## Relationship to the first handoff

Read this file first for current status and decisions, then consult
[HANDOFF.md](HANDOFF.md) for the equation, units, FFT conventions, CUDA contracts,
and numerical approximations. Where dated project-status or design statements
conflict, this continuation records the later decisions. The user's current
instructions and the actual source take precedence over either document.

[FILE_PLAN.md](FILE_PLAN.md) also contains historical milestones; it is not an
up-to-date list of unfinished work. [The editor README](../../gui_periphery/editor/README.md)
documents the current YAML interface. [gpe/run.md](../../gpe/run.md) documents the
diagnostic interface, but all callers should be checked against current signatures.

| Earlier statement | Current position |
|---|---|
| No exports; monitoring and saving are future work | The user subsequently requested both. Diagnostics and exports are implemented, and notebooks save selected final states. |
| N/E/L definitions unresolved | N is wavefunction norm; E is dimensionless total kinetic energy; L is integrated dimensionless Lz. |
| Normalization uses ReductionKernel | `get_N` now uses the user's two-pass RawKernel reduction. The older reduction remains defined in the file. |
| No GPU runs or timings | The user ran Colab A100 benchmarks and supplied results; see below. Agents have not run the solver locally on a GPU. |
| Dynamic propagation is preliminary/unconnected | `ssfmstep_timeV` exists, with midpoint drive sampling and a hardcoded rotating linear potential. It is not a general expression interface. |
| YAML is only a possible future choice | A local editor now exports schema-2 YAML. The solver does not yet consume it. |
| Stage clock policy undecided | Every stage's potential time starts at t=0. Continuing a stage sequence passes ψ, not the previous drive clock. |
| Agent should primarily teach rather than implement | Follow the actual request. The user has asked for implementations and dislikes a teacher-like tone. |

## Working context and user preferences

Current repository root:

```text
/Users/zhanghaoyu/Library/CloudStorage/GoogleDrive-hz3577094@gmail.com/My Drive/GPE_simulator
```

The old Documents path in the first handoff is obsolete. `GPE simulator/` is the
documentation directory, not the code root. The user edits source and notebooks
through Google Drive; reread files before changing them. Parse notebook JSON and
inspect cell source rather than printing whole files containing embedded images.

- Keep communication direct and concise. Do not turn every discussion into a lesson.
- “Discuss,” “print code in chat,” and “do not modify” are real scope boundaries.
  Implement when requested; do not take a design discussion as authorization to edit.
- Keep code close to the existing style, with necessary comments and modest validation.
- Double precision is fixed. Do not add a precision selector or silently switch to single.
- Target Cambridge CSD3 over SSH, initially one GPU. Colab has supplied benchmarks;
  a working CSD3 batch deployment has not been demonstrated here.
- The current UI was explicitly accepted. Further polish is not an outstanding request.
- Do not commit unless requested. This handoff is context, not authorization to build
  the remaining runner or change simulation source.

## Git state

The user explicitly requested a source-only commit excluding notebooks:

```text
d52a39e  primitive 2
```

It contains changes to `gpe/config.py`, `dynamic_potentials.cuh`, `export.py`,
`helper.py`, `kernels.py`, `monitor.py`, `nk_extraction.py`, `propagation.py`,
`static_potentials.cuh`, and `tests/test_diagnostics.py`. No push was performed.

At this handoff, `config_editor/`, the four Colab notebooks, `.gitignore`, and
`simulation_results/` are untracked. Existing modifications also remain in
`GPE simulator/HANDOFF.md`, `GPE simulator/FILE_PLAN.md`, and `gpe/run.md`; those
predate this handoff-writing task. `HANDOFF_2.md` is an additional uncommitted file.
Do not stage everything indiscriminately: the working tree includes large results.
The editor work did not change `gpe/` source or notebooks.

## Solver progress since the first handoff

`export.py`, `monitor.py`, and `nk_extraction.py` are implemented. Intervals count
completed steps; zero disables. The export function saves ψ in NPY slabs, grid
metadata, norm CSV, and shell spectra. The existing scalar exporter is specifically
`N_interval`; the editor's broader `scalar_interval` still needs runner integration.

The isotropic spectrum retains shell mode counts, shell population `Nk`, and
shell-average momentum density `nk`. `sum(Nk)` equals the wavefunction norm; this
is not automatically an atom count. Shells at the Fourier cube's corners are kept
for normalization and marked incomplete. Avoid treating the discrete `Nk` as
exactly `4*pi*k**2*nk` without its shell width and discrete mode weights.

Requested monitor views are implemented: nk log–log and Nk linear–linear;
horizontal TOF integrated over kz; vertical TOF integrated over kx; their zero-plane
slices; in-situ density integrated over z or sliced at z=0; and phase at z=0.
The intentionally retained public spelling is `show_verticle_TOF_interval`, with
`show_verticle_TOF_slice_interval` for the slice. Existing plotting is not yet the
planned headless HPC diagnostic writer/local viewer system.

`helper.get_N` launches `sum_partial` (256 threads, at most 4096 blocks) and
`sum_finish`, including dx³ in the final norm. `normalise` still uses ordinary
in-place CuPy division by the square root. The user wrote this replacement after
profiling identified the original norm reduction as the main bottleneck.

### User-reported GPU measurements

These measurements came from the user's Colab runs, not local agent execution.
They describe the source/settings used then, before later hardcoded kick changes.

| Measurement | Result |
|---|---|
| Original imaginary step, 1024³ on A100 | Median 1277.22 ms; normalization 1045.094 ms (81.9% of profiled stages). |
| Original other stages | Local halves ≈24.7 ms each; FFT 71.218 ms; kinetic 20.094 ms; IFFT 90.892 ms. |
| After RawKernel norm reduction, 1024³ | Imaginary evolution ≈261 ms/step; normalization ≈29 ms. |
| Real evolution, 1024³ | ≈241 ms/step; FFT/IFFT now dominate. |
| Matched 256³ comparison, A100-SXM4-80GB | See table below; monitors disabled. |

| Stage | Current solver (ms/step) | Reference (ms/step) |
|---|---:|---:|
| Imaginary | 3.91 | 10.57 |
| Stirring | 3.63 | 10.98 |
| Relaxation | 3.56 | 10.03 |

The roughly 2.7–3.0× speedup is consistent with two transforms per current step
versus four in the reference, fused local kernels with fewer volume temporaries,
and the improved normalization. This is not an equal-accuracy convergence study:
the integrators' splitting and nonlinear-density treatment differ.

The user observed roughly 56.44 GiB device usage in imaginary evolution and
roughly 64.4 GB in real evolution. A float64 1024³ potential occupies 8 GiB, while
a complex128 potential occupies 16 GiB; this explains the main difference.
Those readings are not an audited peak-memory measurement. Caller-held ψ references,
out-of-place FFT results, plans, and pools still affect lifetimes and memory.
Deleting a function's local name does not release a caller's reference.

Further FFT tuning and 1D kinetic-factor precomputation were discussed and deferred.
The user prefers building a useful workflow before chasing another 10–20% speedup.

### Current source contracts and important unfinished integration

As inspected for this handoff:

```python
imaginary_step(psi, V_static, beta, k_array, dt, N_grids, x_array)
ssfmstep(psi, V_static, beta, k_array, x_array, dt, N_grids)
ssfmstep_kcut(psi, V_static, beta, k_array, x_array, dt, N_grids, kcut)
ssfmstep_timeV(psi, V_static, beta, k_array, x_array, dt, N_grids, kcut, t)
```

- **The current static real half-step always adds `kick(x_array[j], 1)`, i.e. V=y.**
  It is hardcoded, not controlled by the new UI. Ordinary `ssfmstep` therefore
  must not be described as having no extra potential without checking this code.
- The dynamic half-step calls `rotation(x,y,0.065,0.015,t)` from its CUDA header.
  The wrapper supplies `t + dt/2` for both halves. The dynamic and static kernels
  are separate; simultaneous arbitrary static/dynamic expressions are not wired up.
- Both header include options currently use `-Igpe`. Launch from the repository
  root; the include path is not yet independent of the working directory.
- Current notebook `ssfmstep` calls omit the subsequently added `x_array` argument.
  Recheck both production calls and direct profiling kernel launches before reuse.
  This affects the real-time benchmark, solver comparison, and stirring notebook.
- Current config literals: grid 256³, range 70, R=25, L=50, wall thickness=2,
  U_D=49, absorber distance=5/strength=100, dt=0.01, N_atom=150000, a=100 Bohr radii.
  Experiment notebooks can override these; do not assume all saved runs used U_D=49.

The equation remains `i*dψ/dt = (-∇² + V + beta*|ψ|²)*ψ`, with no kinetic 1/2.
Imaginary evolution normalizes; real-time evolution preserves absorber loss.
Use the returned ψ. The accepted approximate imaginary-time local update remains.

## Notebooks and reference scripts

Four separate, uncommitted notebooks exist:

- `colab_imaginary_benchmark.ipynb`: original 1024³/200-step benchmark, warm-up,
  stage profiling, monitoring, and final-only ψ saving.
- `colab_realtime_benchmark.ipynb`: loads a saved preparation state and benchmarks
  real evolution. Check the signature mismatch noted above before launching.
- `colab_stirring_run.ipynb`: preparation, 250 dimensionless units of stirring,
  then 1000 units of ordinary evolution. The chosen momentum cutoff was disabled
  (`kcut = infinity`). Imaginary step count is editable at the cell start and
  preparation ends with an in-situ density view. The user has edited parameters;
  current code says `imag_steps=2000` despite older prose saying 200.
- `colab_solver_comparison.ipynb`: matches settings to the reference, retains
  omega=0.065, disables monitoring, and compares stage speeds. It is not a second
  production solver or proof of identical numerical accuracy.

The requested stirring monitor schedule was scalars every 1000 steps; horizontal
and vertical TOF projections, in-situ z=0 density, and z=0 phase every 2000;
nk log–log every 3000. A later discussion requested relaxation monitoring at twice
those intervals. Check the actual notebook rather than assuming that discussion
was applied to every saved copy.

The user cautioned against unnecessarily long imaginary preparation: these finite
Gaussian barrier walls do not make the box interior the unique global ground-state
location. Short trapped-state preparation is intentional; norm=1 alone is not
evidence of convergence. Preparation length remains an editable experiment choice.

Read-only reference scripts discussed earlier are outside the repository:

- `/Users/zhanghaoyu/Downloads/3dsimulations_2_3.py`: previous solver used for the
  matched timing comparison.
- `/Users/zhanghaoyu/Downloads/prepare_a_kicked_state.py`: Gaussian preparation,
  finite linear kick, deeper-trap hold, then interacting evolution. Its kick is
  along z; the user's later y-kick was an example for discussing general stages.

Neither reference was modified. The kick experiment motivated the stage editor;
do not hardwire that one experiment into the general workflow.

## Agreed staged workflow

The intended workflow is a local configuration UI → YAML copied to CSD3 → one
CLI/batch run executing ordered stages → a separate future local monitoring UI
reading small HPC outputs over SSH.

- Shared simulation range and grid size; double precision throughout.
- Global HPC output root. Stages are ordered and numbered; names are optional.
- Each stage has its own evolution, potentials, trap, absorber, disorder, cloud,
  monitor intervals, save settings, and initial-state choice. No implicit settings
  inheritance is intended.
- `gaussian` generates an initial state, `load` reads an HPC wavefunction path,
  and `continue` passes the preceding stage's final GPU ψ and remaining norm.
  Stage 1 cannot continue. Passing ψ should not require a save/reload between stages.
- Potential time resets to zero at each stage start. Users encode offsets in their
  formula if they need phase continuity. This does not mean that resuming an
  interrupted stage should reset its progress or drive phase.
- Optical disorder belongs to the trap. Its amplitude is a fraction of that
  stage's trap depth; the seed reproduces a base realization for the same grid and
  generation method. No separate “continue previous noise” option is wanted.
- Disorder applies throughout the simulation region, with no box mask.
- Simple analytic potentials should be evaluated inside the local GPU kernel to
  avoid another full potential volume. The user wants flexibility including
  quadratic terms, not an interface limited to uniform forces.
- Imaginary evolution accepts neither a time-dependent potential nor an absorber.
- Stage saves and monitor schedules are independently configurable. The UI has a
  checkpoint interval, but exact restart execution/metadata are not implemented.

## Implemented local configuration editor

Everything is isolated in [config_editor/](../../gui_periphery/editor/). It uses a Python
loopback HTTP server, PyYAML, and vanilla HTML/CSS/JavaScript. No CuPy, GPU, Node
installation, or internet connection is needed to use it after dependency setup.

| File | Responsibility |
|---|---|
| `app.py` | Local server, static assets, defaults, formula feedback, YAML render/import/save APIs. |
| `configuration.py` | Defaults, schema, type/range checks, YAML validation and saving. Reads config literals with AST without importing the solver. |
| `expressions.py` | Restricted Python expression parsing and MathML preview. Does not evaluate user expressions or generate CUDA. |
| `static/index.html`, `style.css`, `app.js` | Stage editing, controls, live feedback, YAML preview, local drafts, save dialog. |
| `examples/kick_and_hold.yaml` | Updated schema-2 example, opened through Open YAML. No example button remains in the main UI. |
| `tests/test_configuration.py`, `test_expressions.py` | Configuration, saving, formula, and notation checks. |
| `README.md`, `requirements.txt` | Usage/schema documentation and PyYAML dependency. |

Launch from the repository root (the local `.venv` already exists at this date):

```sh
config_editor/.venv/bin/python config_editor/app.py
```

The address is `http://127.0.0.1:8765`. `--no-browser` suppresses automatic opening;
`--port 8766` selects another port. A `--no-browser` server was left running after
the editor update, but do not assume that process survives into a later session.
Restart after Python changes; static assets are read on request. See the README
for fresh environment setup. Local sandbox permissions may require approval to bind.

### Current YAML and UI behavior

The format is **schema_version: 2**. Top-level keys are `schema_version`,
`simulation`, and ordered `stages`. The source of truth is `configuration.py`;
the README describes each group. New preparation defaults include 2000 steps,
Gaussian width 3, scalars every 1000, no other monitor plots, `save_initial=false`,
and `save_final=true`. These are editor defaults, not calibrated run parameters.

Each stage has two independent user-potential blocks, which can both be active:

```yaml
potential:
  static:
    enabled: true
    expression: "0.03*x + 0.002*y**2"
  dynamic:
    enabled: true
    expression: "0.015*(sin(0.065*t)*x + cos(0.065*t)*y)"
```

Static expressions accept x,y,z; dynamic expressions also accept t. Coefficients
are written inline. There is no separate parameter block, expression/CUDA selector,
or exclusive static/dynamic selector. The dynamic label is
“Time dependent (t=0 at stage start)”.

Allowed syntax is `+ - * / **`, parentheses, finite real numeric constants,
`pi`, and one-argument `sin`, `cos`, `tan`, `exp`, `sqrt`, `tanh`, `abs`, `log`.
Angles use radians; log is natural. Functions use bare names, not `np.` or `cp.`.
Attributes, arbitrary calls, indexing, assignments, and other Python constructs
are rejected. Do not replace this restricted parser with Python `eval`.

- Live feedback reports names such as “Unrecognised variable A.” Invalid input
  appears below the field in small red text; valid input is green mathematical
  notation rendered offline through native MathML, not a LaTeX service.
- The same parser checks YAML import/export, including stored off formulas.
  Syntax/name checks do not prove the potential is finite over the entire grid.
- Entering a literal zero switches that contribution off on leaving the field,
  not after typing the first `0` in `0.03*x`. Save/import also normalizes zero to off.
  This is literal-zero detection, not symbolic simplification of expressions.
- Manual Off preserves the expression. Editing an off expression keeps it off
  until enabled. Each expression has its own toggle.
- Selecting imaginary evolution switches dynamic potential and absorber off and
  locks the corresponding controls. Enabled incompatible settings are rejected
  on import, too; the static expression remains available.
- Noise has seed, fraction of trap depth, and correlation length (zero means
  uncorrelated); no spatial region selector. Noise requires an enabled box to
  define its amplitude. Correlated-noise generation still needs runner support.
- Save YAML writes to a chosen local path; this is separate from HPC output/state
  paths in the configuration. Existing files require explicit overwrite selection.
- Unknown YAML fields, duplicate keys, and unsafe tags are rejected. Schema-1
  files are not automatically migrated: rewrite their potentials into the two
  schema-2 blocks, inline coefficients, and remove the old noise region setting.
- Browser drafts use `gpe-config-editor-v2`. Earlier v1 drafts remain under their
  old key, with a notice when starting a new draft; they are not automatically restored.

### Verification and its limits

The final editor revision passed 23 Python tests:

```sh
config_editor/.venv/bin/python -m unittest discover -s config_editor/tests -v
```

Live HTTP checks covered formula feedback, simultaneous potentials, YAML
roundtrip/save, overwrite refusal, and rejected invalid/imaginary-stage settings.
JavaScript syntax and mocked behavior checks covered locked controls, red errors,
green MathML, zero-on-blur, and stale response handling. Those JS/HTTP checks were
ad hoc, not additional committed test files.

Automated browser visual inspection was unavailable; Safari control was not
approved. Do not describe mock-DOM checks as end-to-end browser validation. The
user subsequently reviewed the revision and said it was good enough. No GPU or
CSD3 execution was performed as part of the editor work.

## Remaining work, when requested

1. Design and implement the interface between validated expressions and GPU local
   propagation. Preserve static plus dynamic contributions, real-valued potentials,
   double precision, and stage-local time. Translate `**` and functions deliberately;
   Python expression text is not directly valid CUDA source.
2. Build the YAML-driven stage runner/CLI, carrying ψ on the GPU between stages and
   constructing only the required trap/absorber/disorder arrays. Replace hardcoded
   potential choices through this interface; do not silently append the current V=y.
3. Connect headless monitor output, scalar/spectrum saves, final-state saving, and
   restart metadata to the schema. The present UI is configuration generation only.
4. Provide the local SSH diagnostics viewer and a verified CSD3 launch workflow.
   Confirm current cluster/account details then; no Slurm submission is authorized now.
5. Update notebook call signatures if asked to run them again. Later revisit FFT
   efficiency, buffer lifetimes, and 1D kinetic propagators using measurements.

No staged YAML runner, CUDA expression compiler, exact restart engine, remote viewer,
or CSD3 submission integration currently exists. Keep those boundaries explicit
when describing what the accepted editor can do.
