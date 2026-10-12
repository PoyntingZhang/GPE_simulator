# GPE simulator

A staged, single-GPU Gross–Pitaevskii simulator with one local interface for
**Experiment → Launch → Monitor**. The GPU runner remains independent of the GUI.

## Open the GUI

On macOS, double-click **Open GPE.command**. On first launch it creates a local
Python environment and installs the GUI dependencies, then opens your browser.
Python 3 must already be installed. Keep the launcher window open while using the
GUI; closing the browser alone does not stop it.

Alternatively, after installing `gui_periphery/requirements.txt`:

```bash
python -m gui_periphery.app
```

Inside the interface:

1. **Experiment:** create stages or open a YAML file. Saving YAML separately is optional.
2. **Launch:** enter a run name and notes; choose CSD3 or this GPU machine/Colab.
   Enter environment and scheduler settings, then click **Prepare run**.
3. For CSD3, click **Connect**, answer OpenSSH's password/key/MFA or host-verification
   prompts in the interface, then **Upload run**. Use **Set up environment** if needed,
   and **Submit GPU job**. On Colab, use the existing environment and **Launch simulation**.
4. **Monitor:** view the saved spectrum, density/TOF/phase image, scalar history and log.
   Select saved spectra, toggle log axes, set the TOF range, and enable independent
   timed refresh. Full wavefunctions are not transferred by monitor refresh.

See [the GUI guide](docs/GUI_WORKFLOW.md) for reopening runs and the single Colab
bootstrap cell. HPC authentication/submission and Colab GPU execution still require
live environment checks; they are not claimed verified by local tests.

## Software layout

```text
gpe/
  core/                  Evolution, potentials, grids, normalization, CUDA kernels
  io/                    Snapshots, numerical diagnostics (including n(k)), analysis
  main.py                Command-line entry point
  stage.py               Stage configuration and preparation
  stage_factory.py       Turn validated YAML into GPU stage objects
  launch.py              Evolution loop and output scheduling
  benchmark.py           Propagator timing tool
  power_law_study.py      Controlled study orchestration
config/
  yaml_parser.py         CPU-only parsing, defaults merging, validation, YAML export
  expressions.py         Shared expression validation and preview
  defaults.py            Defaults for new configurations
  examples/              Illustrative YAML, tracked with the software
gui_periphery/
  app.py                 Integrated GUI entry point
  editor/                Browser views: Experiment, Launch, Monitor
  workflow.py            Preparation, upload, job control, monitor sync
  connection.py          OpenSSH authentication and connection sharing
  viewer.py              Read existing diagnostic files for display
  colab.py               Display the GUI in a Colab notebook cell
  prepare_run.py         Build a named, portable simulation folder
  templates/             Run, environment setup, and Slurm submission scripts
tests/
```

The physics functions and CUDA kernels were preserved during this reorganization.
`n(k)`, scalar diagnostics, TOF projections, and snapshot handling belong to I/O.
Normalization stays in `core/` because imaginary-time propagation uses it.

`config` imports neither the GUI nor CuPy. Both the editor and runtime import
`config`. Only the stage factory and solver need a GPU. `gpe/plot_results.py` and
`gpe/analyze_power_law.py` are small entry points preserving the existing commands;
their implementations live in `gpe/io/`.

## Create a YAML locally

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r gui_periphery/requirements.txt
python -m gui_periphery.app
```

Save your file, for example as `config/my_run.yaml`. This folder now contains the
parser and can also hold your configurations. Personal YAML files here are ignored
by Git; `config/examples/` remains tracked. Existing schema-2 YAML files still work.

```bash
python -m gpe.main config/my_run.yaml --check
```

Validation checks the schema, types, allowed expressions, and supported stage
combinations. It does not test GPU memory capacity, load snapshots, compile CUDA,
or guarantee that a chosen formula is finite everywhere on the grid.

## Prepare a named run

```bash
python -m gui_periphery.prepare_run config/my_run.yaml \
  --name simulation_rotation_test_20261012 \
  --notes "Study relaxation after rotation; save spectra and final wavefunctions."
```

Use `--notes-file description.txt` for a longer description, `--output /some/folder`
to choose the parent directory, and `--cupy-package cupy-cuda13x` if appropriate for
the target environment. Without `--name`, a UTC timestamp is used. Names accept
letters, numbers, dots, underscores, and hyphens. Existing run folders are never
overwritten.

Each prepared folder also contains a `.gitignore` excluding its contents, including
when you choose a custom name or place it outside the default `runs/` directory.

```text
runs/simulation_rotation_test_20261012/
  gpe/                   Only the runtime core, I/O, and stage-launch code
  config/
    run.yaml             This run's validated configuration
    *.py                 Shared parser, expressions, defaults
  saved_data/            Full wavefunction snapshots and metadata
  monitor_data/          Raw diagnostic arrays and scalar CSV
  logs/                  Runtime log and Slurm output
  notes.txt              Your description
  run.json               Creation time, external inputs, file hashes
  requirements.txt
  requirements-gpu.txt   One CuPy package, selected for this run
  environment.sh         Per-run Python/CUDA module settings
  setup_environment.sh   Optional per-run virtual environment setup
  run.sh                 Launch the simulation and record its log
  submit.sh              CSD3 Slurm template; set your account and wall time
```

The builder changes only output locations in the copied YAML. Stage physics,
timesteps, sampling intervals, seeds, and the original YAML are preserved.
Output files keep their existing timestamp/stage subfolders. Absolute or escaping
stage output overrides must be replaced with relative directories before bundling.

For an initial state loaded from an earlier run, use its **absolute path on the
target machine**. Override a path while building with, for example:

```bash
python -m gui_periphery.prepare_run config/my_run.yaml \
  --name simulation_continuation \
  --input-path 1=/rds/project/previous/saved_data/RUN/stage_01_Preparation/psi_00001000.npy
```

This records a reference; it does not copy any wavefunction. The snapshot and its
required metadata must already be available on the target machine. Local absolute
paths are not automatically translated into HPC paths.

After transfer and environment setup, run inside the folder on an allocated GPU:

```bash
bash run.sh
```

See [upload, environment setup, Slurm submission, and Colab mock workflow](docs/HPC_WORKFLOW.md).
The existing direct command remains `python -u -m gpe.main PATH_TO_YAML`.

## Checks and implementation status

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

CPU checks cover configuration, run preparation, import separation, scheduling,
and I/O plumbing. CUDA tests skip without a working CuPy/GPU environment. Run
those on Colab before launching a long simulation:

```bash
python -m unittest discover -s tests -p test_propagators.py -v
```

The GUI integrates preparation, OpenSSH authentication, runtime upload, environment
setup, Slurm submission/status/cancellation and raw-data viewing. It can also launch
on the same GPU machine where its Python service runs, including Colab. SSH support
currently targets macOS/Linux with OpenSSH installed. No HPC daemon is installed.

Local regression tests cover fake SSH/scheduler responses, HTTP session boundaries,
authentication answer routing, portable bundles and partial diagnostic files. A
headless browser check exercised preparation, download, spectrum/time selection,
image/scalar controls and refresh using synthetic saved data. Actual CSD3 MFA,
submission and Colab GPU execution remain the next integration checks. No physics
code was modified for this GUI.

Existing local `config_editor/.venv` directories are left in place; virtual
environments are not moved because their executable paths can be absolute.
Recreate the editor environment using the command above. Historical notebooks and
original handoff documents are preserved and may refer to the old import paths.

More detail: [runner guide](gpe/run.md), [editor guide](gui_periphery/editor/README.md),
[refactor handoff](docs/HANDOFF_GUI_REORGANIZATION.md).
