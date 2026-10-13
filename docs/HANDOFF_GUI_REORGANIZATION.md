# Software handoff: GUI reorganization

Date: 2026-10-12. Branch: `gui-implementation`.

This supplements the original files now under [history/](history/), formerly
`GPE simulator/`; those handoffs were
not overwritten. They describe earlier layouts and investigation history.

## Boundaries

- `gpe/core/`: evolution, grids, potentials, CUDA kernels and normalization.
- `gpe/io/`: snapshot loading/saving, spectra, scalar diagnostics, legacy output
  APIs, plotting and study analysis. `nk_extraction.py` is here by explicit request.
- `config/`: parser, schema validation, defaults and expression language shared
  by the runner and GUI. No CuPy or GUI imports.
- `gpe/stage_factory.py`: `make_stages(path)` creates the ordered stage objects;
  `config.yaml_parser.parse_yaml(path)` returns a validated dictionary only.
- `gui_periphery/editor/`: the existing editor, now run as a module.
- `gui_periphery/prepare_run.py`: creates named portable runtime folders. It does
  not open an SSH connection, upload anything, or launch a simulation.

Runtime entry points remain `python -m gpe.main`, `python -m gpe.benchmark`,
`python -m gpe.power_law_study`, `python -m gpe.plot_results`, and
`python -m gpe.analyze_power_law`. Internal imports use the new locations; no broad
compatibility layer was added for historical notebooks.

## Physics constraint

Do not change the numerical evolution as part of GUI work. All existing evolution
functions and CUDA definitions were moved without arithmetic changes. The old
`helper.py` was split: normalization and beta stay in core; diagnostic reductions
and scheduling helpers go to I/O. Imports were adjusted. Stage preparation and
the propagation loop preserve their previous behavior.

The source audit compared 15 evolution/I/O modules after removing imports and all
13 top-level functions/kernel definitions from the split helper. They matched
the pre-refactor source. Validation finished with 60 passing tests and 10 GPU-related
skips, including isolated runtime bundle checks. The relocated editor's pages and
YAML endpoints passed a localhost smoke check; all shell templates passed `bash -n`.
CUDA execution still needs the Colab GPU test pass; there is no
local CUDA device on the development machine.

Pre-existing uncommitted study/analysis software, plot improvements, ignore rules,
the illustrative example, and `HANDOFF_3.md` are included in this branch. They are
not new physics changes made by the reorganization. Specific run YAML, wavefunctions,
monitor outputs, plots, logs, study result archives and notebooks remain excluded
from new commits by `.gitignore`.

## Run folders and environments

See the root README and `docs/HPC_WORKFLOW.md` for commands. A bundle contains only
the runtime subset of `gpe/`, shared `config/`, copied `config/run.yaml`, notes,
file hashes, dependencies, shell scripts and empty output directories. The builder
preserves stage settings, redirects output roots to `saved_data`/`monitor_data`,
and supports explicit target snapshot path overrides. Snapshots are not copied.
Names are customizable; existing folders cannot be overwritten. Existing per-run
timestamp and stage subdirectories are retained.

`environment.sh` and `requirements-gpu.txt` can be specialized for each launch.
`setup_environment.sh` optionally creates a virtual environment and records installed
packages. It is never run automatically during folder preparation. After manual
edits, the original `run.json` hashes describe the prepared files, not the edited
files; retain the edited environment files with the results for reproducibility.

Benchmark hashing and study source archives now recurse into `gpe/core`, `gpe/io`
and `config`; otherwise relocated kernels would have disappeared from provenance.

## Next phase

1. Run the analytical GPU tests on Colab and exercise a small editor-created YAML
   using a prepared folder. Check stage transitions and output compatibility.
2. Add the local viewer reading raw diagnostic arrays/CSV. Keep sampling and GUI
   refresh independent, avoid downloading full wavefunctions, and handle files
   that are still being written before enabling live refresh.
3. Add GUI connection/upload/submission controls backed by normal SSH, preserving
   host-key verification and MFA. Use CSD3-documented connection sharing; do not
   store passwords or OTPs in profiles, YAML, logs or the browser draft.
4. Verify the actual CSD3 environment, GPU memory, project account and site Slurm
   template. Submit one small GPU job before production work. Store its job ID and
   provide reconnect/status controls.

Colab is a runtime/output test, not a substitute for CSD3 scheduler/authentication
tests. The lost earlier study was not restarted. No simulation or HPC login was
performed during this refactor.
