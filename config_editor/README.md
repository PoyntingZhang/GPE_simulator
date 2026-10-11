# GPE configuration editor

A local browser UI for creating and reopening staged simulation configurations.
It needs Python 3.10+ and PyYAML. It does not need CuPy, a GPU, Node, or an internet
connection after installation. All assets and data stay on the local computer.

From the repository root:

```sh
python3 -m venv config_editor/.venv
config_editor/.venv/bin/python -m pip install -r config_editor/requirements.txt
config_editor/.venv/bin/python config_editor/app.py
```

The last command opens `http://127.0.0.1:8765`. Stop it with Ctrl+C. Use
`--port 8766` if the port is occupied, or `--no-browser` to open the address yourself.
On Windows, the virtual environment's Python is `.venv\Scripts\python.exe`.

## Using the editor

1. Set the shared grid, simulation range, and output root **on the HPC**.
2. Add, duplicate, reorder, or remove stages. Names are optional; stages remain
   numbered in sequence. Each stage has Evolution, Potentials, and Monitor & save tabs.
3. Use **Show YAML preview** to inspect the configuration. Validation messages
   link back to the relevant field.
4. **Save YAML** writes to the chosen **local** path. Its parent folder must exist.
   Existing files require the explicit Replace existing file option. Download instead
   uses the browser's download location.
5. **Open YAML** reopens an exported file or a hand-written schema-2 configuration.
   Unknown settings and duplicate keys are rejected rather than silently discarded.

A valid browser draft is restored on reopening the editor. Export YAML for a
durable copy. `examples/kick_and_hold.yaml` can be opened as an editable illustration.
It is not a calibrated experiment.

The revised potential format uses schema 2. Earlier schema-1 YAML files must be
updated to the two expression blocks below; separate parameters and CUDA bodies
are no longer accepted. Earlier browser drafts remain stored under their old key;
the new editor starts a separate draft. Noise now always covers the whole grid.

## Configuration conventions

- Double precision is fixed; there is no precision option. `schema_version: 2`
  identifies the configuration format.
- Global settings live in `simulation`; `stages` is an ordered list. Missing
  optional stage fields receive editor defaults. There is no implicit inheritance
  from the preceding stage.
- New experiment defaults are read as literals from `gpe/config.py`, without
  importing or executing the simulation. If that file is absent, built-in defaults
  use a 256³ grid over a range of 70.
- Lengths and times are dimensionless. `cloud.a` is in Bohr radii; `cloud.N_atom`
  is the reference atom number used to calculate beta, not an instruction to
  renormalize a continued real-time state.
- Evolution uses `type: imaginary` or `real`, `steps`, and `dt`. Every stage starts
  at time zero. Imaginary evolution normalizes each step; time-dependent potentials
  and absorbers are locked off and rejected on import if enabled.
- Initial states use `type: gaussian`, `load`, or `continue`. `sigma` is the Gaussian
  width; `path` is the HPC path to a saved wavefunction. Continue carries the previous
  stage's final wavefunction and norm. Stage 1 cannot continue.
- `potential.static` and `potential.dynamic` each have `enabled` and `expression`.
  Both contributions can be active at once, added to the box and disorder.
  Static expressions accept `x, y, z`; dynamic expressions also accept `t`.
  Coefficients go directly in the formula; there is no parameter block.
- Box geometry uses `R`, `L`, `trap_depth`, and `wall_thickness`. Absorber settings
  are `activation_distance` and `strength`. Their `enabled` flags control inclusion.
- Noise has Gaussian statistics and standard deviation
  `fraction_of_trap_depth * box.trap_depth`. It therefore requires an enabled box.
  `correlation_length` is a spatial Gaussian smoothing length (0 = uncorrelated).
  Disorder covers the entire simulation region without a spatial mask. A runner
  must rescale smoothed noise to the intended RMS. The same seed, grid, and generation
  method reproduce the base pattern; trap depth determines its stage-specific amplitude.
- Monitor and save intervals count **completed steps within the stage**; 0 disables.
  Initial and final wavefunction saves have separate flags. Monitoring on CSD3 is
  intended to write small diagnostic arrays and scalar records for a local viewer.
- Blank `save.directory` means `stage_01_Name`, etc. beneath `simulation.output_root`.
  A relative override is beneath that root; an absolute path replaces it. Inactive
  settings remain in YAML so switching an option back on preserves its values.

## Potential formulas

Use Python-style mathematical expressions with `+ - * / **` and parentheses.
Supported functions are `sin`, `cos`, `tan`, `exp`, `sqrt`, `tanh`, `abs`, and `log`
(natural logarithm), each taking one argument. `pi` is available; angles use radians.
Use explicit multiplication and bare function names, such as `sin(x)`, not `np.sin(x)`.

```yaml
potential:
  static:
    enabled: true
    expression: "0.03*x + 0.002*y**2"
  dynamic:
    enabled: true
    expression: "0.015*(sin(0.065*t)*x + cos(0.065*t)*y)"
```

The editor checks formulas while typing. Invalid input appears below the field in
small red text with an explanation, such as `Unrecognised variable A.` Valid input
appears as green mathematical notation, using native MathML without an online service.
The same parser checks imported and exported YAML; it never executes the expression.
These checks cover syntax and allowed names, not values across the simulation grid
(for example, a syntactically valid `1/x` can still be singular).

Entering a literal zero, including `0.0`, switches the contribution off when you leave
the field. A manual Off switch preserves the formula. Editing an off contribution
keeps it off until you enable it. Zero is also normalized to off when saving/importing.

## Scope

This folder implements the configuration editor and YAML contract only. The
current `gpe` package does **not** yet read this schema or provide the staged CLI
runner. Kernel generation, CSD3 submission, restart execution, and the remote
monitoring viewer are separate components still to be implemented. No existing
simulation source or notebook is changed by the editor. Its HTTP server binds
only to loopback and is intended to run on your local device, not as a public service.

## Checks

```sh
config_editor/.venv/bin/python -m unittest discover -s config_editor/tests -v
```

These check formula syntax and notation, allowed variables, YAML roundtrips,
imaginary-time restrictions, stage settings, and saving without overwriting files.
