# GPE solver agent handoff

## Diagnostics update, 9 October 2026

The user explicitly requested implementation of export and monitoring, superseding
the earlier no-export scope for these functions. `export.py` and `monitor.py` are
now implemented; `nk_extraction.py` supplies slab-wise GPU shell histograms, and
`helper.py` adds norm, kinetic-energy, and spectral Lz diagnostics. The confirmed
scalar definitions are N = wavefunction norm, E = kinetic energy, L = integrated
Lz. Intervals default to disabled. See `gpe/run.md` for signatures, file formats,
and spectrum conventions. `colab_imaginary_benchmark.ipynb` now provides a separate
Colab driver: 1024³ by default, 200 total steps, GPU memory preflight, warm-up and
stage timings, periodic monitoring, and final-only wavefunction saving to Drive.
The original script is unchanged. CUDA execution and performance measurements
still require a GPU.
The notebook passed schema/syntax validation and a small CPU-adapted execution
covering the memory guard, warm-up/profile scheduling, monitoring, final-only save,
metadata, and destination copy. Its small real-GPU check runs in Colab before the
large allocation; no GPU benchmark has been run locally.
Validation: 18 CPU reference/orchestration checks passed through a temporary NumPy
adapter, including a host compilation of the shell-map C++ expression; plot previews
were inspected. The GPU tests in `tests/test_diagnostics.py` skip locally because
CuPy is unavailable. This does not establish CUDA compilation or GPU correctness.
The user wants to inspect and commit changes manually; do not commit for them.

The current root is the Google Drive `GPE_simulator` folder, not the old Documents
path below. Older placeholder descriptions and undefined-L notes below are historical.

Updated 9 October 2026 after inspection of the saved source and discussion of monitoring, staged experiments, and reusable states. This supersedes older sketches where they conflict. Read [FILE_PLAN.md](FILE_PLAN.md) for the current file map and future implementation milestones. [README.md](README.md) is the original physics outline and contains schematic errors; it is not an executable specification.

## Collaboration and current status

The user is building a three-dimensional Gross–Pitaevskii solver for an experimental group, initially on one GPU, eventually targeting 1024 × 1024 × 1024 points on Cambridge CSD3 Wilkes3. They know Python/CuPy, can read some C++/CUDA, are learning CUDA and HPC jobs, and want to write as much of the implementation as possible. Help through explanations, bounded examples, code review, and concrete next tasks. Do not treat this handoff as a request to implement the whole solver.

The project root is `/Users/zhanghaoyu/Documents/GPE_simulator`; implementation lives in `gpe/`. Documentation is in `GPE simulator/`, alongside an Obsidian vault. Run package commands from the project root, not the documentation directory. Do not move the Obsidian files.

Current implementation uses embedded CuPy RawKernel CUDA C++, a ReductionKernel for normalization, and ordinary out-of-place FFTs. Earlier suggestions for ElementwiseKernel and precomputed one-dimensional kinetic factors are alternatives/future optimizations, not the current code.

**The user wants no wavefunction exports now.** Export/import, checkpoints, and reusable states are requested future capabilities. Do not automatically enable initial, periodic, final, or stage-boundary saves for the immediate smoke run. Monitoring/visualization will be developed later with agents; each diagnostic should be optional and independently scheduled.

The user knowingly retains approximate imaginary-time local exponentials plus normalization after every full step for initial preparation. A more accurate method, possibly inspired by GPELab, is deferred. Do not reopen this accepted choice during unrelated reviews.

All 11 current Python files passed syntax parsing. CuPy is not installed in the local review environment. No CUDA compilation, GPU execution, numerical convergence test, performance benchmark, memory-fit measurement, or successful cluster run has been completed. Static inspection is not GPU validation.

## Inspected implementation

| File | Current behavior |
|---|---|
| `gpe/config.py` | Python constants, not a YAML loader. Current settings: N_grids=512, domain length=60, box R=21/L=42, wall thickness=2, depth=49, imaginary dt=0.01, total_steps=100000, N_atom=300000, a=150, beta=1000, export_step=1000000, monitor_step=1000. |
| `gpe/grids.py` | One-dimensional float64 position axes with endpoint=False and unshifted angular FFT frequencies. |
| `gpe/initial_states.py` | Normalized separable Gaussian. Package-relative `.helper` import is fixed. `cp.exp(..., dtype=cp.complex128)` produces complex factors and a complex128 volume. |
| `gpe/helper.py` | Consistently named squared-modulus reduction, `normalise(psi,x_array)`, physically scaled FFT helpers, and `get_beta`, which now returns beta. |
| `gpe/potentials.py` | Real Gaussian-wall trap, real absorption profile, complex U−iΓ assembly, and slab-wise additive Gaussian disorder. |
| `gpe/kernels.py` | RawKernel definitions for static real-time local, ordinary/masked kinetic, imaginary-time kinetic/local, and preliminary dynamic local operations. |
| `gpe/dynamic_potentials.cuh` | Device helpers for rotating and shaking linear gradients; dynamic integration is incomplete. |
| `gpe/propagation.py` | `imaginary_step`, `ssfmstep`, `ssfmstep_kcut`. Main array checks, launches, signatures, and returns agree by inspection. Only imaginary evolution normalizes. |
| `gpe/imaginary_run.py` | Initializes axes, Gaussian amplitude width sigma=3, and real box, then repeats imaginary steps. Callback and orchestration issues remain below. |
| `gpe/monitor.py`, `gpe/export.py` | `pass` placeholders with signatures `monitor(psi,k_array,x_array)` and `export(psi)`. |
| `gpe/import.py` | Empty placeholder. Prefer a future name such as `load_state.py` because import is a Python keyword; no rename has been made. |
| `gpe/run.md` | Informal workflow notes, not a runnable entry point. |

The latest Gaussian dtype and `get_beta` return fixes are present in saved files. `dtype=` is supported by the CuPy exponential ufunc; a second full-volume conversion is unnecessary. The three identical 1D Gaussian factors could later share one vector, but this is not a blocker. [CuPy exp documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.exp.html)

### Remaining immediate launch issues

1. The driver calls unimported `export` and `monitor` names. Both conditions match i=0, after the first completed step, with export reached first. Explicitly skip export now and skip monitoring until deliberately connected. Importing a pass placeholder avoids a NameError but does not provide diagnostics.
2. A huge export interval does not disable this zero-based condition: the current settings export only after completed step 1. Later use completed-step numbering and explicit enable switches. Do not solve this by forcing a final save against the no-export preference.
3. The driver recalculates `get_beta(N_atom,a)` each iteration. Compute it once for a constant-interaction stage. With a=150 Bohr radii it gives approximately 59848.5325353, while config.beta=1000 is ignored. Choose one source of truth.
4. The driver executes at import time. Add a guarded main and callable stages when composing workflows, so importing a scheme does not build a new Gaussian or start a run.

After skipping the callbacks, the intended command is `python -m gpe.imaginary_run` from the project root in a supported GPU environment. First use a small smoke configuration, not the current 512³/100000-step settings. No solver source was changed during this documentation update.

## Equation, units, grids, and normalization

The chosen dimensionless equation is

\[
i\partial_t\psi=[-\nabla^2+V(\mathbf r,t)+\beta(t)|\psi|^2]\psi.
\]

Reference length l=1 micrometre; E0=ℏ²/(2ml²), t0=2ml²/ℏ. Species/mass is needed for physical energy/time labels and is not yet formalized. The kinetic multiplier is exp(−i k² dt), with no factor of 1/2.

Initially Q=∫|ψ|²d³x=1. With a fixed reference atom number N_ref used to scale ψ, beta(t)=8π N_ref a(t)/l and remaining atom number during loss is N_ref Q. Do not both let norm fall and substitute the remaining number into beta: that double-counts depletion. Loaded lossy states must not be silently renormalized. Rescaling for a new experiment is an explicit operation with a consistent reference number.

`get_beta(N_atoms,a)` currently interprets a as a number of Bohr radii, using a0=5.29177210544e−11 m and l=1e−6 m. Record scattering-length units explicitly.

For a uniform cubic grid, x is on [−D/2,D/2), dx=D/N_grid, dV=dx³. k=2π·fftfreq(N_grid,d=dx), dk=2π/D, in unshifted FFT order. Axis Nyquist magnitude is π/dx, not 2π/dx. Beyond this radius spherical shells are incompletely represented by the Fourier cube. Arrays are (x,y,z) in C order, with the last axis fastest. Current kernels assume identical axes in all directions.

The initial Gaussian is exp[−(x²+y²+z²)/(2σ²)] before normalization; sigma describes its amplitude formula. The README contains obsolete dense grids, an incorrect Nyquist expression, and some derivation/pseudocode errors. Preserve experimental intent while checking formulas.

## Propagation and CUDA contracts

Each wrapper performs local half-step → FFT → kinetic full-step → IFFT → local half-step. Imaginary evolution then normalizes. Pass the full dt; the local kernels divide by two internally. Always assign the returned state, e.g. `psi = imaginary_step(...)`.

The first local kernel mutates the input buffer, but IFFT returns another array and rebinds the local name. A wrapper is therefore not a completely in-place timestep or a pure function. Ignoring the return leaves the caller holding the original, partially evolved buffer.

Required arrays: ψ is C-contiguous complex128; imaginary-time V is a separately supplied C-contiguous float64 real potential; real-time V is C-contiguous complex128 U−iΓ; k is contiguous float64 of shape (N_grid,). The normalization helper reads a float64 x axis through high-level CuPy indexing. Using a real box potential during preparation is intentional. A complex array's `.real` is an interleaved, strided view and must not be passed as a contiguous raw double pointer.

A RawKernel launch is `kernel((blocks,), (threads,), args)`, currently threads=256 and blocks=(psi.size+threads−1)//threads. These are Python tuples describing CUDA launch dimensions, not physical grid axes. Python float/int map to CUDA double/long long for RawKernel argument passing. CuPy supplies array pointers but does not check dtype, shape, or strides against a raw signature; validate these before launch.

Use `tid=(long long)blockDim.x*blockIdx.x+threadIdx.x`, with a bound guard. Cast before large products. For a cube, k=tid%N_grid, j=(tid/N_grid)%N_grid, i=tid/(N_grid*N_grid); the saved modulo expression for j is equivalent. CUDA complex128 is `complex<double>` from `<cupy/complex.cuh>`. Device helper functions execute within an existing thread, not another launch.

Assertions are useful during development; explicit exceptions are preferable for permanent input validation. Reading dtype/shape/flags is metadata work, not a copy of the volume. All modules may allocate arrays; allocation is not restricted to kernels.py.

### Accepted numerical approximations

For a static real potential, the real-time local phase update is exact: density stays constant during that subproblem. Where U−iΓ and nonlinearity coexist, the saved kernel correctly damps amplitude by exp(−Γh), but uses the initial-density nonlinear phase beta|ψ0|²h. The user accepts this remote-absorber approximation for now. If needed later, replace h only in that nonlinear phase by τ=h at Γ=0 and τ=−expm1(−2Γh)/(2Γ) otherwise. This exact correction remains elementwise. Do not renormalize the real-time result.

Imaginary-time evolution changes density even with real V. The current local update ψ←ψ exp[−h(V+beta|ψ|²)] freezes density within each half-step; the second correctly recomputes it after kinetic evolution. The user retains this approximation for preparation. Do not infer exact nonlinear subflows or guaranteed second-order convergence from the half–full–half ordering alone.

If revisited, the exact fixed-V imaginary local update is ψ←ψ exp(−Vh)/sqrt(1+2beta|ψ|²s), with s=h for V=0 and s=−expm1(−2Vh)/(2V) otherwise, on its interval of existence. It is also pointwise; no dense spatial matrix exponential is required. GPELab's inspected ground-state dispatcher uses normalized-gradient Backward Euler/Crank–Nicolson methods; its documented Strang route is real-time dynamics. These are later alternatives, not a requested rewrite now. [GPELab stationary paper](https://www.math.univ-toulouse.fr/~rduboscq/Articles/CPC-GPELab.pdf), [dynamics paper](https://www.math.univ-toulouse.fr/~rduboscq/Articles/CPC-GPELab2.pdf), [source](https://gpelab.math.cnrs.fr/downloads/GPELab.zip).

The masked step zeros modes with k²>kcut² during its kinetic stage, propagates retained modes, and does not normalize. The final local half-step can regenerate high-k components; returned ψ is not guaranteed band-limited. This mask changes the dynamics and is not automatically de-aliasing or equivalent to a trap depth.

### Dynamic work still pending

The preliminary dynamic local kernel hardcodes omega=A=1. Its custom header needs a robust include option based on the directory of kernels.py; being beside a Python module does not reliably put a header on NVRTC's search path. RawKernel compilation is lazy, so this unused path does not block current static steps.

Current helpers are `rotation(x,y,omega,A,t)` and `shake(z,omega,t,A)`. The user's preferred convention was time last; standardize signatures and calls before using shake. A rotating linear gradient is distinct from geometric trap rotation or an angular-momentum operator.

For a time-dependent real drive, define quadrature. Midpoint sampling for both local half-steps is one straightforward second-order choice, while each kernel uses its current ψ. Ordinary Python callbacks cannot execute once per GPU element inside RawKernel. Use device formulas or compute scalar time coefficients in Python and pass them to the kernel.

## Requested monitoring and plots

Monitoring should grow separately from propagation and full-state exports. Every item should have an enable switch and independent computation/display/output cadence. Plan a headless HPC path and later a convenient viewer, without requiring a GUI during a batch run.

| Requested item | Meaning and remaining choices |
|---|---|
| Scalar N | Label dimensionless norm Q separately from atom number N_ref Q. |
| Scalar E | User wants **kinetic energy**, motivated by TOF measurements. Label E_kin; do not silently substitute total GPE energy. Decide dimensionless/physical units and total/per-particle reporting. |
| Scalar L | Requested symbol, still undefined. Clarification was requested; do not assume angular momentum, a particular component, or normalization. |
| n(k), log–log | Isotropic occupation spectrum; establish shell convention below. Handle k=0/nonpositive values for log axes. |
| k²n(k), linear–linear | Requested companion curve; its interpretation depends on the n(k) definition. |
| Momentum image integrated along z | Integrate momentum density over k_z to obtain a (k_x,k_y) color map, not a k_z=0 slice. |
| z=0 density image | Central-plane |ψ(x,y,0)|². On the current even centered grid z=0 is index N_grid//2. |
| z=0 phase image | arg ψ(x,y,0), with explicit treatment of near-empty pixels where phase is not meaningful. |
| Integrated real-space density | Column or finite-thickness slab integral; confirm axis and thickness. Distinct from a central slice. |

The momentum projection is TOF-like. A quantitative experimental TOF image additionally needs a defined expansion protocol, interactions during expansion, flight time, imaging axis/response. Do not equate a simple momentum projection with a complete expansion/imaging model.

Norm after normalization is a check, not evidence of preparation convergence. Optional internal preparation diagnostics can later include total GPE energy/change, chemical potential, and stationary residual ||(−Δ+V+beta|ψ|²−μ)ψ||₂/||ψ||₂. They complement, not replace, the requested E_kin display. Kinetic energy alone need not decrease. Avoid retaining an extra full ψ merely to compare iterates at 1024³. The Gaussian walls fall off outside the box; if a box-localized prepared state is intended, inside-box population is also a useful diagnostic rather than assuming any stationary state is the desired global ground state.

### Fourier and spectrum conventions

Let M=N_grid³ and F=fftn(ψ) with the default unnormalized forward transform. Then Q=dx³/M·Σ|F|² and dimensionless E_kin=dx³/M·Σk²|F|². Compute k² from the 1D axes. The helper norm_fft is equivalently φ=dx³F/(2π)^(3/2), with Q=dk³Σ|φ|². Its physical spectrum scaling is unnecessary inside ordinary propagation, whose FFT/IFFT already form an inverse pair. [CuPy Fourier conventions](https://docs.cupy.dev/en/stable/reference/fft.html#normalization)

With fixed N_ref, physical total kinetic energy is N_ref E0 E_kin; energy per remaining atom is E0 E_kin/Q. Preserve this distinction in labels and metadata.

For shell b, retain mode count C_b and population P_b=dk³Σ_{q in b}|φ_q|². One can form shell-average three-dimensional density nbar_b=Σ|φ|²/C_b, or radial population density S_b=P_b/Δk_b. Atom population is N_ref P_b. Save bin edges/counts with a spectrum.

If n(k)=nbar(k), then Q≈4π∫k²nbar dk and E_kin≈4π∫k⁴nbar dk. Therefore k²nbar is proportional to radial population, not kinetic-energy density. If n(k)=S(k), then Q=∫S dk and E_kin=∫k²S dk. **Settle this convention before attaching an energy interpretation to k²n(k).** Use exact discrete bin weights for checks and document incomplete shells near the Fourier cube edges.

A z-integrated momentum projection is dk·Σ_kz|φ|²; a z-integrated real-space density is dx·Σ_z|ψ|². Multiply by N_ref when reporting atom-normalized quantities. Shift small plotting axes/results as needed, not the permanent evolving full volume.

When momentum diagnostics are due together, share one FFT of the completed current ψ. The propagation intermediate ψ_k precedes the final local half-step and is not the returned state's spectrum. Invalidate cached diagnostics when ψ changes. Reduce spectra, slices, and projections on the GPU and transfer small results only. Small spectra are cheap to store, but require an FFT and volume-wide reduction to compute. Avoid permanent 3D radii, bin IDs, or density arrays; use fused reductions or bounded slabs where appropriate.

## Staged experiments and reusable states

The user wants callable evolution schemes composing a longer experiment:

1. Generate a state or load a chosen saved state.
2. Optionally prepare it by normalized imaginary evolution.
3. Drive/shake the box to create a desired excited state.
4. Change/turn on interaction and watch subsequent real-time relaxation.
5. Optionally use masked real-time evolution.
6. Eventually save a useful driven state and reuse it, bypassing earlier preparation.

This is the requested workflow, not a fully specified protocol. Do not assume beta was zero in earlier stages just because the example later says to turn on interaction. Each stage should specify duration/dt, beta or scattering length, trap/absorber, drive parameters and phase/time origin, cutoff, and diagnostics.

Separate single-step functions from callable stage loops and small entry scripts. A stage consumes the existing ψ/grid and returns updated state; it must not recreate a Gaussian implicitly. State bookkeeping needs grid/units, N_ref, physical time and global real step count, stage identity/local progress, and evolving drive/random state. Imaginary preparation time is separate from experimental real time. A sudden quench changes parameters with ψ continuous; a ramp needs an explicit schedule. Transition between float64 preparation potential and complex128 real-time potential deliberately, without keeping unnecessary large duplicates.

Real-time relaxation means the requested subsequent evolution, e.g. drive-off dynamics. It does not authorize adding damping or renormalization and does not guarantee ground-state convergence under a conservative GPE.

Saving a reusable prepared state differs from resuming an interrupted trajectory. Reuse may intentionally reset the clock and change parameters. Continuation must restore stage progress and drive phase/time origin. Do not silently normalize or reinterpret an imported state.

Future state metadata should include ψ, shape/dtype/order, axes or domain/spacing, units and mass if known, N_ref and Q, physical/stage time and progress, beta/scattering length, potential/disorder/absorber specification, drive phase/schedule, cutoff, timestep, FFT conventions, and configuration/software identity. Include RNG state when continued stochastic dynamics requires it; a seed alone need not restore mid-run randomness. Validate imports; interpolation or rescaling is an explicit operation.

Format is undecided. NPY plus metadata is a simple candidate; HDF5 offers grouping and partial access, not an automatic bandwidth advantage. Compact diagnostics cannot reconstruct ψ or replace restart checkpoints. Budget CPU staging and filesystem writes. Keep future saving opt-in; no exports now.

## Memory, performance, and HPC

| Array | 512³ | 1024³ |
|---|---:|---:|
| float64 volume | 1 GiB | 8 GiB |
| complex128 volume | 2 GiB | 16 GiB |

At 1024³, ψ+ψ_k+complex V totals 48 GiB; with a real preparation potential, 40 GiB. During the current inverse FFT, the caller's original ψ can coexist with ψ_k and the new inverse result: three complex volumes plus V give 64 GiB in real time or 56 GiB in imaginary time, before FFT workspace, plans/pools, initialization, diagnostics, and IO staging. These are accounting estimates, not measured peaks or proof of fit.

Keep ψ resident, axes 1D, and pointwise work fused. Current kinetic kernels calculate k²/exp per point. Precomputed 1D factors exp(−ik_axis²dt) multiplied in one fused pass remain a future optimization; do not form their full outer product. Keep a time-dependent potential as per-point computation/scalar coefficients rather than another full volume.

High-level FFTs return arrays. Preallocating a variable and then rebinding it to an FFT does not reuse its storage; assigning the result via [...] still creates the FFT result before copying. A plan is reusable transform setup and possible workspace. CuPy caches plans and allocations; del removes references, but the pool can retain blocks. Profile before lower-level buffer reuse or in-place FFT; neither guarantees zero workspace or higher speed. Preserve normalization when moving to raw cuFFT, whose inverse is unnormalized.

Passing arrays between functions does not copy them. Ordinary chained CuPy expressions can create full temporaries even with a final *=. Scalar intermediates inside a kernel are not full arrays. Python per-element GPU loops are too slow; bounded slab loops can be appropriate. Current disorder generation uses a contiguous float64 slab buffer with Generator.standard_normal, then adds into the potential; do not pass a strided complex .real view as its contiguous out argument or reseed each slab. Document disorder correlation/seed, since per-cell white noise changes with resolution.

Cambridge documents Wilkes3 as four A100-SXM-80GB GPUs per node in the ampere Slurm partition. Start with one GPU; four GPUs do not automatically share one combined memory allocation. Confirm account/project, environment, versions, and actual free memory during onboarding. Slurm host RAM and GPU VRAM are different resources. [CSD3 Wilkes3 documentation](https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html)

The workflow is local development, transfer of Python and CUDA headers, then short allocated GPU runs. A .sbatch file requests known resources and launches the program; runtime inspection records the allocation rather than choosing it automatically. No scheduler script or GPU test result exists yet.

First validate small-grid launches, FFT round trips, known phases/norm behavior, preparation diagnostics, and cutoff semantics. Later check composed stages and restart against uninterrupted runs. Measure warmed full timesteps and peak memory, including enabled diagnostics and eventual IO, while increasing resolution. Keep out-of-place FFT if measured fit/speed is satisfactory. Multi-GPU decomposition and a C++ rewrite remain deferred.

## Unresolved choices and next work

Use FILE_PLAN.md's bounded milestones rather than implementing everything at once. Open choices include: meaning of L; n(k) convention; atom/energy units; integrated-image axis/slab thickness; plot layout and independent schedules; stage interactions and drive phase/ramp schedules; reuse versus exact resume and storage format; species/mass and cluster environment; later preparation accuracy/convergence thresholds.

Python config constants are current. YAML/TOML simulation configuration and a CLI can be added later; pyproject.toml has a separate packaging role. No particular class framework or final configuration schema is mandated. Preserve the user's ownership of implementation.

## References

- [CuPy user-defined kernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html), [RawKernel](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html): launch and argument contracts.
- [CuPy memory](https://docs.cupy.dev/en/stable/user_guide/memory.html), [FFT guide](https://docs.cupy.dev/en/stable/user_guide/fft.html), [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html): allocation, plans, and future optimization.
- [Bao and Du, normalized gradient flow](https://arxiv.org/pdf/cond-mat/0303241): ground-state methods and exact local imaginary-time updates; adapt its kinetic coefficient to this project's convention.
- [Bao and Jaksch, damped splitting](https://arxiv.org/pdf/math/0303158): exact damping/nonlinear phase treatment.
- [GPELab documentation](https://gpelab.math.cnrs.fr/downloads.htm): reference implementation, not a requirement to replace the accepted current method.

Reread the actual files before continuing: the user is actively editing them, and this handoff records the inspected state on the date above.
