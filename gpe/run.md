## get configuratoin(config.yaml)
return: all the information that we need later to launch a simulation:
## grid(total length of simulation box edge, grid number along each direction)
return: 
1. real space 1D array
2. k space 1D array

## box potential(box radius, box length, wall thickness, real space 1D array)
return 
1. the 3D real space trap potential grid cp.double64

## absorption potential(absorption strength, activation distance):
return
1. the real part of absorption potential grid cp.double64

## total static potential
return 
1. real box potential -1j real part of absorption potential
cp.complex 128 

## time dependent potential(V(x,y,z,t))
return V_ijk(xi,xj,xk,t) (elementwise at certain time step and perform element wise calculation)
you could use functools import partial and call a function with some parameters fixed using partial(V,A=2,omega=3)


## get initial psi (of using a gaussian, or load some file)

if gaussian == true, we call gaussian(sigma), also need helper function (renormalise)
or we load a already stored file(might be huge)

return psi as cp.complex128

## start the main simulatoin loop
for i in range(total_steps):
call propagations where inside propagations we use different kernels
(these are the main things)
if store condition is reached, we store the wavefunction
if the display monitor condition is reached, we bring up the monitor!

## Export and monitoring

Call these after a complete step with `i+1`. Each interval is independent;
`0` or `None` disables it, and iteration zero produces no output.

```python
from gpe.export import export
from gpe.monitor import monitor

export(i+1,psi,k_array,x_array,psi_interval=10000,nk_interval=1000,
       N_interval=100,output_dir="output")
monitor(i+1,psi,k_array,x_array,print_scalar_interval=100,
        show_nk_interval=1000,show_TOF_image_interval=1000,
        show_verticle_TOF_interval=1000,show_TOF_slice_interval=1000,
        show_verticle_TOF_slice_interval=1000,show_magnifier_interval=1000,
        show_magnifier_slice_interval=1000,show_phase_interval=1000)
```

- `N` is the wavefunction norm; `E` is dimensionless kinetic energy;
  `L` is dimensionless integrated angular momentum `Lz`, without division by N.
- `nk` is the shell average of `|phi(k)|²`, with
  `phi=dx³ fftn(psi)/(2 pi)^(3/2)`. Shells are centered on multiples of `dk`.
  `counts` counts grid modes, and `Nk=nk*counts*dk³` is the population per shell.
  `sum(Nk)=N`; `Nk/bin_width` approaches `4 pi k² nk` for well-resolved full shells.
  Outer shells intersect the Fourier cube boundary; `complete_shells` flags them.
- Momentum images integrate over `kz` or `kx`; slices use the actual zero-frequency
  plane. Real-space images integrate over z or use the central plane.
- Files are `psi_XXXXXXXX.npy`, its axes/norm companion `psi_XXXXXXXX_grid.npz`,
  `nk_XXXXXXXX.npz`, and an appended `N.csv`. Wavefunctions are saved in slabs.
  These are snapshots, not complete trajectory restarts. Reusing an output directory
  and iteration overwrites that snapshot and appends another N row.
- Both functions accept optional `psi_k=cp.fft.fftn(psi)` to share a transform
  when needed at the same iteration. Supply the completed state's transform,
  not the intermediate transform inside propagation. No transform is cached
  between calls. `Lz` uses separate 2D transforms in z slabs.

On a CUDA machine with CuPy, NumPy, Matplotlib, and pytest installed:
`python -m pytest -q tests/test_diagnostics.py`.

## Colab benchmark

Open `colab_imaginary_benchmark.ipynb` from the project root in Colab and select a
GPU runtime. It copies `MyDrive/GPE_simulator/gpe` to the runtime, so the source
does not need to be pushed to GitHub first. Default settings are 1024³, 200 total
imaginary-time steps, 5 warm-up steps, 5 profiled steps, scalars every 50 steps,
and plots every 200 steps. Warm-up and profiled steps are part of the evolution.

The notebook checks the GPU memory before allocation: the main arrays need 56 GiB
at 1024³, plus FFT workspace. It does not automatically reduce the grid size.
Only the final wavefunction is exported, as `psi_final.npy`, along with its grid,
benchmark timings, scalar history, and run metadata. Results are copied to
`MyDrive/GPE_simulator/benchmark_results/<run_id>`. A 1024³ final state is 16 GiB.
