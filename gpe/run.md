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


