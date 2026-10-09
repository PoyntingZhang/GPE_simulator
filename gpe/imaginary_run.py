# this code is intended to launch the simulation
from .config import *
from .grids import get_grids
from .potentials import box_potential, absorption_potential, get_total_static_potential,add_white_noise
from .initial_states import gaussian
from .helper import sum_abs2, normalise, norm_fft, norm_ifft, get_beta
from .propagation import imaginary_step
# initialise 1D x and k arrays
x_array,k_array=get_grids(N_grids,simulation_range)
# initialise psi
psi=gaussian(x_array,3)
# initialise box
box=box_potential(x_array,R,L,wallthickness,U_D)

for i in range(total_steps):
    psi=imaginary_step(psi,box,get_beta(N_atom,a),k_array,img_dt,N_grids,x_array)
    if i%export_step==0:
        export(psi)
    if i%monitor_step==0:
        monitor(psi,k_array,x_array)

    
