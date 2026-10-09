import cupy as cp

sum_abs2 = cp.ReductionKernel(
    'complex128 z',  # input params
    'float64 result',  # output params
    'z.real()*z.real()+z.imag()*z.imag()',  # map
    'a + b',  # reduce
    'result=a',  # post-reduction map
    '0',  # identity value
    'sum_abs2' , # kernel name
    reduce_type='double',
)

def normalise(psi,x_array):
    dx=x_array[1]-x_array[0]
    norm=dx**3*sum_abs2(psi)
    psi/=cp.sqrt(norm)

def norm_fft(psi,x_array,k_array):
    dx=x_array[1]-x_array[0]
    dk=k_array[1]-k_array[0]
    psi_k=cp.fft.fftn(psi,norm='ortho')
    psi_k*=(dx/dk)**(3/2)
    return psi_k
def norm_ifft(psi_k,x_array,k_array):
    dx=x_array[1]-x_array[0]
    dk=k_array[1]-k_array[0]
    psi=cp.fft.ifftn(psi_k,norm='ortho')
    psi*=(dk/dx)**(3/2)
    return psi


def get_beta(N_atoms,a):
    a0=5.29177210544e-11
    l=1e-6
    beta=8*cp.pi*N_atoms*(a*a0)/l
    return beta


