from .kernels import *
from .helper import normalise
import cupy as cp


def imaginary_step(psi,V_static,beta,k_array,dt,N_grids,x_array):
    assert isinstance(psi,cp.ndarray)
    assert isinstance(V_static,cp.ndarray)
    assert psi.dtype==cp.complex128
    assert V_static.dtype==cp.float64 #here for imaginary time evolution we require
    # potential being real
    assert psi.flags.c_contiguous
    assert V_static.flags.c_contiguous
    assert psi.shape==(N_grids,N_grids,N_grids)
    assert V_static.shape==psi.shape 
    assert isinstance(x_array,cp.ndarray)
    assert x_array.shape==(N_grids,)
    assert x_array.dtype==cp.float64
    assert isinstance(k_array, cp.ndarray)
    assert k_array.dtype == cp.float64
    assert k_array.shape == (N_grids,)
    assert k_array.flags.c_contiguous
    
    
    threads=256;
    blocks=(psi.size+threads-1)//threads
    grid=(blocks,)
    block=(threads,)
    beta=float(beta)
    dt=float(dt)
    N_grids=int(N_grids)

    imag_static_real_halfstep(grid,block,(psi,V_static,beta,dt,N_grids))
    psi_k=cp.fft.fftn(psi)
    del psi
    imag_kinetic_full_step(grid,block,(psi_k,k_array,dt,N_grids))
    psi=cp.fft.ifftn(psi_k)
    del psi_k
    imag_static_real_halfstep(grid,block,(psi,V_static,beta,dt,N_grids))
    normalise(psi,x_array)
    return psi

def ssfmstep(psi,V_static,beta,k_array,dt,N_grids):
    assert isinstance(psi,cp.ndarray)
    assert isinstance(V_static, cp.ndarray)
    assert psi.dtype==cp.complex128
    assert V_static.dtype==cp.complex128
    assert psi.flags.c_contiguous
    assert V_static.flags.c_contiguous
    assert psi.shape==(N_grids,N_grids,N_grids)
    assert V_static.shape==psi.shape
    assert isinstance(k_array,cp.ndarray)
    assert k_array.dtype==cp.float64
    assert k_array.shape==(N_grids,)
    assert k_array.flags.c_contiguous


    threads=256
    blocks=(psi.size+threads-1)//threads
    grid=(blocks,)
    block=(threads,)
    beta=float(beta)
    dt=float(dt)
    N_grids=int(N_grids)


    static_real_halfstep(grid,block,(psi,V_static,beta,dt,N_grids))
    psi_k=cp.fft.fftn(psi)
    kinetic_full_step(grid,block,(psi_k,k_array,dt,N_grids))
    psi=cp.fft.ifftn(psi_k)
    static_real_halfstep(grid,block,(psi,V_static,beta,dt,N_grids))
    
    return psi
    
def ssfmstep_kcut(psi,V_static,beta,k_array,dt,N_grids,kcut):
    assert isinstance(psi,cp.ndarray)
    assert isinstance(V_static, cp.ndarray)
    assert psi.dtype==cp.complex128
    assert V_static.dtype==cp.complex128
    assert psi.flags.c_contiguous
    assert V_static.flags.c_contiguous
    assert psi.shape==(N_grids,N_grids,N_grids)
    assert V_static.shape==psi.shape
    assert isinstance(k_array,cp.ndarray)
    assert k_array.dtype==cp.float64
    assert k_array.shape==(N_grids,)
    assert k_array.flags.c_contiguous
    


    threads=256
    blocks=(psi.size+threads-1)//threads
    grid=(blocks,)
    block=(threads,)
    beta=float(beta)
    dt=float(dt)
    N_grids=int(N_grids)
    kcut=float(kcut)


    static_real_halfstep(grid,block,(psi,V_static,beta,dt,N_grids))
    psi_k=cp.fft.fftn(psi)
    masked_kinetic_full_step(grid,block,(psi_k,k_array,dt,N_grids,kcut))
    psi=cp.fft.ifftn(psi_k)
    static_real_halfstep(grid,block,(psi,V_static,beta,dt,N_grids))
    
    return psi







