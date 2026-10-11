import cupy as cp
from .make_kernel import *
from .helper import *

def make_real_propagator(stage):
    # this propagator maker is a funciton that take stage object as an input
    # the output is a funciton that also requires a stage object as an input
    spatial_kernel=make_real_spatial_kernel(stage)
    momentum_kernel=make_real_momentum_kernel(stage)
    if not stage.potential_params["dynamic"]["activate"]:

        def time_independent_propagator(stage):
            psi=stage.psi
            V_static=stage.staticV
            beta=stage.beta
            k_array=stage.k_array
            x_array=stage.x_array
            dt=stage.dt
            N_grids=stage.N_grids
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

            assert x_array.dtype==cp.float64
            assert x_array.shape==(N_grids,)
            assert x_array.flags.c_contiguous

            



            threads=256
            blocks=(psi.size+threads-1)//threads
            grid=(blocks,)
            block=(threads,)
            beta=float(beta)
            dt=float(dt)
            N_grids=int(N_grids)
            spatial_kernel(grid,block,(psi,V_static,x_array,beta,dt,N_grids))
            psi_k=cp.fft.fftn(psi)
            momentum_kernel(grid,block,(psi_k,k_array,dt,N_grids))
            psi=cp.fft.ifftn(psi_k)
            spatial_kernel(grid,block,(psi,V_static,x_array,beta,dt,N_grids))
            return psi
        return time_independent_propagator
    else:
        def time_dependent_propagator(stage,t):
           
            psi=stage.psi
            V_static=stage.staticV
            beta=stage.beta
            k_array=stage.k_array
            x_array=stage.x_array
            dt=stage.dt
            N_grids=stage.N_grids
            assert isinstance(psi,cp.ndarray)
            assert isinstance(V_static, cp.ndarray)
            assert isinstance(x_array, cp.ndarray)
            assert x_array.dtype==cp.float64
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
            assert x_array.shape==(N_grids,)
            assert x_array.flags.c_contiguous

            


            threads=256
            blocks=(psi.size+threads-1)//threads
            grid=(blocks,)
            block=(threads,)
            beta=float(beta)
            dt=float(dt)
            N_grids=int(N_grids)
        
            t_mid=float(t)+dt/2


            spatial_kernel(grid,block,(psi,V_static,x_array,beta,dt,N_grids,t_mid))
            psi_k=cp.fft.fftn(psi)
            momentum_kernel(grid,block,(psi_k,k_array,dt,N_grids))
            psi=cp.fft.ifftn(psi_k)
            spatial_kernel(grid,block,(psi,V_static,x_array,beta,dt,N_grids,t_mid))
            
            return psi
        return time_dependent_propagator

def make_imag_propagator(stage):
    spatial_kernel=make_imag_spatial_kernel(stage)
    momentum_kernel=make_imag_momentum_kernel(stage)
    def standard_imaginary_step(stage):
        psi=stage.psi
        V_static=stage.staticV
        beta=stage.beta
        k_array=stage.k_array
        dt=stage.dt
        N_grids=stage.N_grids
        x_array=stage.x_array
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
        assert x_array.flags.c_contiguous
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
        spatial_kernel(grid,block,(psi,V_static,x_array,beta,dt,N_grids))
        psi_k=cp.fft.fftn(psi)
        del psi
        momentum_kernel(grid,block,(psi_k,k_array,dt,N_grids))
        psi=cp.fft.ifftn(psi_k)
        del psi_k
        spatial_kernel(grid,block,(psi,V_static,x_array,beta,dt,N_grids))
        normalise(psi,x_array)
        return psi
    return standard_imaginary_step
    

                
                
                    
                    