import cupy as cp
from .kernels import *

def make_real_spatial_kernel(stage)->cp.RawKernel:
    assert stage.type=="real"

    if stage.potential_params["dynamic"]["activate"]:
        time_dependent_expression=stage.potential_params["dynamic"]["expression"]
        if stage.potential_params["static"]["activate"]:
            time_independent_expression=stage.potential_params["static"]["expression"]
        else:
            time_independent_expression="0"

        dynamic_real_halfstep = cp.RawKernel(
            r'''
            #include <cupy/complex.cuh>
            
            extern "C" __global__
            void dynamic_real_halfstep(complex<double>* psi, const complex<double>* V_static, 
            const double* x_array, const double beta, const double dt,const long long N_grid, double t ) 
        
            {
                const long long tid =(long long) blockDim.x * blockIdx.x + threadIdx.x;
                const long long count=(long long)N_grid*N_grid*N_grid;
                if (tid>=count) return;
                long long k=tid%N_grid;
                long long j=(tid%((long long)N_grid*N_grid))/N_grid;
                long long i=tid/((long long)N_grid*N_grid);
                const double x=x_array[i];
                const double y=x_array[j];
                const double z=x_array[k];
                const double time_potential=__TIME_POTENTIAL__;
                const double time_independent_potential=__TIME_INDEPENDENT_POTENTIAL__;
                double abs2psi=psi[tid].real()*psi[tid].real()+psi[tid].imag()*psi[tid].imag();
                double V_static_real=V_static[tid].real();
                double V_static_imag=V_static[tid].imag();
                double real_phase= (V_static_real+beta*abs2psi+time_potential+time_independent_potential)*(dt/2);
                double imag_phase=V_static_imag*(dt/2);
                complex<double> propagator=exp(complex<double>(imag_phase,-real_phase));
                psi[tid]=psi[tid]*propagator;
                
            }
            '''.replace("__TIME_POTENTIAL__",time_dependent_expression).replace("__TIME_INDEPENDENT_POTENTIAL__",
                                                                                time_independent_expression), 
            'dynamic_real_halfstep')
        return dynamic_real_halfstep
    else:
        if stage.potential_params["static"]["activate"]:

            time_independent_expression=str(stage.potential_params["static"]["expression"])
        else:
            time_independent_expression="0"
        
        static_real_halfstep = cp.RawKernel(
        r'''
        #include <cupy/complex.cuh>
        
        extern "C" __global__
        void static_real_halfstep(complex<double>* psi, const complex<double>* V_static, 
        const double* x_array, const double beta, const double dt,const long long N_grid ) 

        {
            const long long tid =(long long) blockDim.x * blockIdx.x + threadIdx.x;
            const long long count=(long long)N_grid*N_grid*N_grid;
            if (tid>=count) return;
            long long k=tid%N_grid;
            long long j=(tid%((long long)N_grid*N_grid))/N_grid;
            long long i=tid/((long long)N_grid*N_grid);
            const double x=x_array[i];
            const double y=x_array[j];
            const double z=x_array[k];
            const double time_independent_potential=__TIME_INDEPENDENT_POTENTIAL__;
            double abs2psi=psi[tid].real()*psi[tid].real()+psi[tid].imag()*psi[tid].imag();
            double V_static_real=V_static[tid].real();
            double V_static_imag=V_static[tid].imag();
            double real_phase= (V_static_real+beta*abs2psi+time_independent_potential)*(dt/2);
            double imag_phase=V_static_imag*(dt/2);
            complex<double> propagator=exp(complex<double>(imag_phase,-real_phase));
            psi[tid]=psi[tid]*propagator;
            
        }
        '''.replace("__TIME_INDEPENDENT_POTENTIAL__",time_independent_expression), 
        'static_real_halfstep')
        return static_real_halfstep
        
def make_imag_spatial_kernel(stage)->cp.RawKernel:
    
    if stage.potential_params["static"]["activate"]:
    
        time_independent_expression=str(stage.potential_params["static"]["expression"])
    else:
        time_independent_expression="0"
    imag_static_real_halfstep = cp.RawKernel(
    r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void imag_static_real_halfstep(complex<double>* psi, const double* V_static,const double* x_array, 
    const double beta, const double dt,const long long N_grid ) 

    {
        const long long tid =(long long) blockDim.x * blockIdx.x + threadIdx.x;
        const long long count=(long long)N_grid*N_grid*N_grid;
        if (tid>=count) return;
        long long k=tid%N_grid;
        long long j=(tid%((long long)N_grid*N_grid))/N_grid;
        long long i=tid/((long long)N_grid*N_grid);
        const double x=x_array[i];
        const double y=x_array[j];
        const double z=x_array[k];
        const double time_independent_potential=__TIME_INDEPENDENT_POTENTIAL__;

        double abs2psi=psi[tid].real()*psi[tid].real()+psi[tid].imag()*psi[tid].imag();
        double V_static_real=V_static[tid];
        double real_phase= (V_static_real+beta*abs2psi+time_independent_potential)*(dt/2);
        complex<double> propagator=exp(complex<double>(-real_phase,0));
        psi[tid]=psi[tid]*propagator;
        
    }
    '''.replace("__TIME_INDEPENDENT_POTENTIAL__",time_independent_expression), 'imag_static_real_halfstep')
    return imag_static_real_halfstep

def make_real_momentum_kernel(stage)->cp.RawKernel:
    if stage.k_cut_params["activate"]:
        k_cut=stage.k_cut_params["k_cut"]
        return make_masked_kinetic_full_step(k=k_cut)
    else:
        return kinetic_full_step
    

    
        
        


def make_imag_momentum_kernel(stage)->cp.RawKernel:
    return imag_kinetic_full_step