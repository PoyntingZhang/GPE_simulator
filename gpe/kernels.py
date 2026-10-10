import cupy as cp

static_real_halfstep = cp.RawKernel(
r'''
#include"static_potentials.cuh"
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
    
    double abs2psi=psi[tid].real()*psi[tid].real()+psi[tid].imag()*psi[tid].imag();
    double V_static_real=V_static[tid].real();
    double V_static_imag=V_static[tid].imag();
    double extra_V=kick(x_array[j],1);
    double real_phase= (V_static_real+beta*abs2psi+extra_V)*(dt/2);
    double imag_phase=V_static_imag*(dt/2);
    complex<double> propagator=exp(complex<double>(imag_phase,-real_phase));
    psi[tid]=psi[tid]*propagator;
    
}
''', 'static_real_halfstep', options=('-Igpe',))


dynamic_real_halfstep = cp.RawKernel(
r'''
#include <cupy/complex.cuh>
#include"dynamic_potentials.cuh"
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
    const double time_potential=rotation(x,y,0.065,0.015,t);
    double abs2psi=psi[tid].real()*psi[tid].real()+psi[tid].imag()*psi[tid].imag();
    double V_static_real=V_static[tid].real();
    double V_static_imag=V_static[tid].imag();
    double real_phase= (V_static_real+beta*abs2psi+time_potential)*(dt/2);
    double imag_phase=V_static_imag*(dt/2);
    complex<double> propagator=exp(complex<double>(imag_phase,-real_phase));
    psi[tid]=psi[tid]*propagator;
    
}
''', 'dynamic_real_halfstep', options=('-Igpe',))



kinetic_full_step=cp.RawKernel(
    r'''
#include<cupy/complex.cuh>
extern "C" __global__
void kinetic_full_step(complex<double>* psi_k, const double* k_array,const double dt, const long long N_grid )
{  
    const long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long count=(long long)N_grid*N_grid*N_grid;
    if (tid>=count) return;
    long long k=tid%N_grid;
    long long j=(tid%((long long)N_grid*N_grid))/N_grid;
    long long i=tid/((long long)N_grid*N_grid);
    double K_2=k_array[i]*k_array[i]+k_array[j]*k_array[j]+k_array[k]*k_array[k];
    double phase=K_2*dt;
    complex<double> propagator=exp(complex<double>(0,-phase));
    psi_k[tid]=psi_k[tid]*propagator;
}
'''
,'kinetic_full_step')


masked_kinetic_full_step=cp.RawKernel(
    r'''
#include<cupy/complex.cuh>
extern "C" __global__
void masked_kinetic_full_step(complex<double>* psi_k, const double* k_array,const double dt, const long long N_grid, double k_mask )
{  
    const long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long count=(long long)N_grid*N_grid*N_grid;
    if (tid>=count) return;
    long long k=tid%N_grid;
    long long j=(tid%((long long)N_grid*N_grid))/N_grid;
    long long i=tid/((long long)N_grid*N_grid);
    double K_2=k_array[i]*k_array[i]+k_array[j]*k_array[j]+k_array[k]*k_array[k];
    if (K_2>k_mask*k_mask){
    psi_k[tid]=complex<double>(0,0);
    return;}
    double phase=K_2*dt;
    complex<double> propagator=exp(complex<double>(0,-phase));
    psi_k[tid]=psi_k[tid]*propagator;
}
'''
,'masked_kinetic_full_step')


imag_kinetic_full_step=cp.RawKernel(
    r'''
#include<cupy/complex.cuh>
extern "C" __global__
void imag_kinetic_full_step(complex<double>* psi_k, const double* k_array,const double dt, const long long N_grid )
{  
    const long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long count=(long long)N_grid*N_grid*N_grid;
    if (tid>=count) return;
    long long k=tid%N_grid;
    long long j=(tid%((long long)N_grid*N_grid))/N_grid;
    long long i=tid/((long long)N_grid*N_grid);
    double K_2=k_array[i]*k_array[i]+k_array[j]*k_array[j]+k_array[k]*k_array[k];
    double phase=K_2*dt;
    complex<double> propagator=exp(complex<double>(-phase,0));
    psi_k[tid]=psi_k[tid]*propagator;
}
'''
,'imag_kinetic_full_step')

imag_static_real_halfstep = cp.RawKernel(
r'''
#include <cupy/complex.cuh>
extern "C" __global__
void imag_static_real_halfstep(complex<double>* psi, const double* V_static, 
const double beta, const double dt,const long long N_grid ) 

{
    const long long tid =(long long) blockDim.x * blockIdx.x + threadIdx.x;
    const long long count=(long long)N_grid*N_grid*N_grid;
    if (tid>=count) return;
    
    double abs2psi=psi[tid].real()*psi[tid].real()+psi[tid].imag()*psi[tid].imag();
    double V_static_real=V_static[tid];
    double real_phase= (V_static_real+beta*abs2psi)*(dt/2);
    complex<double> propagator=exp(complex<double>(-real_phase,0));
    psi[tid]=psi[tid]*propagator;
    
}
''', 'imag_static_real_halfstep')