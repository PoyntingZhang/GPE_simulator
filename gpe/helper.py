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

sum_partial=cp.RawKernel(
    r'''
#include<cupy/complex.cuh>
extern "C" __global__
void sum_partial(complex<double>* psi, double* partial, const long long count)
{
    
    __shared__ double sums[256];
    const int t=threadIdx.x;
    const long long tid=(long long)blockIdx.x*blockDim.x+t;
    const long long stride=(long long)blockDim.x*gridDim.x;
    double local_sum=0;
    for(long long j=tid;j<count;j+=stride)
    {
        local_sum=local_sum+psi[j].real()*psi[j].real()+psi[j].imag()*psi[j].imag();
    }
    sums[t]=local_sum;
    __syncthreads();
    for (int offset=blockDim.x/2;offset>0;offset/=2)
    {
        if (t<offset)
        {
            sums[t]+=sums[t+offset];
        }
        __syncthreads();
        
    }
    
    if (t==0)
    {   
        partial[blockIdx.x]=sums[0];
    }
}

''',"sum_partial"
)

sum_finish=cp.RawKernel(r'''

    extern "C" __global__
    void sum_finish(double* partial, double* x_array, double* norm, long long previous_block_nums)
    {
        const int t=threadIdx.x;
        __shared__ double sums[256];
        double local_sum=0;
        for (long long j=t;j<previous_block_nums;j+=blockDim.x)
        {
            local_sum+=partial[j];
        }
        sums[t]=local_sum;
        __syncthreads();
        for (int offset=blockDim.x/2;offset>0;offset/=2)
        {
            if(t<offset)
            {
                sums[t]+=sums[t+offset];
            }
            __syncthreads();
        }
        if(t==0)
        {
            double dx=x_array[1]-x_array[0];
            norm[0]=dx*dx*dx*sums[0];
        }

    }
''','sum_finish')
def get_N(psi,x_array):
    threads=256
    blocks=min((psi.size+threads-1)//256,4096)
    partial=cp.empty(blocks,dtype=cp.float64)
    norm=cp.empty((),dtype=cp.float64)
    sum_partial((blocks,),(threads,),(psi,partial,int(psi.size)))
    sum_finish((1,),(threads,),(partial,x_array, norm,int(blocks)))
    return norm


def normalise(psi,x_array):
    norm=get_N(psi,x_array)
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


energy_sum = cp.ReductionKernel(
    'complex128 z, float64 kx, float64 ky, float64 kz',
    'float64 result',
    '(z.real()*z.real()+z.imag()*z.imag())*(kx*kx+ky*ky+kz*kz)',
    'a + b', 'result=a', '0', 'energy_sum', reduce_type='double',
)

angular_sum = cp.ReductionKernel(
    'complex128 z, complex128 momentum, float64 coordinate',
    'float64 result',
    'coordinate*(z.real()*momentum.real()+z.imag()*momentum.imag())',
    'a + b', 'result=a', '0', 'angular_sum', reduce_type='double',
)


def is_due(i,interval):
    return i>0 and interval is not None and interval>0 and i%interval==0


def get_E(psi_k,x_array,k_array):
    # psi_k is the unnormalized fftn of the completed state.
    dx=x_array[1]-x_array[0]
    energy=energy_sum(psi_k,k_array[:,None,None],
                      k_array[None,:,None],k_array[None,None,:])
    return energy*dx**3/psi_k.size


def get_L(psi,x_array,k_array,slab_size=16):
    # Lz = Re integral psi* (x py - y px) psi dV; spectral derivatives.
    dx=x_array[1]-x_array[0]
    L=cp.zeros((),dtype=cp.float64)
    for start in range(0,psi.shape[2],slab_size):
        slab=psi[:,:,start:start+slab_size]
        slab_k=cp.fft.fftn(slab,axes=(0,1))
        momentum=cp.fft.ifftn(slab_k*k_array[None,:,None],axes=(0,1))
        L+=angular_sum(slab,momentum,x_array[:,None,None])
        del momentum
        momentum=cp.fft.ifftn(slab_k*k_array[:,None,None],axes=(0,1))
        L-=angular_sum(slab,momentum,x_array[None,:,None])
        del momentum,slab_k
    return L*dx**3

