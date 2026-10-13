ultimate goal: run GPE simulation on hpc clusters, using ssh login
## Grid 
default is to perform simulation in a 3D periodic boundary condition cube
parameters: cube length (in $\mathrm{\mu m}$) D, grid number (256,512,1024...) N
upon defining the spatial grid we could define the K grids
the spatial grid and K grid share the same array shape ($N \times N \times N$)
so the minimal spatial grid size $dx=\frac{D}{N}$, corresponding to the kgrid's range: $k_{max}=\frac{2\pi}{dx}=\frac{2\pi N}{D}$
the finest k grid size: $dk=\frac{2\pi}{D}$, therefore the k array should be: 
```python
k_grid=np.arange(-N*np.pi/L,N*np.pi/L,2*pi/L)
```
## Time independent potential
### Box trap:
Here we have an option to use a box trap or not
if we use box trap then we need to define the 3D box trap:
with parameter R and L, and trap depth U_D(in dimensionless unit)
some artificial parameter wall thickness (to replace the experimental power law trap with trap wall being Gaussian):
```python
import Grid
grid = Grid(...)
X=grid.X
Y=grid.Y
Z=grid.Z
sheets=np.exp(-(np.abs(Z)-L/2)**2/(2*thickness**2))
tube=np.exp(-(np.sqrt(X**2+Y**2)-R)**2/(2*thickness)**2)
box_trap= U_D*np.max(sheets, tube)
```
### Absorption potenital:
Upon choosing a box trap we shall define absorption potential at edge of the simulation region, i..e near the boundary of the total cube:
parameter: activation range r_act
maximum absorption amplitude: Absorption_max
```python

minimum_d=np.min((X+(D/2)),(X-(D/2)),(Y+(D/2)),(Y-(D/2)),(Z+(D/2)),(Z-(D/2)))

V_absorb=-1j*Absorption_max*np.max((r_act-minimim_d),0)/r_act

```

## Non-box trap:

the the simulation is performed in a 3 dimensional periodic boundary condition cubic box, this is suitable for problems that does not care about the box geometry, we could either later impose a U_D in Fourier space, or just use the Nyquist wave vector directly from the grid which gives:
$k_{D}=\frac{\pi N}{D}$ so this quantity is proportional to the number of grid and inversely proportional to the simulation region's linear dimension.


## Gaussian noise:
parameter: fraction to the trap depth, in this example we choose 1%
```python

rng=np.random.default_rng(42)
epsilon=0.01 # noise is one percent of the trap depth
sigma=epsilon*U_D
V_disorder=rng.normal(loc=0,scales=sigma,size=(grid.X,grid.Y,grid.Z))

```

## Fourier Split Step Method
##
```python
def ssfm_static(psi,V,grid,beta,dt,k_mask=None):
	V=V_static+beta*np.abs(psi)**2
	psi*=np.exp(-1j*(dt/2)*V)
	psi_k=np.fft.fftn(psi)
	psi_k*=np.exp(-1j*dt*grid.Ksq)
	
	if k_mask is not None:
		mask=np.sqrt(grid.Ksq)<k_mask
		psi_k*=mask
		
	psi=np.fft.ifftn(psi_k)
	V=V_static+beta*np.abs(psi)**2
	psi*=np.exp(-1j*(dt/2)*V)
	return psi
```

```python
def ssfm_dynamic(psi,V,grid,beta,dt,t,k_mask=None):
	V=V_static+beta*np.abs(psi)**2+V_dyn(t+dt/2)
	psi*=np.exp(-1j*(dt/2)*V)
	psi_k=np.fft.fftn(psi)
	psi_k*=np.exp(-1j*dt*grid.Ksq)
		
	if k_mask is not None:
		mask=np.sqrt(grid.Ksq)<k_mask
		psi_k*=mask
		
	psi=np.fft.ifftn(psi_k)
	V=V_static+beta*np.abs(psi)**2+V_dyn(t+dt/2)
	psi*=np.exp(-1j*(dt/2)*V)
	return psi
```

```python

def ssfm_img(psi,V,grid,beta,dt):
	V=V_static+beta*np.abs(psi)**2
	psi*=np.exp(-1j*(-1j*dt/2)*V)
	psi_k=np.fft.fftn(psi)
	psi_k*=np.exp(-1j*(-1j*dt)*grid.Ksq)
	psi=np.fft.ifftn(psi_k)
	V=V_static+beta*np.abs(psi)**2
	psi*=np.exp(-1j*((-1j*dt)/2)*V)
	return psi
```



## helper functions:
```python 
def get_norm(psi,grid):
	return sum(np.abs(psi)**2)*grid.dV

def get_totalK(psi,grid):
	psi_k= np.fft.fftn(psi)*grid.dV/(2*np.pi)**(1.5)
	return np.sum(np.abs(psi_k)**2*grid.Ksq)
def normalise_psi(psi):
	psi*=1/np.sqrt(get_norm(psi))
```
## Initial states:
### generating an Gaussian initial state:
```python
def gaussian_initial(grid,sigma):
	return normalise_psi(np.exp(-grid.R**2/2*sigma**2))
```
notice that the it is the amplitude of the initial wave function has width sigma, not the probability density

### getting true G.S. by imaginary time evolution
```python
imaginary_steps=10000
dt_i=0.01
psi=gaussian_initial(grid,sigma)
for i in range(imaginary_steps):
	psi=ssfm_img(psi,grid,beta,dt_i)
psi_gs=psi
```
### getting g.s. using more sophisticated method:
we could perform zero a state by adding a periodic shake potential and also create a kicked state using a kick potential, as long as we have the correct numerical preparation protocol

### RPA:
random phase approximation:
well this random phase is just a "phrasing" which is not actually the exact random phase concept that is used to describe the statistical ensemble in turbulence theory, but it does not hurt to borrow it for our simulation purpose(I do not know why yet)

This method is much safer to be performed in the non-box trap, there would be some numerical issue for a box trap to hold this artificial state, or some iteration methods are required, could as well try Bessel modes and functions.


the idea is to assume some shape of the occupation number say: $n(k)\propto e^{ -(k/k_{0})^4 }$
Then we would generate wave function by:
```python
rng=np.random.default_rng(seed=42)
def nk_function(k):
	return np.exp(-(k/3)**4)
def RPA(nk_function,grid)
	amplitude_k=np.sqrt(nk_function(np.sqrt(grid.Ksq)))
	phase=2*np.pi*rng.random((grid.N,grid.N,grid.N))
	psi_k=amplitude_k*np.exp(1j*phase)
	psi=np.fft.ifft(psi_k)
	normalise_psi(psi)
```



## Unit conversion
this simulator follows one standard convention in unit conversion:
we define one dimensionless length correspond to $1 \mathrm{\mu m}$.
and the dimensionless form of GPE is:
$$
i \frac{\partial \psi}{\partial t}=\left[-\Delta+\tilde{V}+\beta|\psi|^{2}\right]\psi
$$
Where $\beta=\frac{8\pi Na}{l}$.

$$
i\hbar \frac{\partial \psi}{\partial t}=\left[ -\frac{\hbar^{2}\Delta}{2m} +V+\frac{4\pi \hbar^{2}a}{m}|\psi|^{2}\right]\psi
$$
let's set some unit length scale $l$ and divide both sides of GPE by the energy scale: $\frac{\hbar^{2}}{2ml^{2}}$
$$
i \frac{2ml^{2}}{\hbar}\frac{\partial \psi}{\partial t}=\left[ -\Delta l^{2}+\frac{2ml^{2}V}{\hbar^{2}}+ 8\pi l^{2}a|\psi|^{2}\right]\psi^{2}
$$
let's define: 
$$\frac{\hbar t}{2ml^{2}}=\tilde{t}$$
$$
\frac{2ml^{2}V}{\hbar^{2}}=\tilde{V}
$$
$$
\tilde{x}=\frac{x}{l},\tilde{y}=\frac{y}{l},\tilde{z}=\frac{z}{l}
$$
By definition we have:

$$
\begin{aligned}
&\int |\psi|^{2}\mathrm{d}\vec{r}=N\\
&\int \frac{|\psi|^{2}l^3}{N}\mathrm{d}\tilde{\vec{r}}=1

\end{aligned}

$$
Therefore if we define 
$$
\tilde{\psi}=\frac{ l^{3/2}}{\sqrt{ N }}\psi
$$
then we have normalisation:
$$
\int |\tilde{\psi}|^{2}\mathrm{d}\tilde{\vec{r}}=1
$$
$8\pi l^{2}a|\psi|^{2}=\frac{8\pi l^{2}aN}{l^3}|\tilde{\psi}|^{2}=\frac{8\pi Na}{l}|\tilde{\psi}|^{2}$
Therefore we can back substitute these definition into the dimensionless GPE:
we finally have:
$$
i \frac{\partial \psi}{\partial t}=\left[-
\Delta+\tilde{V}+\beta|\psi|^{2}\right]\psi
$$
To get a feeling of numbers, we are going to list the dimensionless form of different quantities:
1. healing length $\xi$:
$$
\xi=\frac{1}{\sqrt{ \frac{8\pi Na}{V} }}
$$
in dimensionless unit $\tilde{\xi}=\frac{\xi}{l}=\frac{1}{\sqrt{ \frac{\beta l^3}{V} }}=\sqrt{ \frac{\tilde{V}}{\beta} }$
Healing length is just the square root of dimensionless volume divided by the dimensionless interaction strength
2. $k_{\xi}$
by definition $k_{\xi}=\frac{1}{\xi}$ Therefore $\tilde{k_{\xi}}=\sqrt{ \frac{\beta}{\tilde{V}} }$

3.  mean field chemical potential $\mu$
$\mu=gn=\frac{4\pi \hbar^{2}aN}{mV}$
$\tilde{\mu}=\frac{8\pi al^{2}N}{V}=\frac{8\pi Na}{l} /\tilde{V}=\frac{\beta}{\tilde{V}}$ therefore the dimensionless chemical potential is just $\tilde{k_{\xi}}^{2}$
4. $t_{\xi}$
$t_{\xi =\frac{\hbar}{gn}}=\frac{mV}{4\pi \hbar a N}$
$\tilde{t_{\xi}}=\frac{mV}{4\pi \hbar aN} \frac{\hbar}{2ml^{2}}=\frac{V}{8\pi al^{2}N}=\frac{\tilde{V}}{\beta}$








