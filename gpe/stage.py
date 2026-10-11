import cupy as cp
from .potentials import *
from .helper import *
from .make_propagator import *




class stage:
    def __init__ (self,simulation_range:float,x_array:cp.ndarray, k_array:cp.ndarray, N_grids:int,
                  type:str, steps:int, dt:float,
                  initial_state:dict,box_params:dict, absorber_params:dict,
                   noise_params: dict,cloud_params:dict,monitor_params:dict,save_params:dict,potential_params:dict,k_cut_params:dict ):
        if not isinstance(x_array,cp.ndarray):
            raise TypeError("x_array should be a Cupy array")
        if not isinstance(k_array,cp.ndarray):
            raise TypeError("k_array should be a Cupy array")
        if not x_array.dtype==cp.float64:
            raise TypeError("x_array should be a real array")
        if not k_array.dtype==cp.float64:
            raise TypeError("k_array should be a real array")
        if not isinstance(type,str):
            raise TypeError("type of the stage should be a string")
        if type not in {"imaginary","real"}:
            raise KeyError("type of the stage should either be 'real' or 'imaginary")
        self.absorber_params=absorber_params
        self.noise_params=noise_params
        self.simulation_range=simulation_range
        self.x_array=x_array
        self.k_array=k_array
        self.N_grids=N_grids
        self.type=type
        self.steps=steps
        self.dt=dt
        self.initial_state=initial_state
        self.psi=self._get_initial_state()
        self.box_params=box_params
        if self.type=="imaginary":
            self.staticV=self._get_box()
        elif self.type=="real":
            self.staticV=get_total_static_potential(self._get_box(),self._get_absorber())  
        if noise_params["activate"]==True:
            add_white_noise(N_grids,self.box_params["trap_depth"],self.noise_params["fraction"],self.staticV,noise_params["seed"])
        
        self.a=cloud_params["a"]
        self.N_atoms=cloud_params["N_atom"]
        self.save_params=save_params
        self.monitor_params=monitor_params
        self.potential_params=potential_params
        self.beta=get_beta(self.N_atoms,self.a)
        self.k_cut_params=k_cut_params
        
        
        if self.type=="real":
            self.propagator=make_real_propagator(self)
        else:
            self.propagator=make_imag_propagator(self)



            

        
            

            

    

        

        

            




    def _get_initial_state(self):
        if self.initial_state=="continue":
            pass
        elif self.initial_state=="load":
            pass
        elif self.initial_state=="generate":
            pass
        # return the pointer to the complex128 wave function
    def _get_box(self):
        if self.box_params["activate"]==False:
            return cp.zeros((self.N_grids,self.N_grids,self.N_grids),dtype=cp.float64)
        if self.box_params["activate"]==True:
            x_grid=self.x_array
            box_radius=self.box_params["R"]
            box_length=self.box_params["L"]
            wall_thickness=self.box_params["wall_thickness"]
            trap_depth=self.box_params["trap_depth"]
            box=box_potential(x_grid,box_radius,box_length,wall_thickness,trap_depth)
            return box
          
    def _get_absorber(self):
        if self.absorber_params["activate"]==True:
            x_grid=self.x_array
            activation_distance=self.absorber_params["activation_distance"]
            amplitude=self.absorber_params["strength"]
            simulation_range=self.simulation_range
            absorber=absorption_potential(activation_distance,amplitude,x_grid, simulation_range)
            return absorber # this is a real array
        else:
            return cp.zeros((self.N_grids,self.N_grids,self.N_grids),dtype=cp.float64)


    

    