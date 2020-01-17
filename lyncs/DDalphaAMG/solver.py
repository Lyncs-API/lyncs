from lyncs.DDalphaAMG import get_lib

class solver:
    "The DDalphaAMG solver class"
    def __init__(self):
        from cppyy import nullptr
        self.init_params = get_lib().DDalphaAMG_init()
        
        #self.init_params.comm_cart
        self.init_params.Cart_rank = nullptr
        self.init_params.Cart_coords = nullptr

        for i in range(4):
            self.init_params.global_lattice[i] = 1
            self.init_params.procs[i] = 1

            self.init_params.block_lattice[i] = 1
            
            self.init_params.theta[i] = 1

        self.init_params.bc = 0

        self.init_params.number_of_levels = 2
        self.init_params.number_openmp_threads = 1

        self.init_params.kappa = 0
        self.init_params.mu = 0
        self.init_params.csw = 0

        #self.init_params.init_file = nullptr
        #self.init_params.rnd_seeds = nullptr
        
        

        
        


