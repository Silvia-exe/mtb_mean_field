import yaml
from tqdm import tqdm
import numpy as np
import h5py
from fipy import *

class mtb:
    def load_param(self, file = None):
        if file is None:
            print('No parameter .yaml file provided')
            return False
   
        with open(file) as f:
            config = yaml.safe_load(f)
        
        params={
            'save_every': int(config['metadata']['save_every']),
            'o2_init_time': int(config['metadata']['o2_init_time']),
            'o2_left': bool(config['metadata']['o2_left']),
            'o2_right': bool(config['metadata']['o2_right']),
            'o2_top': bool(config['metadata']['o2_top']),
            'o2_bottom': bool(config['metadata']['o2_bottom']),

            'Lx': float(config['grid']['Lx']),
            'Ly': float(config['grid']['Ly']),
            'Nx': int(config['grid']['Nx']),
            'Ny': int(config['grid']['Ny']),
            'T': float(config['grid']['T']),
            'dt': float(config['grid']['dt']),

            'D_o2': float(config['diff']['D_o2']),
            'D_bact': float(config['diff']['D_bact']),
        
            'c0_o2': float(config['o2']['c0_o2']),
            'ca_o2': float(config['o2']['ca_o2']),
            'copt_o2': float(config['o2']['copt_o2']),
            'cmin_o2': float(config['o2']['cmin_o2']),

            'p': float(config['bact']['p']),
            'k_cons': float(config['bact']['k_cons']),
            'K': float(config['bact']['K']),
            'chi': float(config['bact']['chi']),
            'b0': float(config['bact']['b0']),
            'mu': float(config['bact']['mu']),

            'B': float(config['magnetism']['B']),
            'B_phi': float(config['magnetism']['B_phi']),
        }

        return params
    
    def __init__(self, args):

        self.O2_init = args.O2_init

        params = self.load_param(args.arg_file)

        self.eps = 1e-12
        self.save_every = params['save_every']
        self.o2_init_time = params['o2_init_time']
        self.o2_left = params['o2_left']
        self.o2_right = params['o2_right']
        self.o2_top = params['o2_top']
        self.o2_bottom = params['o2_bottom']
        
        self.Lx = params['Lx']
        self.Ly = params['Ly']
        self.Nx = params['Nx']
        self.Ny = params['Ny']

        self.T = params['T']
        self.dt = params['dt']

        self.D_o2 = params['D_o2']
        self.D_bact = params['D_bact']
        self.c0_o2 = params['c0_o2']
        self.ca_o2 = params['ca_o2']
        self.copt_o2 = params['copt_o2']
        self.cmin_o2 = params['cmin_o2']
        self.p = params['p']
        self.k_cons = params['k_cons']
        self.K = params['K']
        self.chi = params['chi']
        self.b0 = params['b0']
        self.mu = params['mu']

        self.B = params['B']
        self.B_phi = np.deg2rad(params['B_phi'])

        self.dx, self.dy = self.Lx/(self.Nx-1), self.Ly/(self.Ny-1)
        self.t = np.arange(0,self.T,self.dt)
        self.mesh = Grid2D(nx= self.Nx, ny = self.Ny, dx = self.dx, dy = self.dy)
        self.mesh_bacteria = PeriodicGrid2DTopBottom(nx= self.Nx, ny = self.Ny, dx = self.dx, dy = self.dy)

        self.b = CellVariable(mesh = self.mesh_bacteria, name = 'bacteria', value = self.b0)
        self.c = CellVariable(mesh = self.mesh, name = 'oxygen', value = self.cmin_o2)
        self.c = self.o2_constraint()
        self.v = FaceVariable(mesh = self.mesh, rank = 1, name = 'velocity')

        self.Bx = self.B*np.cos(self.B_phi) #For now, without magnitude, since magnetic field should not affect advection magnitude...?
        self.By = self.B*np.sin(self.B_phi)
    
    def o2_constraint(self):
        if self.o2_left == True:
            self.c.constrain(self.c0_o2, where = self.mesh.facesLeft)
        elif self.o2_right == True:
            self.c.constrain(self.c0_o2, where = self.mesh.facesRight)
        elif self.o2_top == True:
            self.c.constrain(self.c0_o2, where = self.mesh.facesTop)
        elif self.o2_bottom == True:
            self.c.constrain(self.c0_o2, where = self.mesh.facesBottom)
        else:
            print('Oxygen has no input boundary condition. Check parameters.yaml')
        
        return self.c

    def init_h5(self, save_every = 20):
        filename = 'mtb_simulation_Bphi' + str(np.rad2deg(self.B_phi)) + 'Nx' + str(self.Nx) + '_Ny' + str(self.Ny) + '_T' + str(self.T) + '_dt' + str(self.dt) + '.h5'
        self.h5file = h5py.File(filename, 'w')

        Nt_save = int(len(self.t) // save_every + 1)

        self.dset_c = self.h5file.create_dataset('o2', shape = (Nt_save, self.Ny, self.Nx), dtype = 'f4', chunks = (1, self.Ny, self.Nx), compression = 'gzip')
        self.dset_b = self.h5file.create_dataset('bacteria', shape = (Nt_save, self.Ny, self.Nx), dtype = 'f4', chunks = (1, self.Ny, self.Nx), compression = 'gzip')
        self.dset_v = self.h5file.create_dataset('velocity', shape = (Nt_save, 2, self.Ny, self.Nx), dtype = 'f4', chunks = (1, 2, self.Ny, self.Nx), compression = 'gzip')
        self.dset_t = self.h5file.create_dataset('t', shape = (Nt_save,), dtype = 'f4')

        self.h5file.attrs['Lx'] = self.Lx
        self.h5file.attrs['Ly'] = self.Ly
        self.h5file.attrs['Nx'] = self.Nx
        self.h5file.attrs['Ny'] = self.Ny
        self.h5file.attrs['T'] = self.T
        self.h5file.attrs['dt'] = self.dt

        self.h5file.attrs['D_o2'] = self.D_o2
        self.h5file.attrs['D_bact'] = self.D_bact
        self.h5file.attrs['c0_o2'] = self.c0_o2
        self.h5file.attrs['ca_o2'] = self.ca_o2
        self.h5file.attrs['copt_o2'] = self.copt_o2
        self.h5file.attrs['cmin_o2'] = self.cmin_o2
        self.h5file.attrs['copt_o2'] = self.copt_o2
        self.h5file.attrs['p'] = self.p
        self.h5file.attrs['k_cons'] = self.k_cons
        self.h5file.attrs['K'] = self.K
        self.h5file.attrs['chi'] = self.chi
        self.h5file.attrs['b0'] = self.b0

        self.h5file.attrs['B'] = self.B
        self.h5file.attrs['B_phi'] = self.B_phi

        return self.h5file

    def save_state_h5(self, i, t):
        self.dset_c[i] = self.c.value.reshape ((self.Ny, self.Nx))
        self.dset_b[i] = self.b.value.reshape((self.Ny, self.Nx))
        vel = self.advection_vectors()
        self.dset_v[i] = vel.reshape((2, self.Ny, self.Nx))
        self.dset_t[i] = t

    def advection_vectors(self):
        o2_diff = (self.c - self.copt_o2) / (self.c + self.copt_o2 + self.eps)
        v0 = -1/2 * self.chi * o2_diff * self.c.grad
        v_dot_B = v0[0]*self.Bx + v0[1]*self.By
        vx = v_dot_B * self.Bx
        vy = v_dot_B * self.By
        
        return np.array([vx.value, vy.value])
    
    def aerotaxis_vectors(self):
        o2_diff = (self.c - self.copt_o2) / (self.c + self.copt_o2 + self.eps)
        v = -1/2 * self.chi * o2_diff * self.c.grad

        flux_x = v[0].value
        flux_y = v[1].value

        return np.array([flux_x, flux_y])
    
    def consumption_magnitude(self):
        consumption = (self.consumption()*self.b).value
        consumption = consumption.reshape((self.Ny, self.Nx))
        return consumption
    
    def growth_magnitude(self):
        growth = self.p * self.b * (1-(self.b/self.K))
        growth = growth.reshape((self.Ny, self.Nx))
        return growth

    def init_oxygen(self, o2_file = None):
        if o2_file is None:
            eq = TransientTerm(var = self.c) == DiffusionTerm(coeff = self.D_o2, var = self.c)
            for _ in tqdm(range(self.o2_init_time), desc="Initializing O2 gradient from scratch..."):
                eq.solve(dt = 4e-1)
            filename = 'mtb_oxygen_init_Nx'+str(self.Nx)+'_Ny'+str(self.Ny)+'_init_time'+str(self.o2_init_time)+'.h5'
            with h5py.File(filename, 'w') as f:
                print('Saving initialized oxygen gradient as ' +filename + '...')
                f.create_dataset('oxygen', data=self.c.value.reshape((self.Ny, self.Nx)), dtype='float64')
                f.close()
            return self.c
        else:
            print('Initializing from .h5 file...')
            with h5py.File(o2_file, 'r') as f:
                o2 = f['oxygen'][:]
                f.close()
            self.c.value[:] = o2.flatten()
            return self.c

    def inoculum_center(self, center = None, r = 400.0, c_bact = None):
        self.b.value[:] = 0.0
        if c_bact is None:
            c_bact = self.b0
        x, y = [np.array(arr) for arr in self.b.mesh.cellCenters]
        if center is None:
            center_x, center_y = self.Lx / 2, self.Ly / 2
        if r is None:
            r = min(center_x, center_y, self.Nx - center_x, self.Ny - center_y)
        
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
 
        mask = dist_from_center <= r
        
        self.b.value[mask] = c_bact

        return self.b

    def consumption(self):
        k_prime = self.k_cons*(self.c/(self.c+self.ca_o2+self.eps))
        return k_prime
    
    def update_velocity(self):
        o2_diff_face = (self.c.arithmeticFaceValue - self.copt_o2)/(self.c.arithmeticFaceValue + self.copt_o2 + self.eps)
        v0 = - 0.5 * self.chi * o2_diff_face * self.c.faceGrad
        v_dot_B = v0[0]*self.Bx + v0[1]*self.By
        self.v[0] = v_dot_B * self.Bx
        self.v[1] = v_dot_B * self.By
        return self.v

    def build_equations(self):

        self.eq_c = TransientTerm(var=self.c) == DiffusionTerm(coeff = self.D_o2, var = self.c) - self.consumption()*self.b
      
        self.eq_b = TransientTerm( var = self.b) == DiffusionTerm(coeff = self.D_bact, var = self.b) \
                                                - UpwindConvectionTerm(coeff = self.v, var = self.b) \
                                                + self.p * self.b * (1-(self.b/self.K))
    def run(self):

        self.c = self.init_oxygen(self.O2_init) 
        self.b = self.inoculum_center()
        self.build_equations()
        
        for _ in tqdm(self.t, desc="Running..."):
            self.update_velocity()
            self.eq_c.solve(dt = self.dt)
            self.eq_b.solve(dt = self.dt)

    def run_save(self):

        self.h5file = self.init_h5()

        self.c = self.init_oxygen(self.O2_init) 
        self.b = self.inoculum_center()
        self.build_equations()

        save_idx = 0

        for i, t in tqdm(enumerate(self.t), total = len(self.t), desc = 'Running and saving...'):
            self.update_velocity()
            self.eq_c.solve(dt = self.dt)
            self.eq_b.solve(dt = self.dt)
            
            if i% self.save_every == 0:
                self.save_state_h5(save_idx, t)
                save_idx += 1

        self.h5file.close()
