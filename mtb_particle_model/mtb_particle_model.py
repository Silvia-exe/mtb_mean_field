import numpy as np
import matplotlib.pyplot as plt
from fipy import *
import yaml
import argparse
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


class bacteria:
    def __init__(self):
        self.r = np.array([0,0])
        self.theta= 0
    
    def position(self):
        return self.r
    
    def orientation(self):
        return self.theta
    
    def update_position(self, new_position):
        self.r = new_position
    
    def update_orientation(new_theta):
        self.theta = new_theta

class model:
    def load_parameters(self, file):
        if file is None:
            print('No parameter .yaml file provided')
            return False
        
        required_params = ['D_o2', 'D_bact', 'c0_o2', 'ca_o2', 'copt_o2', 'cmin_o2','p','k_cons','K', 'chi']
        with open(file) as f:
            config = yaml.safe_load(f)
        
        params = { 
            'Lx': float(config['grid']['Lx']),
            'Ly': float(config['grid']['Ly']),
            'Nx': int(config['grid']['Nx']),
            'Ny': int(config['grid']['Ny']),
            'T': float(config['grid']['T']),
            'dt': float(config['grid']['dt']),
            'n_bact': int(config['grid']['n_bact']),

            'k_B': float(config['noise']['k_B']),
            'TK': float(config['noise']['TK']),
            'gamma_t': float(config['noise']['gamma_t']),
            'gamma_r': float(config['noise']['gamma_r']),

            'k_cons': float(config['bacteria']['k_cons']),
            'chi': float(config['bacteria']['chi']),
            'mu': float(config['bacteria']['mu']),
            'v0': float(config['bacteria']['v0']),

            'B': float(config['magnetic_field']['B']),
            'B_phi': float(config['magnetic_field']['B_phi']),

            'D_o2': float(config['o2']['D_o2']),
            'c0_o2': float(config['o2']['c0_o2']),
            'ca_o2': float(config['o2']['ca_o2']),
            'copt_o2': float(config['o2']['copt_o2']),
            'cmin_o2': float(config['o2']['cmin_o2'])
         }
        
        return params
    
    def __init__(self, args):
        params = self.load_parameters(args.params)

        self.Lx = params['Lx']
        self.Ly = params['Ly']
        self.Nx = params['Nx']
        self.Ny = params['Ny']
        self.T = params['T']
        self.dt = params['dt']
        self.n_bact = params['n_bact']
        self.k_B = params['k_B']
        self.TK = params['TK']
        self.gamma_t = params['gamma_t']
        self.gamma_r = params['gamma_r']
        self.k_cons = params['k_cons']
        self.chi = params['chi']
        self.mu = params['mu']
        self.v0 = params['v0']
        self.B = params['B']
        self.B_phi = np.deg2rad(params['B_phi'])
        self.D_o2 = params['D_o2']
        self.c0_o2 = params['c0_o2']
        self.ca_o2 = params['ca_o2']
        self.copt_o2 = params['copt_o2']
        self.cmin_o2 = params['cmin_o2']

        self.dx, self.dy = self.Lx/(self.Nx-1), self.Ly/(self.Ny-1)
        self.t = np.linspace(0,self.T,int(self.T/self.dt))
        self.mesh = Grid2D(nx= self.Nx, ny = self.Ny, dx = self.dx, dy = self.dy)

        self.c = CellVariable(mesh = self.mesh, name = 'oxygen', value = self.cmin_o2)
       
        self.c.constrain(self.c0_o2, where = self.mesh.facesLeft)
        self.init_bacteria()


    def init_bacteria(self):
        n_bact = self.n_bact
        self.bacteria = []
        for _ in range(n_bact):
            b = bacteria()
            b.r = np.random.rand(2)*[self.Lx-0.1, self.Ly-0.1]
            b.theta = np.random.uniform(0, 2*np.pi)
            self.bacteria.append(b)

    def init_o2(self):
        eq = TransientTerm(var = self.c) == DiffusionTerm(coeff = self.D_o2, var = self.c)
        for _ in tqdm(range(200), desc="Initializing O2 gradient from scratch..."):
            eq.solve(dt = 4e-1)

    def aerotaxis(self):
        x = [b.r[0] for b in self.bacteria]
        y = [b.r[1] for b in self.bacteria]
        theta = [b.theta for b in self.bacteria]
        
        c_grad = self.c.grad

        grad = c_grad[0]((x,y)) * np.cos(theta) +  c_grad[1]((x,y)) * np.sin(theta)

        signal = -1/2 * self.chi * grad * (self.c((x,y)) - self.ca_o2)/(self.c((x,y)) + self.ca_o2)
        print(signal)

        return signal

    def update_alignment(self):
        thetas = np.array([b.theta for b in self.bacteria])
        
        noise = np.sqrt(2 * self.k_B * self.TK * self.gamma_r) * np.sqrt(self.dt) * np.random.normal(0,1 ,len(self.bacteria))

        torque = self.mu * self.B * np.sin( self.B_phi-thetas)
    
        dtheta = 1/self.gamma_r * (torque) * self.dt + noise

        new_thetas = thetas + dtheta

        # assign back
        for i, b in enumerate(self.bacteria):
            b.theta = new_thetas[i]
            b.theta = (b.theta + np.pi) % (2 * np.pi) - np.pi
        
        return new_thetas
    
    def update_positions(self):
        noise = np.sqrt(2 * self.k_B * self.TK * self.gamma_t) * np.sqrt(self.dt) * np.random.normal(0,1,(len(self.bacteria),2))
        signal = self.aerotaxis()
        for i, b in enumerate(self.bacteria):
            new_position = b.r[0] + signal[i] * self.dt + noise[0]
            
            b.update_position(new_position)

    def consumption(self):
        return self.k_cons * self.b * (self.c/(self.c + self.ca_o2)) * (1 - np.exp(-self.c/self.copt_o2))
    
    def update_bacteria(self):
        self.update_alignment()
        self.update_positions()
    
    def build_o2_dynamics(self):
        self.eq_c = TransientTerm(var=self.c)== DiffusionTerm(coeff=self.D_o2, var=self.c) #- self.consumption() * self.b)

    def run(self):
        self.build_o2_dynamics()
        self.init_o2()

        for t in self.t:
            self.update_bacteria()
            #self.eq_c.solve(dt=self.dt)


def plot_bacteria(mtb):
    bacteria = mtb.bacteria
    B = mtb.B
    B_phi = mtb.B_phi
    Lx = mtb.Lx
    Ly = mtb.Ly

    x = np.array([b.r[0] for b in bacteria])
    y = np.array([b.r[1] for b in bacteria])
    theta = np.array([b.theta for b in bacteria])

    u = np.cos(theta)
    v = np.sin(theta)

    plt.figure(figsize=(6, 6))

    c = mtb.c  # FiPy field
    nx, ny = mtb.Nx, mtb.Ny

    C = c.value.reshape((ny, nx))

    plt.imshow(C,origin='lower',extent=[0, Lx, 0, Ly],cmap='viridis',alpha=0.7)

    q = plt.quiver(
        x, y, u, v,theta,cmap='hsv',angles='xy',scale_units='xy',scale=1.0,width=0.04)

    cbar = plt.colorbar(q)
    cbar.set_label('Angle (rad)')


    x0, y0 = Lx * 0.8, Ly * 0.2
    Bx = np.cos(B_phi)
    By = np.sin(B_phi)

    plt.quiver(x0, y0, Bx, By,color='red',scale_units='xy',scale=1.0,width=0.01)

    plt.text(x0, y0 - 1.0,f"|B| = {B:.2f}",color='red',fontsize=12,ha='center')

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.gca().set_aspect('equal')

    plt.xlabel('x (µm)')
    plt.ylabel('y (µm)')
    plt.title('Bacteria positions and orientations')

    plt.show()

def animate_bacteria(mtb, steps, interval=50):

    fig, ax = plt.subplots(figsize=(6, 6))

    bacteria = mtb.bacteria
    x = np.array([b.r[0] for b in bacteria])
    y = np.array([b.r[1] for b in bacteria])
    theta = np.array([b.theta for b in bacteria])
    B = mtb.B
    B_phi = mtb.B_phi
    Lx = mtb.Lx
    Ly = mtb.Ly

    x0, y0 = Lx * 0.8, Ly * 0.2
    Bx = np.cos(B_phi)
    By = np.sin(B_phi)
    ax.quiver(x0, y0, Bx, By,color='black',scale_units='xy',scale=1.0, width=0.01)
    ax.text(x0, y0 - 1.0, f"|B| = {B:.2f}", color='red', fontsize=12, ha='center')

    q = ax.quiver(x, y, np.cos(theta), np.sin(theta), theta, cmap='hsv', angles='xy', scale_units='xy', scale=1.0, width=0.04)

    mtb.init_o2()

    def update(frame):
        
        mtb.update_bacteria()


        x = np.array([b.r[0] for b in bacteria])
        y = np.array([b.r[1] for b in bacteria])
        theta = np.array([b.theta for b in bacteria])

        u = np.cos(theta)
        v = np.sin(theta)

        q.set_offsets(np.c_[x, y])
        q.set_UVC(u, v, theta)

        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect('equal')
        ax.set_title('Bacteria simulation')

        return q,

    ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit = False)

    plt.show()

    return ani

def main():
    
    parser = argparse.ArgumentParser(description='Run MTB particle model simulation')
    parser.add_argument('--params', type=str, default='mtb_particle_parameters.yaml', help='Path to parameters .yaml file')
    args = parser.parse_args()

    run = model(args)
    #plot_bacteria(run)
    #run.run()
    #plot_bacteria(run)
    ani = animate_bacteria(run, steps=10, interval=1)

if __name__ == "__main__":
    main()