
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mtb_class import mtb
from mpl_toolkits.axes_grid1 import make_axes_locatable

def animate_colormaps(mtb, steps=100, interval=20):
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), tight_layout = 'True')
    fig.suptitle('MTB model data', fontsize=16)

    slice = int(mtb.Ny/2)

    x = np.linspace(0, mtb.Lx, mtb.Nx)
    y = np.linspace(0, mtb.Ly, mtb.Ny)

    x_ticks = np.linspace(0, mtb.Lx, 5)
    y_ticks = np.linspace(0, mtb.Ly, 5)

    mtb.init_oxygen()
    mtb.build_equations()

    oxygen = mtb.c.value.reshape((mtb.Ny,mtb.Nx))
    bacteria = mtb.b.value.reshape((mtb.Ny, mtb.Nx))
    aerotaxis = mtb.aerotaxis_magnitude().reshape((mtb.Ny, mtb.Nx))
    consumption = mtb.consumption_magnitude().reshape((mtb.Ny, mtb.Nx))
    
    im1 = axes[0, 0].imshow(oxygen, cmap='viridis', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[0, 0].set_title('Oxygen Concentration')
    axes[0, 0].set_ylabel(r'y [$\mu m$]')
    axes[0, 0].set_xticks(x_ticks)
    axes[0, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax)
    cbar1.set_label(r'Concentration [$\mu M$]', fontsize=12)
    im1_slice = axes[1,0].plot(x, oxygen[slice,:])
    axes[1, 0].set_xlabel(r'x [$\mu m$]')
    axes[1, 0].set_xticks(x_ticks)
    axes[1, 0].set_yticks(y_ticks)
    axes[1,0].sharex(axes[0,0])
    
    im2 = axes[0, 1].imshow(bacteria, cmap='plasma', aspect = 'auto',extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[0, 1].set_title('Bacteria Concentration')
    axes[0, 1].set_ylabel(r'y [$\mu m$]')
    axes[0, 1].set_xticks(x_ticks)
    axes[0, 1].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax)
    cbar2.set_label(r'Concentration [O.D.]', fontsize=12)
    im2_slice = axes[1,1].plot(x, bacteria[slice,:])
    axes[1, 1].set_xlabel(r'x [$\mu m$]')
    axes[1, 1].set_xticks(x_ticks)
    axes[1, 1].set_yticks(y_ticks)
    axes[1,1].sharex(axes[0,1])
    
    im3 = axes[2, 0].imshow(aerotaxis, cmap='jet', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[2, 0].set_title('Taxis Magnitude')
    axes[2, 0].set_ylabel(r'y [$\mu m$]')
    axes[2, 0].set_xticks(x_ticks)
    axes[2, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax)
    cbar3.set_label(r'Velocity [$\mu m^2/s$]', fontsize=12)
    im3_slice = axes[3,0].plot(x, aerotaxis[slice,:])
    axes[3, 0].set_xlabel(r'x [$\mu m$]')
    axes[3, 0].set_xticks(x_ticks)
    axes[3, 0].set_yticks(y_ticks)
    axes[3,0].sharex(axes[2,0])
    
    
    im4 = axes[2, 1].imshow(consumption, cmap='Oranges', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[2, 1].set_title('Oxygen Consumption by Bacteria')
    axes[2, 1].set_ylabel(r'y [$\mu m$]')
    axes[2, 1].set_xticks(x_ticks)
    axes[2, 1].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar4 = fig.colorbar(im4, cax=cax)
    cbar4.set_label(r'Consumption [$\mu M/\mu m^2$]', fontsize=12)
    im4_slice = axes[3,1].plot(x, consumption[slice,:])
    axes[3, 1].set_xlabel(r'x [$\mu m$]')
    axes[3, 1].set_xticks(x_ticks)
    axes[3, 1].set_yticks(y_ticks)
    axes[3,1].sharex(axes[2,1])
    
    # Animation update function
    def animate(frame):
        for _ in range(100):
            mtb.eq_c.solve(dt = mtb.dt)
            mtb.eq_b.solve(dt = mtb.dt)

        oxygen = mtb.c.value.reshape((mtb.Ny,mtb.Nx))
        bacteria = mtb.b.value.reshape((mtb.Ny, mtb.Nx))
        aerotaxis = mtb.aerotaxis_magnitude().reshape((mtb.Ny, mtb.Nx))
        consumption = mtb.consumption_magnitude().reshape((mtb.Ny, mtb.Nx))
        
        im1.set_array(oxygen)
        im1_slice[0].set_ydata(oxygen[slice,:])
        im2.set_array(bacteria)
        im2_slice[0].set_ydata(bacteria[slice,:])
        im3.set_array(aerotaxis)
        im3_slice[0].set_ydata(aerotaxis[slice,:])
        im4.set_array(consumption)
        im4_slice[0].set_ydata(consumption[slice,:])

        axes[0, 0].set_title(f'Oxygen Concentration (Frame {frame})')
        axes[0, 1].set_title(f'Bacteria Concentration (Frame {frame})')
        axes[2, 0].set_title(f'Taxis Magnitude (Frame {frame})')
        axes[2, 1].set_title(f'Consumption (Frame {frame})')
        
        return [im1, im2, im3, im4, im1_slice, im2_slice, im3_slice, im4_slice]
    
    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=interval, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def plot_colormaps(mtb):
    step_x = max(1, mtb.Nx // 11)
    step_y = max(1, mtb.Ny // 6)
    slice = int(mtb.Ny//2)

    x = np.linspace(0, mtb.Lx, mtb.Nx)
    y = np.linspace(0, mtb.Ly, mtb.Ny)

    X, Y = np.meshgrid(x, y)

    x_ticks = np.linspace(0, mtb.Lx, 5)
    y_ticks = np.linspace(0, mtb.Ly, 5)

    oxygen = mtb.c.value.reshape((mtb.Ny,mtb.Nx))
    bacteria = mtb.b.value.reshape((mtb.Ny, mtb.Nx))
    consumption = mtb.consumption_magnitude().reshape((mtb.Ny, mtb.Nx))
    aerotaxis_x, aerotaxis_y = mtb.aerotaxis_vectors()
    U = aerotaxis_x.reshape((mtb.Ny, mtb.Nx))
    V = aerotaxis_y.reshape((mtb.Ny, mtb.Nx))
    mag = np.sqrt(U**2 + V**2) + 1e-12
    U_n = U/mag
    V_n = V/mag

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle('MTB model data', fontsize=16)

    im1 = axes[0, 0].imshow(oxygen, cmap='viridis', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[0, 0].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    axes[0, 0].set_title('Oxygen Concentration')
    axes[0, 0].set_ylabel(r'y [$\mu m$]')
    axes[0, 0].set_xticks(x_ticks)
    axes[0, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax)
    cbar1.set_label(r'Concentration [$\mu M$]', fontsize=12)
    im1_slice = axes[1,0].plot(x, oxygen[slice,:])
    axes[1, 0].set_xlabel(r'x [$\mu m$]')
    axes[1,0].sharex(axes[0,0])
    
    im2 = axes[0, 1].imshow(bacteria, cmap='plasma', aspect = 'auto',extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[0, 1].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    axes[0, 1].set_title('Bacteria Concentration')
    axes[0, 1].set_ylabel(r'y [$\mu m$]')
    axes[0, 1].set_xticks(x_ticks)
    axes[0, 1].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax)
    cbar2.set_label('Concentration [O.D.]', fontsize=12)
    im2_slice = axes[1,1].plot(x, bacteria[slice,:])
    axes[1, 1].set_xlabel(r'x [$\mu m$]')
    axes[1,1].sharex(axes[0,1])
    
    im3 = axes[2, 0].imshow(mag, cmap='jet', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[2, 0].quiver(X[::step_y, ::step_x],Y[::step_y, ::step_x],U_n[::step_y, ::step_x],V_n[::step_y, ::step_x],color='white', scale = None, width=0.01,
    headwidth=3,
    headlength=4,
    headaxislength=3,
    pivot='middle')
    axes[2, 0].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    axes[2, 0].set_title('Taxis Magnitude')
    axes[2, 0].set_ylabel(r'y [$\mu m$]')
    axes[2, 0].set_xticks(x_ticks)
    axes[2, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax)
    cbar3.set_label(r'Velocity [$\mu m^2/s$]', fontsize=12)
    im3_slice = axes[3,0].plot(x, mag[slice,:])
    axes[3, 0].set_xlabel(r'x [$\mu m$]')
    axes[3,0].sharex(axes[2,0])
    
    
    im4 = axes[2, 1].imshow(consumption, cmap='Oranges', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[2, 1].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    axes[2, 1].set_title('Oxygen Consumption by Bacteria')
    axes[2, 1].set_ylabel(r'y [$\mu m$]')
    axes[2, 1].set_xticks(x_ticks)
    axes[2, 1].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar4 = fig.colorbar(im4, cax=cax)
    cbar4.set_label(r'Consumption [$\mu M/\mu m^2$]', fontsize=12)
    im4_slice = axes[3,1].plot(x, consumption[slice,:])
    axes[3, 1].set_xlabel(r'x [$\mu m$]')
    axes[3,1].sharex(axes[2,1])

    plt.tight_layout()
    plt.show()
     
def plot_lines(mtb):
    slice = int(mtb.Ny/2)
    oxygen = mtb.c
    bacteria = mtb.b
    aerotaxis = mtb.aerotaxis()
    consumption = mtb.b*mtb.consumption()

    #plt.plot(oxygen[slice,:], label = 'Oxygen')
    plt.plot(consumption[slice,:], label = 'Consumption')
    plt.plot(bacteria[slice,:], label = 'Bacteria')
    plt.plot(bacteria[slice,:], label = 'Aerotaxis')

    plt.legend()
    plt.show()
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bacterial simulation")
    parser.add_argument("arg_file", help="Path to parameter YAML file")
    parser.add_argument('--O2_init', help = 'Path to .h5 file for O2 gradient initialization', required=False, default = None)

    args = parser.parse_args()

    run = mtb(args)

    #run.init_oxygen()
    #run.inoculum_center()

    #animate_colormaps(run)

    plot_colormaps(run)
    run.run()
    plot_colormaps(run)
  


    
