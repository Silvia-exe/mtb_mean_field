import argparse
import matplotlib.pyplot as plt 
import h5py
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

def plot_velocities(mtb):
    step_x = max(1, mtb.Nx // 11)
    step_y = max(1, mtb.Ny // 6)

    x = np.linspace(0, mtb.Lx, mtb.Nx)
    y = np.linspace(0, mtb.Ly, mtb.Ny)

    X, Y = np.meshgrid(x, y)

    aerotaxis_x, aerotaxis_y = mtb.aerotaxis_vectors()
    advection_x, advection_y = mtb.advection_vectors()
    U_aero = aerotaxis_x.reshape((mtb.Ny, mtb.Nx))
    V_aero = aerotaxis_y.reshape((mtb.Ny, mtb.Nx))
    mag = np.sqrt(U_aero**2 + V_aero**2) + 1e-12
    U_n = U_aero/mag
    V_n = V_aero/mag

    U_adv = advection_x.reshape((mtb.Ny, mtb.Nx))
    V_adv = advection_y.reshape((mtb.Ny, mtb.Nx))
    mag_adv = np.sqrt(U_adv**2 + V_adv**2) + 1e-12
    U_adv_n = U_adv/mag_adv
    V_adv_n = V_adv/mag_adv

    fig,axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.figure(figsize=(8, 6))
    im1 = axes[0].imshow(mag, cmap='jet', aspect='auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    cbar_1 = fig.colorbar(im1, label=r'Velocity Magnitude [$\mu m^2/s$]')
    axes[0].quiver(X[::step_y, ::step_x], Y[::step_y, ::step_x], U_n[::step_y, ::step_x], V_n[::step_y, ::step_x], color='white', scale=None, width=0.01,
               headwidth=3,
               headlength=4,
               headaxislength=3,
               pivot='middle')
    axes[0].set_title('Advection Velocity Field')
    axes[0].set_xlabel(r'x [$\mu m$]')
    axes[0].set_ylabel(r'y [$\mu m$]')
    axes[0].set_xticks(np.linspace(0, mtb.Lx, 5))
    axes[0].set_yticks(np.linspace(0, mtb.Ly, 5))

    im2 = axes[1].imshow(mag_adv, cmap='jet', aspect='auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    cbar2 = plt.colorbar(im2, label=r'Velocity Magnitude [$\mu m^2/s$]')
    axes[1].quiver(X[::step_y, ::step_x], Y[::step_y, ::step_x], U_adv_n[::step_y, ::step_x], V_adv_n[::step_y, ::step_x], color='white', scale=None, width=0.01,
               headwidth=3,
               headlength=4,
               headaxislength=3,
               pivot='middle')
    axes[1].set_title('Advection Velocity Field')
    axes[1].set_xlabel(r'x [$\mu m$]')
    axes[1].set_ylabel(r'y [$\mu m$]')
    axes[1].set_xticks(np.linspace(0, mtb.Lx, 5))
    axes[1].set_yticks(np.linspace(0, mtb.Ly, 5))
    plt.tight_layout()
    plt.show()

def plot_colormaps(mtb):
    Lx = mtb.Lx
    Ly = mtb.Ly
    B_phi = mtb.B_phi

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
    aerotaxis_x, aerotaxis_y = mtb.advection_vectors()
    U = aerotaxis_x.reshape((mtb.Ny, mtb.Nx))
    V = aerotaxis_y.reshape((mtb.Ny, mtb.Nx))
    mag = np.sqrt(U**2 + V**2) + 1e-12
    U_n = U/mag
    V_n = V/mag

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle('MTB model data', fontsize=16)

    im1 = axes[0, 0].imshow(oxygen, cmap='viridis', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower', vmin=0, vmax=mtb.c0_o2)
    #axes[0, 0].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    arrow_start_x = Lx * 0.75
    arrow_start_y = Ly * 0.75
    arrow_length = min(Lx, Ly) * 0.1
    axes[0, 0].arrow(arrow_start_x, arrow_start_y,
                     arrow_length * np.cos(B_phi),
                     arrow_length * np.sin(B_phi),
                     color='red', head_width=Ly * 0.04, head_length=Lx * 0.04, linewidth=2)
    axes[0, 0].set_title('Oxygen Concentration')
    axes[0, 0].set_ylabel(r'y [$\mu m$]')
    axes[0, 0].set_xticks(x_ticks)
    axes[0, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax)
    cbar1.set_label(r'Concentration [$\mu M$]', fontsize=12)
    im1_slice = axes[1,0].plot(x, np.mean(oxygen, axis=0))
    axes[1, 0].set_xlabel(r'x [$\mu m$]')
    axes[1,0].sharex(axes[0,0])
    
    im2 = axes[0, 1].imshow(bacteria, cmap='plasma', aspect = 'auto',extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower', vmin = 0, vmax = 1.0)
    #axes[0, 1].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    axes[0, 1].set_title('Bacteria Concentration')
    axes[0, 1].set_ylabel(r'y [$\mu m$]')
    axes[0, 1].set_xticks(x_ticks)
    axes[0, 1].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax)
    cbar2.set_label('Concentration [O.D.]', fontsize=12)
    im2_slice = axes[1,1].plot(x, np.mean(bacteria, axis=0))
    axes[1, 1].set_xlabel(r'x [$\mu m$]')
    axes[1,1].sharex(axes[0,1])
    
    im3 = axes[2, 0].imshow(mag, cmap='jet', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[2, 0].quiver(X[::step_y, ::step_x],Y[::step_y, ::step_x],U_n[::step_y, ::step_x],V_n[::step_y, ::step_x],color='white', scale = None, width=0.01,
    headwidth=3,
    headlength=4,
    headaxislength=3,
    pivot='middle')
    #axes[2, 0].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    axes[2, 0].set_title('Taxis Magnitude')
    axes[2, 0].set_ylabel(r'y [$\mu m$]')
    axes[2, 0].set_xticks(x_ticks)
    axes[2, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax)
    cbar3.set_label(r'Velocity [$\mu m^2/s$]', fontsize=12)
    im3_slice = axes[3,0].plot(x, np.mean(mag, axis=0))
    axes[3, 0].set_xlabel(r'x [$\mu m$]')
    axes[3,0].sharex(axes[2,0])
    
    
    im4 = axes[2, 1].imshow(consumption, cmap='Oranges', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    #axes[2, 1].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    axes[2, 1].set_title('Oxygen Consumption by Bacteria')
    axes[2, 1].set_ylabel(r'y [$\mu m$]')
    axes[2, 1].set_xticks(x_ticks)
    axes[2, 1].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar4 = fig.colorbar(im4, cax=cax)
    cbar4.set_label(r'Consumption [$\mu M/\mu m^2$]', fontsize=12)
    im4_slice = axes[3,1].plot(x, np.mean(consumption, axis=0))
    axes[3, 1].set_xlabel(r'x [$\mu m$]')
    axes[3,1].sharex(axes[2,1])

    plt.tight_layout()
    plt.show()

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

def plot_from_h5(filename, frame=-1):
    with h5py.File(filename, 'r') as f:
        # Get number of frames
        num_frames = f['o2'].shape[0]
        print(f"Total frames available: {num_frames}")
        print(f"Plotting frame {frame} (which is frame {frame % num_frames})") 
        # Load metadata
        Lx = f.attrs['Lx']
        Ly = f.attrs['Ly']
        Nx = f.attrs['Nx']
        Ny = f.attrs['Ny']
        B_phi = f.attrs['B_phi']
        B = f.attrs['B']

        k_cons = f.attrs['k_cons']
        ca_o2 = f.attrs['ca_o2']

        # Load data at specified frame
        o2 = f['o2'][frame].reshape((Ny, Nx))
        bacteria = f['bacteria'][frame].reshape((Ny, Nx))
        velocity = f['velocity'][frame]

        print(velocity.shape)
        
        # Create meshgrid for plotting
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)
        
        # Calculate velocity vectors from velocity field
        U = velocity[0, :, :]
        V = velocity[1, :, :]
        mag = np.sqrt(U**2 + V**2) + 1e-12
        U_n = U/mag
        V_n = V /mag
        # Calculate consumption
        consumption = k_cons * bacteria * o2 / (ca_o2 + o2)
        
        # Setup figure
        step_x = max(1, Nx // 25)
        step_y = max(1, Ny // 9)
        slice_y = int(Ny // 2)
        
        x_ticks = np.linspace(0, Lx, 5)
        y_ticks = np.linspace(0, Ly, 5)
        
        fig, axes = plt.subplots(4, 2, figsize=(12, 10))
        fig.suptitle(f'MTB Simulation - Frame {frame}', fontsize=16)
        
        # --- Oxygen Concentration ---
        im1 = axes[0, 0].imshow(o2, cmap='viridis', aspect='auto', 
                                extent=[0, Lx, 0, Ly], origin='lower')
        axes[0,0 ].arrow(Lx*0.85, Ly*0.85, 500*np.cos(B_phi), 500*np.sin(B_phi), color='red', head_width=100, head_length=50, linewidth=5)
        axes[0, 0].set_title('Oxygen Concentration')
        axes[0, 0].set_ylabel(r'y [$\mu m$]')
        axes[0, 0].set_xticks(x_ticks)
        axes[0, 0].set_yticks(y_ticks)
        divider = make_axes_locatable(axes[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(im1, cax=cax)
        cbar1.set_label(r'Concentration [$\mu M$]', fontsize=12)
        axes[1, 0].plot(x, np.mean(o2, axis=0))
        axes[1, 0].set_xlabel(r'x [$\mu m$]')
        axes[1, 0].sharex(axes[0, 0])
        
        # --- Bacteria Concentration ---
        im2 = axes[0, 1].imshow(bacteria, cmap='plasma', aspect='auto',
                                extent=[0, Lx, 0, Ly], origin='lower', vmin=np.min(bacteria), vmax=np.max(bacteria))
        axes[0, 1].set_title('Bacteria Concentration')
        axes[0, 1].set_ylabel(r'y [$\mu m$]')
        axes[0, 1].set_xticks(x_ticks)
        axes[0, 1].set_yticks(y_ticks)
        divider = make_axes_locatable(axes[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(im2, cax=cax)
        cbar2.set_label('Concentration [O.D.]', fontsize=12)
        axes[1, 1].plot(x, np.mean(bacteria, axis=0))
        axes[1, 1].set_xlabel(r'x [$\mu m$]')
        axes[1, 1].set_ylim([np.min(bacteria), np.max(bacteria)])
        
        # --- Oxygen Consumption ---
        im4 = axes[2, 1].imshow(consumption, cmap='Oranges', aspect='auto', 
                                extent=[0, Lx, 0, Ly], origin='lower', vmin=np.min(consumption), vmax=np.max(consumption))
        axes[2, 1].set_title('Oxygen Consumption by Bacteria')
        axes[2, 1].set_ylabel(r'y [$\mu m$]')
        axes[2, 1].set_xticks(x_ticks)
        axes[2, 1].set_yticks(y_ticks)
        divider = make_axes_locatable(axes[2, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar4 = fig.colorbar(im4, cax=cax)
        cbar4.set_label(r'Consumption [$\mu M/\mu m^2$]', fontsize=12)
        axes[3, 1].plot(x, np.mean(consumption, axis=0))
        axes[3, 1].set_xlabel(r'x [$\mu m$]')
        axes[3, 1].set_ylim([np.min(consumption), np.max(consumption)])
        
        # --- Velocity magnitude with Vectors ---
        im3 = axes[2, 0].quiver(X[::step_y, ::step_x], Y[::step_y, ::step_x], U_n[::step_y, ::step_x], V_n[::step_y, ::step_x], scale=30, pivot='mid', color = 'white')
        axes[2,0].imshow(mag, cmap='jet', aspect='auto', extent=[0, Lx, 0, Ly], origin='lower', alpha=0.8, zorder = 0)
        axes[2, 0].set_title(r'Bacteria velocity $\theta$ [rad]')
        axes[2, 0].set_ylabel(r'y [$\mu m$]')
        axes[2, 0].set_xticks(x_ticks)
        axes[2, 0].set_yticks(y_ticks)
        divider = make_axes_locatable(axes[2, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar3 = fig.colorbar(im3, cax=cax)
        cbar3.set_label(r'Velocity [\mu²/s]', fontsize=12)
        axes[3, 0].plot(x, np.mean(mag, axis=0))
        axes[3, 0].set_xlabel(r'x [$\mu m$]')
        axes[3, 0].set_ylim(np.min(mag), np.max(mag))
        
        plt.tight_layout()
        plt.show()

def create_video_from_h5(filename, output_file='mtb_simulation.mp4'):

    with h5py.File(filename, 'r') as f:
        num_frames = f['o2'].shape[0]
        Lx = f.attrs['Lx']
        Ly = f.attrs['Ly']
        Nx = f.attrs['Nx']
        Ny = f.attrs['Ny']
        B_phi = f.attrs['B_phi']
        B = f.attrs['B']
        k_cons = f.attrs['k_cons']
        ca_o2 = f.attrs['ca_o2']
        
        # Calculate global min/max across all frames
        print("Computing global min/max values across all frames...")
        o2_min, o2_max = np.inf, -np.inf
        bacteria_min, bacteria_max = np.inf, -np.inf
        v_min, v_max = np.inf, -np.inf
        mag_min, mag_max = np.inf, -np.inf
        consumption_min, consumption_max = np.inf, -np.inf
        
        for t in range(num_frames):
            o2_data = f['o2'][t].reshape((Ny, Nx))
            bacteria_data = f['bacteria'][t].reshape((Ny, Nx))
            velocity_data = f['velocity'][t]
            mag = np.sqrt(velocity_data[0, :, :]**2 + velocity_data[1, :, :]**2) + 1e-12
            
            o2_min = min(o2_min, np.min(o2_data))
            o2_max = max(o2_max, np.max(o2_data))
            bacteria_min = min(bacteria_min, np.min(bacteria_data))
            bacteria_max = max(bacteria_max, np.max(bacteria_data))
            v_min = min(v_min, np.min(velocity_data))
            v_max = max(v_max, np.max(velocity_data))
            mag_min = min(mag_min, np.min(mag))
            mag_max = max(mag_max, np.max(mag))
            
            consumption_data = k_cons * bacteria_data * o2_data / (ca_o2 + o2_data)
            consumption_min = min(consumption_min, np.min(consumption_data))
            consumption_max = max(consumption_max, np.max(consumption_data))
        
        print(f"O2: [{o2_min:.4f}, {o2_max:.4f}]")
        print(f"Bacteria: [{bacteria_min:.4f}, {bacteria_max:.4f}]")
        print(f"Velocity: [{v_min:.4f}, {v_max:.4f}]")
        print(f"Consumption: [{consumption_min:.4f}, {consumption_max:.4f}]")

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    U = velocity_data[0, :, :]
    V = velocity_data[1, :, :]
    mag = np.sqrt(U**2 + V**2) + 1e-12
    U_n = U/mag
    V_n = V /mag

    # Setup figure
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    step_x = max(1, Nx // 25)
    step_y = max(1, Ny // 9)
    slice_y = int(Ny // 2)
    x_ticks = np.linspace(0, Lx, 5)
    y_ticks = np.linspace(0, Ly, 5)
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    
    # Initialize plots
    o2 = np.zeros((Ny, Nx))
    bacteria = np.zeros((Ny, Nx))
    consumption = np.zeros((Ny, Nx))
    
    im1 = axes[0, 0].imshow(o2, cmap='viridis', aspect='auto', extent=[0, Lx, 0, Ly], origin='lower', vmin=o2_min, vmax=o2_max) 
    axes[0, 0].arrow(Lx*0.85, Ly*0.85, 500*np.cos(B_phi), 500*np.sin(B_phi), color='red', head_width=20, head_length=30, linewidth=2)
    axes[0, 0].set_title('Oxygen Concentration')
    axes[0, 0].set_ylabel(r'y [$\mu m$]')
    axes[0, 0].set_xticks(x_ticks)
    axes[0, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0,  0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax)
    cbar1.set_label(r'Concentration [$\mu M$]', fontsize=12)
    line1, = axes[1, 0].plot(x, np.mean(o2, axis=0))
    axes[1, 0].set_xlabel(r'x [$\mu m$]')
    axes[1, 0].set_ylim([o2_min, o2_max])
    
    im2 = axes[0, 1].imshow(bacteria, cmap='plasma', aspect='auto', extent=[0, Lx, 0, Ly], origin='lower', vmin=bacteria_min, vmax=bacteria_max) 
    axes[0, 1].set_title('Bacteria Concentration')
    axes[0, 1].set_ylabel(r'y [$\mu m$]')
    axes[0, 1].set_xticks(x_ticks)
    axes[0, 1].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax)
    cbar2.set_label('Concentration [O.D.]', fontsize=12)
    line2, = axes[1, 1].plot(x, np.mean(bacteria, axis=0))
    axes[1, 1].set_xlabel(r'x [$\mu m$]')
    axes[1, 1].set_ylim([bacteria_min, bacteria_max])

    im3 = axes[2, 0].quiver(X[::step_y, ::step_x], Y[::step_y, ::step_x], U_n[::step_y, ::step_x], V_n[::step_y, ::step_x], scale=30, pivot='mid', color = 'white')
    im3_imshow = axes[2,0].imshow(mag, cmap='jet', aspect='auto', extent=[0, Lx, 0, Ly], origin='lower', alpha=0.8, zorder = 0)
    axes[2, 0].set_title(r'Bacteria velocity $\mu m²/s$ [rad]')
    axes[2, 0].set_ylabel(r'y [$\mu m$]')
    axes[2, 0].set_xticks(x_ticks)
    axes[2, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3_imshow, cax=cax)
    cbar3.set_label(r'Velocity [\mu m²/s]', fontsize=12)   
    line3, = axes[3, 0].plot(x, np.mean(mag, axis=0))
    axes[3, 0].set_xlabel(r'x [$\mu m$]')
    axes[3, 0].set_ylim([mag_min, 50])
    
    im4 = axes[2, 1].imshow(consumption, cmap='Oranges', aspect='auto', extent=[0, Lx, 0, Ly], origin='lower', vmin=consumption_min, vmax=consumption_max)
    axes[2, 1].set_title('Oxygen Consumption by Bacteria')
    axes[2, 1].set_ylabel(r'y [$\mu m$]')
    axes[2, 1].set_xticks(x_ticks)
    axes[2, 1].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar4 = fig.colorbar(im4, cax=cax)
    cbar4.set_label(r'Consumption [$\mu M/\mu m^2$]', fontsize=12)
    line4, = axes[3, 1].plot(x, np.mean(consumption, axis=0))
    axes[3, 1].set_xlabel(r'x [$\mu m$]')
    axes[3, 1].set_ylim([consumption_min, consumption_max])
    
    plt.tight_layout()
    
    def animate(frame):
        with h5py.File(filename, 'r') as f:
            o2_data = f['o2'][frame].reshape((Ny, Nx))
            bacteria_data = f['bacteria'][frame].reshape((Ny, Nx))
            velocity_data = f['velocity'][frame]

            U = velocity_data[0, :, :]
            V = velocity_data[1, :, :]
            mag = np.sqrt(U**2 + V**2) + 1e-12
            U_n = U / mag
            V_n = V / mag
        consumption_data = k_cons * bacteria_data * o2_data / (ca_o2 + o2_data)
        
        im1.set_data(o2_data)
        im2.set_data(bacteria_data)
        im3.set_UVC(U_n[::step_y, ::step_x], V_n[::step_y, ::step_x])
        im4.set_data(consumption_data)
        
        line1.set_ydata(np.mean(o2_data, axis=0))
        line2.set_ydata(np.mean(bacteria_data, axis=0))
        line3.set_ydata(np.mean(mag, axis=0))
        line4.set_ydata(np.mean(consumption_data, axis=0))
        
        fig.suptitle(f'MTB Simulation - Frame {frame}/{num_frames-1} | Magnetic Field Angle: {B_phi:.1f} rad', fontsize=16)
        return im1, im2, im3, im4, line1, line2, line3, line4
    
    print("Generating video frames...")
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=200, blit=True)
    anim.save(output_file, writer='ffmpeg', fps=5)  # Adjust fps as needed
    print(f"Video saved as {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot MTB HDF5 data or create video.')
    parser.add_argument('filename', help='HDF5 file to process')
    parser.add_argument('--frame', type=int, default=-1, help='Frame index to plot (ignored if --video)')
    parser.add_argument('--video', nargs='?', const='mtb_simulation.mp4', help='Create video and save to specified file (default: mtb_simulation.mp4)')
    args = parser.parse_args()

    if args.video:
        create_video_from_h5(args.filename, args.video)
    else:
        plot_from_h5(args.filename, frame=args.frame)