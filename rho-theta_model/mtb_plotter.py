import argparse
import matplotlib.pyplot as plt 
import h5py
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

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
        theta = f['theta'][frame].reshape((Ny, Nx))
        
        # Create meshgrid for plotting
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)
        
        # Calculate velocity vectors from theta
        U = np.cos(theta)
        V = np.sin(theta)
        mag = np.sqrt(U**2 + V**2) + 1e-12
        U_n = U / mag
        V_n = V / mag
        
        # Calculate consumption
        consumption = k_cons * bacteria * o2 / (ca_o2 + o2)
        
        # Setup figure
        step_x = max(1, Nx // 11)
        step_y = max(1, Ny // 6)
        slice_y = int(Ny // 2)
        
        x_ticks = np.linspace(0, Lx, 5)
        y_ticks = np.linspace(0, Ly, 5)
        
        fig, axes = plt.subplots(4, 2, figsize=(12, 10))
        fig.suptitle(f'MTB Simulation - Frame {frame}', fontsize=16)
        
        # --- Oxygen Concentration ---
        im1 = axes[0, 0].imshow(o2, cmap='viridis', aspect='auto', 
                                extent=[0, Lx, 0, Ly], origin='lower')
        axes[0,0 ].arrow(0,0, 500*np.cos(np.deg2rad(B_phi)), 500*np.sin(np.deg2rad(B_phi)), color='red', head_width=100, head_length=50, linewidth=5)
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
        
        # --- Polarization Angle with Vectors ---
        theta_normalized = np.mod(theta, 2*np.pi)
        
        im3 = axes[2, 0].imshow(theta_normalized, cmap='hsv', aspect='auto', 
                                extent=[0, Lx, 0, Ly], origin='lower', vmin=0, vmax=2*np.pi)
        axes[2, 0].set_title(r'Bacteria angle $\theta$ [rad]')
        axes[2, 0].set_ylabel(r'y [$\mu m$]')
        axes[2, 0].set_xticks(x_ticks)
        axes[2, 0].set_yticks(y_ticks)
        divider = make_axes_locatable(axes[2, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar3 = fig.colorbar(im3, cax=cax)
        cbar3.set_label(r'Angle [rad]', fontsize=12)
        axes[3, 0].plot(x, np.mod(theta[0, :], 2*np.pi))
        axes[3, 0].set_xlabel(r'x [$\mu m$]')
        axes[3, 0].set_ylim([np.min(theta), np.max(theta)])
        
        plt.tight_layout()
        plt.show()

def create_video(filename, output_file='mtb_simulation.mp4'):
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
        theta_min, theta_max = np.inf, -np.inf
        consumption_min, consumption_max = np.inf, -np.inf
        
        for t in range(num_frames):
            o2_data = f['o2'][t].reshape((Ny, Nx))
            bacteria_data = f['bacteria'][t].reshape((Ny, Nx))
            theta_data = f['theta'][t].reshape((Ny, Nx))
            
            o2_min = min(o2_min, np.min(o2_data))
            o2_max = max(o2_max, np.max(o2_data))
            bacteria_min = min(bacteria_min, np.min(bacteria_data))
            bacteria_max = max(bacteria_max, np.max(bacteria_data))
            theta_min = min(theta_min, np.min(theta_data))
            theta_max = max(theta_max, np.max(theta_data))
            
            consumption_data = k_cons * bacteria_data * o2_data / (ca_o2 + o2_data)
            consumption_min = min(consumption_min, np.min(consumption_data))
            consumption_max = max(consumption_max, np.max(consumption_data))
        
        print(f"O2: [{o2_min:.4f}, {o2_max:.4f}]")
        print(f"Bacteria: [{bacteria_min:.4f}, {bacteria_max:.4f}]")
        print(f"Theta: [{theta_min:.4f}, {theta_max:.4f}]")
        print(f"Consumption: [{consumption_min:.4f}, {consumption_max:.4f}]")
    
    # Setup figure (same as plot_from_h5)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    step_x = max(1, Nx // 11)
    step_y = max(1, Ny // 6)
    slice_y = int(Ny // 2)
    x_ticks = np.linspace(0, Lx, 5)
    y_ticks = np.linspace(0, Ly, 5)
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    
    # Initialize plots
    o2 = np.zeros((Ny, Nx))
    bacteria = np.zeros((Ny, Nx))
    theta = np.zeros((Ny, Nx))
    consumption = np.zeros((Ny, Nx))
    theta_normalized = np.zeros((Ny, Nx))
    
    im1 = axes[0, 0].imshow(o2, cmap='viridis', aspect='auto', extent=[0, Lx, 0, Ly], origin='lower', vmin=o2_min, vmax=o2_max) 
    axes[0,0 ].arrow(0,0, np.cos(np.deg2rad(B_phi)), np.sin(np.deg2rad(B_phi)), color='red', head_width=20, head_length=30, linewidth=2)
    axes[0, 0].set_title('Oxygen Concentration')
    axes[0, 0].set_ylabel(r'y [$\mu m$]')
    axes[0, 0].set_xticks(x_ticks)
    axes[0, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax)
    cbar1.set_label(r'Concentration [$\mu M$]', fontsize=12)
    line1, = axes[1, 0].plot(x, o2[slice_y, :])
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
    
    im3 = axes[2, 0].imshow(theta_normalized, cmap='hsv', aspect='auto', extent=[0, Lx, 0, Ly], origin='lower', vmin=theta_min, vmax=theta_max)
    axes[2, 0].set_title(r'Bacteria angle $\theta$ [rad]')
    axes[2, 0].set_ylabel(r'y [$\mu m$]')
    axes[2, 0].set_xticks(x_ticks)
    axes[2, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax)
    cbar3.set_label(r'Angle [rad]', fontsize=12)   
    line3, = axes[3, 0].plot(x, np.mod(theta[0, :], 2*np.pi))
    axes[3, 0].set_xlabel(r'x [$\mu m$]')
    axes[3, 0].set_ylim([theta_min, theta_max])
    
    plt.tight_layout()
    
    def animate(frame):
        with h5py.File(filename, 'r') as f:
            o2_data = f['o2'][frame].reshape((Ny, Nx))
            bacteria_data = f['bacteria'][frame].reshape((Ny, Nx))
            theta_data = f['theta'][frame].reshape((Ny, Nx))
        
        U = np.cos(theta_data)
        V = np.sin(theta_data)
        mag = np.sqrt(U**2 + V**2) + 1e-12
        U_n = U / mag
        V_n = V / mag
        consumption_data = k_cons * bacteria_data * o2_data / (ca_o2 + o2_data)
        theta_normalized_data = np.mod(theta_data, 2*np.pi)
        
        im1.set_data(o2_data)
        im2.set_data(bacteria_data)
        im3.set_data(theta_normalized_data)
        im4.set_data(consumption_data)
        
        line1.set_ydata(np.mean(o2_data, axis=0))
        line2.set_ydata(np.mean(bacteria_data, axis=0))
        line3.set_ydata(np.mean(theta_normalized_data, axis=0))
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
        create_video(args.filename, args.video)
    else:
        plot_from_h5(args.filename, frame=args.frame)