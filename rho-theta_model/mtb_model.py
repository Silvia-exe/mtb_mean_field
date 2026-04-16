
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mtb_class import mtb
from mpl_toolkits.axes_grid1 import make_axes_locatable

def animate_colormaps(mtb, steps=100, interval=20):
    step_x = max(1, mtb.Nx // 11)
    step_y = max(1, mtb.Ny // 6)
    slice = int(mtb.Ny//2)

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle('MTB model data', fontsize=16)

    x = np.linspace(0, mtb.Lx, mtb.Nx)
    y = np.linspace(0, mtb.Ly, mtb.Ny)
    X, Y = np.meshgrid(x, y)

    x_ticks = np.linspace(0, mtb.Lx, 5)
    y_ticks = np.linspace(0, mtb.Ly, 5)

    mtb.init_oxygen()
    mtb.build_equations()

    oxygen = mtb.c.value.reshape((mtb.Ny,mtb.Nx))
    bacteria = mtb.b.value.reshape((mtb.Ny, mtb.Nx))
    consumption = mtb.consumption_magnitude().reshape((mtb.Ny, mtb.Nx))
    theta = mtb.theta.value.reshape((mtb.Ny, mtb.Nx))
    vx, vy = mtb.velocity_vectors()
    U = vx.reshape((mtb.Ny, mtb.Nx))
    V = vy.reshape((mtb.Ny, mtb.Nx))
    mag = np.sqrt(U**2 + V**2) + 1e-12
    U_n = U/mag
    V_n = V/mag
    
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
    quiv = axes[2, 0].quiver(X[::step_y, ::step_x],Y[::step_y, ::step_x],U_n[::step_y, ::step_x],V_n[::step_y, ::step_x],color='white', scale = None, width=0.01,
    headwidth=3,
    headlength=4,
    headaxislength=3,
    pivot='middle')
    axes[2, 0].axhline(y=slice*(mtb.Ly/(mtb.Ny-1)), color='white', linestyle='--', linewidth=1)
    axes[2, 0].set_title('Polarization Magnitude')
    axes[2, 0].set_ylabel(r'y [$\mu m$]')
    axes[2, 0].set_xticks(x_ticks)
    axes[2, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax)
    cbar3.set_label(r'Velocity [$\mu m/s$]', fontsize=12)
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
    
    # Add magnetic field arrow for animation
    if mtb.B > 0:
        # Compute unit vector in direction of B
        B_mag = np.sqrt(mtb.Bx**2 + mtb.By**2)
        B_unit_x = mtb.Bx / B_mag
        B_unit_y = mtb.By / B_mag
        # Scale arrow to reasonable size
        arrow_length = 300  # pixels in data coordinates
        arrow_start_x = mtb.Lx * 0.15
        arrow_start_y = mtb.Ly * 0.85
        axes[2, 0].arrow(arrow_start_x, arrow_start_y, 
                         B_unit_x * arrow_length, B_unit_y * arrow_length,
                         head_width=150, head_length=150, fc='yellow', ec='yellow', linewidth=2.5, zorder=10)
        axes[2, 0].text(arrow_start_x - 200, arrow_start_y + 200, f'B = {mtb.B:.1f} μT', 
                       fontsize=11, color='yellow', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Animation update function
    def animate(frame):
        for _ in range(10):
            mtb.eq_c.solve(dt = mtb.dt)
            mtb.eq_b.solve(dt = mtb.dt)
            mtb.eq_theta.solve(dt = mtb.dt)

        oxygen = mtb.c.value.reshape((mtb.Ny,mtb.Nx))
        bacteria = mtb.b.value.reshape((mtb.Ny, mtb.Nx))
        consumption = mtb.consumption_magnitude().reshape((mtb.Ny, mtb.Nx))
        vx, vy = mtb.velocity_vectors()
        U = vx.reshape((mtb.Ny, mtb.Nx))
        V = vy.reshape((mtb.Ny, mtb.Nx))
        mag = np.sqrt(U**2 + V**2) + 1e-12
        U_n = U/mag
        V_n = V/mag
        
        im1.set_array(oxygen)
        im1.set_clim(vmin=np.min(oxygen), vmax=np.max(oxygen))
        im1_slice[0].set_ydata(oxygen[slice,:])
        
        im2.set_array(bacteria)
        im2.set_clim(vmin=np.min(bacteria), vmax=np.max(bacteria))
        im2_slice[0].set_ydata(bacteria[slice,:])
        
        im3.set_array(mag)
        im3.set_clim(vmin=np.min(mag), vmax=np.max(mag))
        im3_slice[0].set_ydata(mag[slice,:])
        
        im4.set_array(consumption)
        im4.set_clim(vmin=np.min(consumption), vmax=np.max(consumption))
        im4_slice[0].set_ydata(consumption[slice,:])

        quiv.set_UVC(U_n[::step_y, ::step_x], V_n[::step_y, ::step_x])

        axes[0, 0].set_title(f'Oxygen Concentration (Frame {frame})')
        axes[0, 1].set_title(f'Bacteria Concentration (Frame {frame})')
        axes[2, 0].set_title(f'Polarization Magnitude (Frame {frame})')
        axes[2, 1].set_title(f'Consumption (Frame {frame})')
        
        return [im1, im2, im3, im4, quiv, im1_slice[0], im2_slice[0], im3_slice[0], im4_slice[0]]
    
    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=interval, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def plot_colormaps(mtb):

    step_x = 3
    step_y = 3

    Lx = mtb.Lx 
    Ly = mtb.Ly
    Nx = mtb.Nx
    Ny = mtb.Ny
    B_phi = mtb.B_phi
    B = mtb.B

    x = np.linspace(0, mtb.Lx, mtb.Nx)
    y = np.linspace(0, mtb.Ly, mtb.Ny)

    X, Y = np.meshgrid(x, y)

    x_ticks = np.linspace(0, mtb.Lx, 5)
    y_ticks = np.linspace(0, mtb.Ly, 5)

    oxygen = mtb.c.value.reshape((mtb.Ny,mtb.Nx))
    bacteria = mtb.b.value.reshape((mtb.Ny, mtb.Nx))
    consumption = mtb.consumption_magnitude().reshape((mtb.Ny, mtb.Nx))
    vx, vy = mtb.active_swimming_vectors()
    U = vx.reshape((mtb.Ny, mtb.Nx))
    V = vy.reshape((mtb.Ny, mtb.Nx))
    mag = np.sqrt(U**2 + V**2) + 1e-12
    U_n = U/mag
    V_n = V/mag

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle('MTB model data', fontsize=16)

    im1 = axes[0, 0].imshow(oxygen, cmap='viridis', aspect='auto', 
                                extent=[0, Lx, 0, Ly], origin='lower')
    arrow_start_x = Lx * 0.75
    arrow_start_y = Ly * 0.85
    arrow_length = min(Lx, Ly) * 0.2
    axes[0, 0].arrow(arrow_start_x, arrow_start_y,
                     arrow_length * np.cos(np.deg2rad(B_phi)),
                     arrow_length * np.sin(np.deg2rad(B_phi)),
                     color='red', head_width=Ly * 0.04, head_length=Lx * 0.04, linewidth=2)
    axes[0, 0].set_title('Oxygen Concentration')
    axes[0, 0].set_ylabel(r'y [$\mu m$]')
    axes[0, 0].set_xticks(x_ticks)
    axes[0, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax)
    cbar1.set_label(r'Concentration [$\mu M$]', fontsize=12)
    axes[1, 0].plot(x, np.mean(oxygen, axis=0))
    axes[1, 0].set_xlabel(r'x [$\mu m$]')
    axes[1, 0].sharex(axes[0, 0])


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
    
    im3 = axes[2, 0].imshow(mag, cmap='jet', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
    axes[2, 0].quiver(X[::step_x, ::step_y],Y[::step_x, ::step_y],U_n[::step_x, ::step_y],V_n[::step_x, ::step_y],color='white', scale = None, width=0.01,
    headwidth=3,
    headlength=4,
    headaxislength=3,
    pivot='middle')
    axes[2, 0].set_title('Velocity Magnitude')
    axes[2, 0].set_ylabel(r'y [$\mu m$]')
    axes[2, 0].set_xticks(x_ticks)
    axes[2, 0].set_yticks(y_ticks)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax)
    cbar3.set_label(r'Velocity [$\mu m/s$]', fontsize=12)
    im3_slice = axes[3,0].plot(x, np.mean(mag, axis=0))
    axes[3, 0].set_xlabel(r'x [$\mu m$]')
    axes[3, 0].sharex(axes[2,0])
    
    
    im4 = axes[2, 1].imshow(consumption, cmap='Oranges', aspect = 'auto', extent=[0, mtb.Lx, 0, mtb.Ly], origin = 'lower')
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

def plot_velocities(mtb):
    vx_swim, vy_swim = mtb.active_swimming_vectors()
    vx_aero, vy_aero = mtb.aerotactic_vectors()
    vx_total, vy_total = mtb.velocity_vectors()

    U_swim = vx_swim.reshape((mtb.Ny, mtb.Nx))
    V_swim = vy_swim.reshape((mtb.Ny, mtb.Nx))
    U_aero = vx_aero.value.reshape((mtb.Ny, mtb.Nx))
    V_aero = vy_aero.value.reshape((mtb.Ny, mtb.Nx))
    U_tot = vx_total.reshape((mtb.Ny, mtb.Nx))
    V_tot = vy_total.reshape((mtb.Ny, mtb.Nx))

    oxygen = mtb.c.value.reshape((mtb.Ny, mtb.Nx))
    bacteria = mtb.b.value.reshape((mtb.Ny, mtb.Nx))

    mag_swim = np.sqrt(U_swim**2 + V_swim**2) + 1e-12
    mag_aero = np.sqrt(U_aero**2 + V_aero**2) + 1e-12
    mag_tot = np.sqrt(U_tot**2 + V_tot**2) + 1e-12

    U_swim_n = U_swim / mag_swim
    V_swim_n = V_swim / mag_swim
    U_aero_n = U_aero / mag_aero
    V_aero_n = V_aero / mag_aero
    U_tot_n = U_tot / mag_tot
    V_tot_n = V_tot / mag_tot

    x = np.linspace(0, mtb.Lx, mtb.Nx)
    y = np.linspace(0, mtb.Ly, mtb.Ny)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)

    # Oxygen concentration
    im0 = axes[0, 0].imshow(oxygen, cmap='viridis', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    axes[0, 0].set_title('Oxygen Concentration')
    axes[0, 0].set_xlabel(r'x [$\mu m$]')
    axes[0, 0].set_ylabel(r'y [$\mu m$]')
    axes[0, 0].set_xticks(np.linspace(0, mtb.Lx, 5))
    axes[0, 0].set_yticks(np.linspace(0, mtb.Ly, 5))
    fig.colorbar(im0, ax=axes[0, 0], label=r'[O$_2$] [$\mu M$]')

    # Bacteria concentration
    im0b = axes[0, 1].imshow(bacteria, cmap='plasma', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    axes[0, 1].set_title('Bacterial Concentration')
    axes[0, 1].set_xlabel(r'x [$\mu m$]')
    axes[0, 1].set_ylabel(r'y [$\mu m$]')
    axes[0, 1].set_xticks(np.linspace(0, mtb.Lx, 5))
    axes[0, 1].set_yticks(np.linspace(0, mtb.Ly, 5))
    fig.colorbar(im0b, ax=axes[0, 1], label='Concentration [O.D.]')

    # Self-propelled velocity
    im1 = axes[1, 0].imshow(mag_swim, cmap='viridis', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    axes[1, 0].quiver(X[::5, ::5], Y[::5, ::5], U_swim_n[::5, ::5], V_swim_n[::5, ::5], color='red', scale=None, width=0.008)
    axes[1, 0].set_title('Self-Propelled Velocity Magnitude')
    axes[1, 0].set_xlabel(r'x [$\mu m$]')
    axes[1, 0].set_ylabel(r'y [$\mu m$]')
    axes[1, 0].set_xticks(np.linspace(0, mtb.Lx, 5))
    axes[1, 0].set_yticks(np.linspace(0, mtb.Ly, 5))
    fig.colorbar(im1, ax=axes[1, 0], label=r'|v_swim| [$\mu m/s$]')

    # Aerotactic velocity
    im2 = axes[1, 1].imshow(mag_aero, cmap='plasma', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    axes[1, 1].quiver(X[::5, ::5], Y[::5, ::5], U_aero_n[::5, ::5], V_aero_n[::5, ::5], color='cyan', scale=None, width=0.008)
    axes[1, 1].set_title('Aerotactic Velocity Magnitude')
    axes[1, 1].set_xlabel(r'x [$\mu m$]')
    axes[1, 1].set_ylabel(r'y [$\mu m$]')
    axes[1, 1].set_xticks(np.linspace(0, mtb.Lx, 5))
    axes[1, 1].set_yticks(np.linspace(0, mtb.Ly, 5))
    fig.colorbar(im2, ax=axes[1, 1], label=r'|v_aero| [$\mu m/s$]')

    # Total velocity
    im3 = axes[2, 0].imshow(mag_tot, cmap='jet', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    axes[2, 0].quiver(X[::5, ::5], Y[::5, ::5], U_tot_n[::5, ::5], V_tot_n[::5, ::5], color='white', scale=None, width=0.008)
    axes[2, 0].set_title('Total Velocity Magnitude and Vector Field')
    axes[2, 0].set_xlabel(r'x [$\mu m$]')
    axes[2, 0].set_ylabel(r'y [$\mu m$]')
    axes[2, 0].set_xticks(np.linspace(0, mtb.Lx, 5))
    axes[2, 0].set_yticks(np.linspace(0, mtb.Ly, 5))
    fig.colorbar(im3, ax=axes[2, 0], label=r'|v_{total}| [$\mu m/s$]')

    # Hide the last subplot
    axes[2, 1].axis('off')

    plt.show()

def plot_gradients(mtb):
    c = mtb.c.value.reshape((mtb.Ny, mtb.Nx))
    grad = mtb.c.grad
    grad_x = grad[0].value.reshape((mtb.Ny, mtb.Nx))
    grad_y = grad[1].value.reshape((mtb.Ny, mtb.Nx))
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    x = np.linspace(0, mtb.Lx, mtb.Nx)
    y = np.linspace(0, mtb.Ly, mtb.Ny)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(c, cmap='viridis', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    axes[0].set_title('Oxygen Concentration')
    axes[0].set_xlabel(r'x [$\mu m$]')
    axes[0].set_ylabel(r'y [$\mu m$]')
    fig.colorbar(im1, ax=axes[0], label=r'[O$_2$] [$\mu M$]')

    im2 = axes[1].imshow(grad_magnitude, cmap='plasma', extent=[0, mtb.Lx, 0, mtb.Ly], origin='lower')
    axes[1].quiver(X[::10, ::10], Y[::10, ::10], grad_x[::10, ::10], grad_y[::10, ::10], color='white', scale=None, width=0.008)
    axes[1].set_title('Oxygen Gradient Magnitude')
    axes[1].set_xlabel(r'x [$\mu m$]')
    axes[1].set_ylabel(r'y [$\mu m$]')
    fig.colorbar(im2, ax=axes[1], label=r'|∇c| [$\mu M/\mu m$]')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bacterial simulation")
    parser.add_argument("arg_file", help="Path to parameter YAML file")
    parser.add_argument('--O2_init', help = 'Path to .h5 file for O2 gradient initialization', required=False, default = None)

    args = parser.parse_args()

    run = mtb(args)

    #animate_colormaps(run)

    plot_velocities(run)
    run.run_save()
    #plot_gradients(run)
    plot_velocities(run)
  


    
