import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# Init. Conditions
nx = 200                    # Number of grid points in x
dx = 0.1                    # Spatial step size
dt = 0.01                   # Time step
n_steps = 500               # Number of time steps
driver_position = 20        # Initial position of the electron bunch (driver)
driver_velocity = 1       # Speed of the electron bunch (normalized to c)
plasma_density = 1.0        # Plasma density (normalized to c)

# Particle 
n_particles = 1000          # Number of plasma particles
positions = np.random.uniform(0, nx * dx, n_particles)  # Randomly distribution of plasma
velocities = np.zeros(n_particles)  # Plasma electrons are stationary to begin

# Tracking
tracked_particle_index = 0  # Index of particle to track
speeds = []                 # Speeds of the tracked particle over time
total_kinetic_energies = [] # Total kinetic energy of the plasma particles over time

# Driver 
driver_density = np.zeros(nx)
driver_density[int(driver_position)] = 10.0  # High density at the driver position

# Electric Field / Charge Density
electric_field = np.zeros(nx)
charge_density = np.zeros(nx)

def charge_density_calculation():
    """Update charge density based on particle positions and driver charge."""
    global charge_density
    charge_density = np.histogram(positions, bins=nx, range=(0, nx * dx))[0] / dx
    charge_density += driver_density  # Add driver density to plasma density

def electric_field_fft():
    """Solve Poisson's equation using FFT to calculate the electric field."""
    global electric_field
    k = fftfreq(nx, d=dx) * 2 * np.pi  # Wavenumbers
    k[0] = 1e-10  # Avoid division by zero for the zero frequency

    # FFT of charge density
    charge_density_fft = fft(charge_density - plasma_density)

    # Solve Poisson's equation in Fourier space: E(k) = -i * k * rho(k) / |k|^2
    FFT = -1j * charge_density_fft / k

    # Transform back to real space
    electric_field = np.real(ifft(FFT))

def advance_particles():
    """Advance particle positions and velocities using the electric field."""
    global positions, velocities
    indices = (positions / dx).astype(int)  # Find the grid index of each particle
    indices = np.clip(indices, 0, nx - 1)   # Particle bounds
    electric_force = electric_field[indices]  # Get the field at each particle's position
    velocities += electric_force * dt  # Update velocity based on field
    positions += velocities * dt  # Update position based on velocity

def update_driver_position():
    """Move the driver (electron bunch) forward."""
    global driver_position, driver_charge_density
    driver_position += driver_velocity * dt / dx
    if int(driver_position) < nx:
        driver_charge_density = np.zeros(nx)
        driver_charge_density[int(driver_position)] = 10.0
    else:
        driver_position = nx - 1  # Stop the driver at the end of the grid

def kinetic_energy():
    """Calculate total kinetic energy of all particles."""
    kinetic_energy = 0.5 * np.sum(velocities ** 2)
    total_kinetic_energies.append(kinetic_energy)

def particle_speed():
    """Track the speed of a specific particle."""
    speed = abs(velocities[tracked_particle_index])
    speeds.append(speed)

# Visualization setup
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
line1, = ax1.plot([], [], lw=2, label='Electric Field')
line2, = ax2.plot([], [], lw=2, label='Charge Density')
line3, = ax3.plot([], [], lw=2, label='Particle Speed')
line4, = ax4.plot([], [], lw=2, label='Kinetic Energy')

# Setting appropriate y-limits for each plot
ax1.set_ylim(-5, 5)                  # Electric field
ax1.set_xlim(0,20)
ax2.set_ylim(0, 100)                  # Charge density
ax2.set_xlim(0,20)
ax3.set_ylim(0, 3)                   # Particle speed
ax4.set_ylim(0, 1500)                 # Total kinetic energy

def update_plot(step):
    """Update the plot for visualization."""
    line1.set_data(np.arange(nx) * dx, electric_field)
    line2.set_data(np.arange(nx) * dx, charge_density)
    line3.set_data(range(step + 1), speeds)
    line4.set_data(range(step + 1), total_kinetic_energies)
    
    # Titles and labels
    ax1.set_title("Electric Field in Plasma Wakefield Accelerator")
    ax2.set_title("Charge Density in Plasma Wakefield Accelerator")
    ax3.set_title("Tracked Particle Speed Over Time")
    ax4.set_title("Total Kinetic Energy Over Time")
    plt.subplots_adjust(hspace=0.35)
    
    ax1.legend()
    ax2.legend()
    ax3.set_xlim(0, n_steps)
    ax4.set_xlim(0, n_steps)
    plt.pause(0.01)

# Simulation / Plotting loop
for step in range(n_steps):
    charge_density_calculation()        # Calculate new charge density from particles
    electric_field_fft()    # Solve Poisson equation using FFT to update electric field
    advance_particles()             # Move plasma particles based on electric field
    update_driver_position()        # Move the driver forward
    kinetic_energy()      # Calculate and store total kinetic energy
    particle_speed()          # Track speed of a specific particle
    update_plot(step)               # Update plot for visualization
    
plt.show()