import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from pathlib import Path
import pandas as pd

# Get the directory of the current script (project/animations)
script_dir = Path(__file__).resolve().parent
# Define the path relative to the script directory (go up one level to project/ then to data_out)
input_file = script_dir.parent / 'data_out' / 'simulation_data_file_pw_gpu.csv'

# Extract meta info from CSV comments
sim_years = ""
integration_method = ""
initializer = ""
execution_mode = ""
force_method = ""

with open(input_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            if 'years:' in line:
                sim_years = float(line.split(':')[-1].strip())
            elif 'integration method:' in line:
                integration_method = line.split(':')[-1].strip()
                # Simplify method name
                if integration_method == "Velocity Verlet":
                    integration_method = "verlet"
            elif 'Initializer:' in line:
                initializer = line.split(':')[-1].strip().replace('From', '').strip()
            elif 'force method:' in line:
                force_method = line.split(':')[-1].strip()
                # Simplify force method
                if force_method == "Pairwise":
                    force_method = "pw"
            elif 'execution mode:' in line:
                execution_mode = line.split(':')[-1].strip().lower()  # Convert to lowercase

# Load data using pandas - map the columns correctly
print("Loading data...")
data = pd.read_csv(input_file, comment='#', low_memory=False)
# Check if the data is loaded correctly
if data.empty:
    raise ValueError("No data loaded. Check the input file.")

# Get column names from the loaded data
print(f"Available columns: {data.columns.tolist()}")

# Extract columns: particle_id, time (years), x, y, mass
particle_ids = data['particle_id'].values
time_years = data['time'].values
x = data['x'].values if 'x' in data.columns else data['r0'].values
y = data['y'].values if 'y' in data.columns else data['r1'].values
mass = data['mass'].values

# Create a colormap for particles
unique_particles = np.unique(particle_ids)
num_particles = len(unique_particles)
colors = plt.cm.tab10(np.linspace(0, 1, num_particles))  # tab10 gives 10 distinct colors
color_dict = dict(zip(unique_particles, colors))

# Calculate size scaling for masses (log scale)
mass_min = np.min(mass[mass > 0])  # Avoid log(0)
mass_max = np.max(mass)
print(f"Mass range: {mass_min:.2e} to {mass_max:.2e}")

def get_size_from_mass(m):
    """Calculate particle size based on mass, working for any scale of masses.
    
    Uses logarithmic scaling with safeguards against numerical issues.
    """
    # Size bounds
    min_size = 3
    max_size = 100
    
    # Handle zero or negative masses
    if m <= 0:
        return min_size
    
    # Calculate logarithmic values safely
    try:
        log_mass = np.log10(m) if m > 0 else -20
        log_mass_min = np.log10(mass_min) if mass_min > 0 else -20
        log_mass_max = np.log10(mass_max)
        
        # Check for valid range
        log_range = log_mass_max - log_mass_min
        if log_range < 1e-10:  # Almost no range
            return (min_size + max_size) / 2  # Return middle size
        
        # Calculate normalized position in the log scale
        normalized_position = (log_mass - log_mass_min) / log_range
        
        # Clamp between 0 and 1 to handle outliers
        normalized_position = max(0, min(1, normalized_position))
        
        # Linear interpolation between min_size and max_size
        return min_size + normalized_position * (max_size - min_size)
        
    except (ValueError, RuntimeWarning):
        # Fallback for any calculation errors
        return min_size

# Function to convert years to (years, days)
def convert_time(years_val):
    y = int(years_val)
    d = (years_val - y) * 365.0
    return y, d

# Get unique time steps (sorted)
unique_times = np.sort(data['time'].unique())
print(f"Found {len(unique_times)} time steps")

# Create a mapping from time step to indices (each time step contains positions for all particles)
time_to_indices = {t: data.index[data['time'] == t].tolist() for t in unique_times}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 8))  # Larger figure for better quality
scat = ax.scatter([], [], s=10)  # Initial scatter plot, sizes will be updated
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

# Define the axis limits with some padding
x_min, x_max = np.min(x) * 1.1, np.max(x) * 1.1
y_min, y_max = np.min(y) * 1.1, np.max(y) * 1.1
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Particles Trajectories')
ax.grid(True, alpha=0.3)  # Add light grid for better position reference

# Update function for animation: update scatter plot with positions for all particles at current time step
def update(frame):
    current_time = unique_times[frame]
    indices = time_to_indices[current_time]
    current_x = x[indices]
    current_y = y[indices]
    current_masses = mass[indices]
    
    # Calculate sizes based on mass (log scale)
    sizes = np.array([get_size_from_mass(m) for m in current_masses])
    
    scat.set_offsets(np.column_stack((current_x, current_y)))
    scat.set_sizes(sizes)
    
    # Set colors for each particle based on its id
    current_particle_ids = particle_ids[indices]
    current_colors = [color_dict[p] for p in current_particle_ids]
    scat.set_color(current_colors)
    
    # Update time text
    years, days = convert_time(current_time)
    time_text.set_text(f'Time: {years} years, {days:.1f} days')
    return scat, time_text

# Create animation with frames equal to number of unique time steps
frames_count = len(unique_times)
frame_interval = 1000 / 30  # milliseconds per frame at 30fps (fixed at 30fps)

print(f"Creating animation with {frames_count} frames...")
ani = FuncAnimation(fig, update, frames=frames_count, blit=True, interval=frame_interval)

# Save the animation as an MP4 file (requires ffmpeg installed)
output_dir = script_dir / 'video_out'
output_dir.mkdir(exist_ok=True)

# Create a more concise descriptive name including execution mode
descriptive_name = f"video_{sim_years:.1f}yrs_{integration_method}_{initializer}_{force_method}_{execution_mode}.mp4"
output_filename = output_dir / descriptive_name

print(f"Saving animation to {output_filename}...")
# Use a higher fps and lower dpi for faster rendering
ani.save(output_filename, writer='ffmpeg', fps=30, dpi=150, bitrate=-1)
plt.close()

# Plot energy for particle_id 0 only (represents system energy)
print("Creating energy plot...")
particle0_data = data[data['particle_id'] == 0]

# Create a clean energy plot with dual y-axes
fig2, ax1 = plt.subplots(figsize=(12, 6))

# Calculate energy variation statistics
energy_values = particle0_data['energy'].values
energy_mean = np.mean(energy_values)
energy_min = np.min(energy_values)
energy_max = np.max(energy_values)
energy_initial = energy_values[0]
max_relative_variation = 100 * (energy_max - energy_min) / abs(energy_initial)

# Plot absolute energy
ax1.plot(particle0_data['time'], particle0_data['energy'], 
        color='#1f77b4', linewidth=1.5, label='Absolute Energy')
ax1.set_xlabel('Time [years]', fontsize=12)
ax1.set_ylabel('System Energy [absolute value]', fontsize=12)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Create a second y-axis to show relative energy change
ax2 = ax1.twinx()
relative_energy = [(e - energy_initial)/abs(energy_initial) * 100 for e in energy_values]
ax2.plot(particle0_data['time'], relative_energy, color='#ff7f0e', linestyle='--', linewidth=1.5, label='Relative Change [%]', alpha=0.3)
ax2.set_ylabel('Relative Energy Change [%]', color='#ff7f0e', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

# Add grid and title
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Total System Energy over {sim_years} years', fontsize=14)

# Add legend with both lines
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Add a simpler text box with energy statistics
stats_text = f"Energy Statistics:\nMean: {energy_mean:.10e}\nMax Relative Variation: {max_relative_variation:.8f}%"
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
ax1.text(0.05, 0.05, stats_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='left', bbox=props)

# Save with higher DPI for better quality
plt.tight_layout()
energy_filename = output_dir / f"energy_{sim_years:.1f}yrs_{integration_method}_{initializer}_{force_method}_{execution_mode}.png"
plt.savefig(energy_filename, dpi=300)
plt.close()

print("All done!")
