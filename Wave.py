import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Parameters
L = 10.0         # Domain size
J = 40        # Grid resolution (faster)
T = 50.0      # Total time (optional)
N = 5000      # Time steps
magic = 10   # Save every 100th frame
c = 1.0          # Wave speed
gamma = 0.1      # Damping coefficient

# New parameters for initial Gaussian
A = 1.0          # Amplitude
spread = 0.5     # Standard deviation

dx = L / (J - 1)
dt = T / (N - 1)
x = np.linspace(0, L, J)
y = np.linspace(0, L, J)
X, Y = np.meshgrid(x, y)

# Initial Gaussian displacement
U0 = A * np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (2 * spread**2))
U1 = U0.copy()

# Initialize previous and current states
U_prev = U0.copy()
U_curr = U1.copy()

# Store for animation and energy
U_anim = [U0.copy()]
energies = []

# Time stepping with damping
for n in range(N):
    U_next = np.zeros_like(U_curr)

    # Discrete Laplacian
    lap = (
        U_curr[2:, 1:-1] + U_curr[:-2, 1:-1] +
        U_curr[1:-1, 2:] + U_curr[1:-1, :-2] -
        4 * U_curr[1:-1, 1:-1]
    )

    # Central difference with damping
    U_next[1:-1, 1:-1] = (
        (2 - gamma * dt) * U_curr[1:-1, 1:-1]
        - (1 - gamma * dt) * U_prev[1:-1, 1:-1]
        + (c * dt / dx)**2 * lap
    )

    # Neumann BCs
    U_next[0, :] = U_next[1, :]
    U_next[-1, :] = U_next[-2, :]
    U_next[:, 0] = U_next[:, 1]
    U_next[:, -1] = U_next[:, -2]

    # Save animation frames and energy
    if n % magic == 0:
        velocity = (U_next - U_prev) / (2 * dt)
        kinetic = 0.5 * velocity**2

        dudx = (U_curr[:, 2:] - U_curr[:, :-2]) / (2 * dx)
        dudy = (U_curr[2:, :] - U_curr[:-2, :]) / (2 * dx)

        dudx_full = np.zeros_like(U_curr)
        dudx_full[:, 1:-1] = dudx
        dudy_full = np.zeros_like(U_curr)
        dudy_full[1:-1, :] = dudy

        potential = 0.5 * c**2 * (dudx_full**2 + dudy_full**2)
        total_energy = np.sum(kinetic + potential)
        energies.append(total_energy)
        U_anim.append(U_curr.copy())

    # Advance states
    U_prev = U_curr
    U_curr = U_next

# Create time vector for energy plot
time_values = np.arange(len(energies)) * magic * dt

# Plot energy decay
plt.figure(figsize=(10, 5))
plt.plot(time_values, energies, label='Total Mechanical Energy (Damped)', color='darkred')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy Decay in Damped 2D Wave System')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Animation setup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
zmin, zmax = np.min(U_anim[0]), np.max(U_anim[0])

def update(frame):
    ax.clear()
    ax.set_title(f"Time Step: {frame * magic}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Displacement")
    ax.set_zlim(zmin, zmax)
    return ax.plot_surface(X, Y, U_anim[frame], cmap='viridis', vmin=zmin, vmax=zmax)


# Create and save the animation
anim = FuncAnimation(fig, update, frames=len(U_anim), interval=30)
anim.save("damped_wave.gif", writer=animation.PillowWriter(fps=30))



zmin_all = np.min(U_anim)
zmax_all = np.max(U_anim)


# Save the initial time step as a PNG
fig_init = plt.figure(figsize=(8, 6))
ax_init = fig_init.add_subplot(111, projection='3d')
ax_init.plot_surface(X, Y, U_anim[0], cmap='inferno', vmin=zmin_all, vmax=zmax_all)
ax_init.set_title("Initial Condition (t = 0)")
ax_init.set_xlabel("X")
ax_init.set_ylabel("Y")
ax_init.set_zlabel("Temperature")
ax_init.set_zlim(zmin_all, zmax_all)
plt.savefig("initial_wave.png", dpi=300, bbox_inches='tight')
plt.close(fig_init)

# Save the final time step as a PNG
fig_final = plt.figure(figsize=(8, 6))
ax_final = fig_final.add_subplot(111, projection='3d')
ax_final.plot_surface(X, Y, U_anim[-1], cmap='inferno', vmin=zmin_all, vmax=zmax_all)
ax_final.set_title(f"Final Condition (t = {T:.2f})")
ax_final.set_xlabel("X")
ax_final.set_ylabel("Y")
ax_final.set_zlabel("Temperature")
ax_final.set_zlim(zmin_all, zmax_all)
plt.savefig("final_wave.png", dpi=300, bbox_inches='tight')
plt.close(fig_final)
