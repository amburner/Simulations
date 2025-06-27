import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import spsolve

# Parameters
L = 10.0         # Physical domain size
J = 50           # Grid resolution
T = 10.0         # Total time
N = 2000         # Number of time steps
kappa = 1.0      # Diffusion coefficient

# Derived quantities
dx = L / (J - 1)
dt = T / (N - 1)
r = kappa * dt / dx**2
x = np.linspace(0, L, J)
y = np.linspace(0, L, J)
X, Y = np.meshgrid(x, y)

# Initial condition: centered Gaussian
U = np.zeros((J, J))
cx, cy = J // 2, J // 2
spread = 10
for i in range(J):
    for j in range(J):
        U[i, j] = np.exp(-((i - cx)**2 + (j - cy)**2) / (2 * spread**2))
U *= 1000.0 / np.sum(U)  # Normalize total heat
u = U.flatten()

# Laplacian with Neumann BCs (1D unscaled)
def laplacian_1d_unscaled(J):
    main_diag = -2.0 * np.ones(J)
    off_diag = np.ones(J - 1)
    data = [off_diag, main_diag, off_diag]
    offsets = [-1, 0, 1]
    L = diags(data, offsets).toarray()
    L[0, 0] = -1
    L[0, 1] = 1
    L[-1, -1] = -1
    L[-1, -2] = 1
    return csc_matrix(L)

# 2D Laplacian with Kronecker sum
L1D = laplacian_1d_unscaled(J)
L2D = kron(eye(J), L1D) + kron(L1D, eye(J))

# Backward Euler matrix: (I - r * Laplacian)
A_BE = eye(J * J) - r * L2D

magic = 10

# Time stepping
U_record = [U.copy()]
for t in range(1, N):
    u = spsolve(A_BE, u)
    if t % magic == 0:
        U_record.append(u.reshape(J, J))

U_record = np.array(U_record)

# Animation setup
frames = range(len(U_record))
zmin = np.min(U_record[0])
zmax = np.max(U_record[0])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    ax.set_title(f"Time Step: {frame * magic}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')
    ax.set_zlim(zmin, zmax)
    return ax.plot_surface(X, Y, U_record[frame], cmap='inferno', vmin=zmin, vmax=zmax)

'''
# Run animation
anim = FuncAnimation(fig, update, frames=frames, interval=30)
plt.show()

# Optional: print total heat over time
for i, U in enumerate(U_record):
    print(f"Time {i * magic}: total heat = {np.sum(U)}")

import matplotlib.animation as animation


# Save the animation as a GIF
gif_writer = animation.PillowWriter(fps=30)
anim.save("heat.gif", writer=gif_writer)
'''

zmin_all = 0
zmax_all = np.max(U_record)

# Save the initial time step as a PNG
fig_init = plt.figure(figsize=(8, 6))
ax_init = fig_init.add_subplot(111, projection='3d')
ax_init.plot_surface(X, Y, U_record[0], cmap='inferno', vmin=zmin, vmax=zmax)
ax_init.set_title("Initial Condition (t = 0)")
ax_init.set_xlabel("X")
ax_init.set_ylabel("Y")
ax_init.set_zlabel("Temperature")
ax_init.set_zlim(zmin_all, zmax_all)
plt.savefig("initial_timestep.png", dpi=300, bbox_inches='tight')
plt.close(fig_init)

# Save the final time step as a PNG
fig_final = plt.figure(figsize=(8, 6))
ax_final = fig_final.add_subplot(111, projection='3d')
ax_final.plot_surface(X, Y, U_record[-1], cmap='inferno', vmin=zmin, vmax=zmax)
ax_final.set_title(f"Final Condition (t = {T:.2f})")
ax_final.set_xlabel("X")
ax_final.set_ylabel("Y")
ax_final.set_zlabel("Temperature")
ax_final.set_zlim(zmin_all, zmax_all)
plt.savefig("final_timestep.png", dpi=300, bbox_inches='tight')
plt.close(fig_final)

