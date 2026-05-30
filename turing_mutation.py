import numpy as np
import matplotlib.pyplot as plt

# Spatial parameters
Lx = 10.0
Nx = 200
dx = Lx / (Nx - 1)
x = np.linspace(0, Lx, Nx)

# Time parameters
dt = 0.001
T = 50.0
steps = int(T / dt)

# Diffusion coefficients
DA = 0.01
DI = 1.0

# General parameters
muA = 1.0
muI = 1.0
alphaA = 0.1
alphaI = 0.1

def simulate(rhoA, rhoI):
    A = np.ones(Nx) * 0.1
    I = np.ones(Nx) * 0.1
    # Small spatial perturbation to trigger instability
    A += 0.01 * np.sin(2 * np.pi * x / Lx * 5)
    I += 0.01 * np.cos(2 * np.pi * x / Lx * 5)
    
    # Time evolution
    for t in range(steps):
        # Second-order central difference for Laplacian
        lapA = (np.roll(A, 1) - 2*A + np.roll(A, -1)) / dx**2
        lapI = (np.roll(I, 1) - 2*I + np.roll(I, -1)) / dx**2
        
        # Reaction terms
        fA = rhoA * A**2 / I - muA * A + alphaA
        fI = rhoI * A**2 - muI * I + alphaI
        
        # Update
        A_new = A + dt * (DA * lapA + fA)
        I_new = I + dt * (DI * lapI + fI)
        
        # Numerical stability clipping
        A = np.clip(A_new, 0.0, 10.0)
        I = np.clip(I_new, 0.0, 10.0)
    
    return A, I

# Wild-type simulation
A_wt, I_wt = simulate(rhoA=1.0, rhoI=1.0)
# Mutant simulation (reduced rhoA)
A_mut, I_mut = simulate(rhoA=0.3, rhoI=1.0)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, A_wt, 'b-', linewidth=1.5, label='A (activator)')
ax1.plot(x, I_wt, 'r--', linewidth=1.5, label='I (inhibitor)')
ax1.set_title(r'Wild-type ($\beta = 1.0$)', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('Concentration')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x, A_mut, 'b-', linewidth=1.5, label='A (activator)')
ax2.plot(x, I_mut, 'r--', linewidth=1.5, label='I (inhibitor)')
ax2.set_title(r'Mutant ($\beta = 0.3$)', fontsize=14)
ax2.set_xlabel('x')
ax2.set_ylabel('Concentration')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('turing_mutation.pdf')
plt.show()
