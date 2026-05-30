import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Automatically find fonts that support Chinese on macOS (optional, can be removed for English)
# Since the plot is now in English, we skip font configuration; just use default.
plt.rcParams['axes.unicode_minus'] = False

def quasispecies_simulation(L, q, sigma, x0_init=0.9, generations=200):
    """
    Deterministic simulation of quasispecies dynamics.
    L: sequence length
    q: single-symbol replication accuracy
    sigma: fitness advantage of the functional sequence
    x0_init: initial frequency of the master sequence
    generations: number of generations to simulate
    """
    Q = q**L
    x0 = x0_init
    history = [x0]
    
    for _ in range(generations):
        # ODE: dx0/dt = x0 * (Q*sigma - 1 - (sigma-1)*x0)
        # Use explicit Euler integration with dt small enough for stability
        dt = 0.1
        dx0 = x0 * (Q*sigma - 1 - (sigma - 1) * x0) * dt
        x0 = x0 + dx0
        # Keep x0 within [0,1]
        x0 = max(0.0, min(1.0, x0))
        history.append(x0)
    return history

# Parameter settings
L_values = [10, 20, 50, 100]
q = 0.99
sigma = 2.0

plt.figure(figsize=(10, 6))
for L in L_values:
    history = quasispecies_simulation(L, q, sigma)
    Qs = q**L * sigma
    label = f"L={L}, Qσ={Qs:.3f}"
    plt.plot(range(len(history)), history, label=label)

plt.axhline(y=1e-3, color='grey', linestyle='--', linewidth=1, label='Extinction threshold')
plt.yscale('log')
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Master sequence concentration $x_0$', fontsize=14)
plt.title('Error Catastrophe: Phase Transition of Information Annihilation', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('error_catastrophe.pdf')
plt.show()
