import numpy as np
import matplotlib.pyplot as plt

# Chinese font settings for macOS (kept for compatibility, but labels are in English)
plt.rcParams['font.sans-serif'] = ['STHeiti']   # Use Heiti to support Chinese (if needed)
plt.rcParams['axes.unicode_minus'] = False      # Fix minus sign display

def fixation_prob(Qsigma, Ne):
    """
    Calculate the fixation probability starting from a single functional type (Kimura's formula).
    Supports scalar or array input, fully vectorized, avoids numerical overflow.
    """
    # Handle near‑neutral cases (Qsigma close to 1)
    near_neutral = np.abs(Qsigma - 1.0) < 1e-12
    result = np.zeros_like(Qsigma, dtype=float)
    
    # Neutral case: Pfix = 1/Ne
    result[near_neutral] = 1.0 / Ne[near_neutral]
    
    # Non‑neutral cases
    not_neutral = ~near_neutral
    if np.any(not_neutral):
        Qs = Qsigma[not_neutral]
        N = Ne[not_neutral]
        s = Qs - 1.0          # Selection advantage (can be positive or negative)
        
        # Exponential arguments in Kimura's formula
        exp_arg1 = -2 * s
        exp_arg2 = -2 * N * s
        
        # Clip exponents to [-700, 700] to avoid overflow
        # exp(700) ≈ 1e304, still within double precision (max ~1e308)
        # Values beyond 700 are clipped, ratio remains approximately correct
        exp_arg1_clipped = np.clip(exp_arg1, -700, 700)
        exp_arg2_clipped = np.clip(exp_arg2, -700, 700)
        
        # Compute numerator and denominator
        num = 1.0 - np.exp(exp_arg1_clipped)
        denom = 1.0 - np.exp(exp_arg2_clipped)
        
        result[not_neutral] = num / denom
    
    # Theoretically fixation probability should be in [0,1]; clip due to numerical errors
    result = np.clip(result, 0.0, 1.0)
    return result

# Parameter ranges
Ne_vals = np.logspace(1, 6, 200)      # From 10 to 10^6
Qsigma_vals = np.linspace(0.8, 1.5, 200)
Ne_grid, Qs_grid = np.meshgrid(Ne_vals, Qsigma_vals)
Pfix = fixation_prob(Qs_grid, Ne_grid)

# Plot phase diagram
plt.figure(figsize=(10, 8))
logPfix = np.log10(np.maximum(Pfix, 1e-300))   # Avoid log10(0)
contour = plt.contourf(Ne_grid, Qs_grid, logPfix, levels=50, cmap='viridis')
cbar = plt.colorbar(contour, label=r'$\log_{10}(P_{\mathrm{fix}})$')
plt.xscale('log')
plt.xlabel(r'Effective population size $N_e$', fontsize=14)
plt.ylabel(r'$Q\sigma$', fontsize=14)
plt.title('Lineage persistence phase diagram', fontsize=16)

# Neutral line Qσ = 1
plt.axhline(y=1.0, color='white', linestyle='--', linewidth=2, label=r'Error threshold $Q\sigma=1$')
# Lineage persistence boundary: Ne * (Qσ-1) = ln Ne
Ne_boundary = np.logspace(2, 5, 100)
Qsigma_boundary = 1 + np.log(Ne_boundary) / Ne_boundary
plt.plot(Ne_boundary, Qsigma_boundary, 'r-', linewidth=3, label=r'Persistence boundary $N_e(Q\sigma-1) = \ln N_e$')

plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('lineage_phase_diagram.pdf')
plt.show()
