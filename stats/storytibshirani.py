import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

# 1. Simulate text distance scores (90% null states, 10% genuine anomalies)
np.random.seed(42)
n_text_segments = 1000

is_anomaly_ground_truth = np.random.choice([0, 1], size=n_text_segments, p=[0.90, 0.10])
distance_scores = np.zeros(n_text_segments)

for i in range(n_text_segments):
    if is_anomaly_ground_truth[i] == 1:
        distance_scores[i] = np.random.normal(loc=3.5, scale=0.8)  # Anomalous text blocks
    else:
        distance_scores[i] = np.random.normal(loc=1.5, scale=0.5)  # Regular text blocks

# Convert distance scores into empirical p-values against regular text baseline properties
p_values = 1 - norm.cdf(distance_scores, loc=1.5, scale=0.5)

# 2. Storey-Tibshirani Method: Extract the True Proportion of Null States (pi_0)
def estimate_pi0(p_vals, lambda_sweep=np.arange(0.05, 0.95, 0.05)):
    """Estimates pi_0 by sweeping lambda and interpolating its convergence limit at lambda -> 1."""
    n = len(p_vals)
    pi0_estimates = []
    
    # Calculate empirical pi_0 for each lambda tuning value
    for lmbda in lambda_sweep:
        num_above_lambda = np.sum(p_vals > lmbda)
        pi0_lmbda = num_above_lambda / (n * (1 - lmbda))
        pi0_estimates.append(pi0_lmbda)
        
    pi0_estimates = np.array(pi0_estimates)
    
    # Fit a smooth cubic spline over the sweep array to evaluate the target convergence limit at lambda=1.0
    spline = UnivariateSpline(lambda_sweep, pi0_estimates, k=3, s=0.1)
    estimated_pi0 = spline(1.0)
    
    # Bound the final estimate value mathematically between absolute baseline constraints
    estimated_pi0 = np.clip(estimated_pi0, 0.0, 1.0)
    
    return estimated_pi0, lambda_sweep, pi0_estimates, spline

pi0_hat, lambdas, pi0_arr, fit_spline = estimate_pi0(p_values)

# 3. Less Conservative Adaptive FDR Testing Adjustment
def apply_adaptive_fdr(p_vals, pi0, target_q=0.05):
    """Applies Benjamini-Hochberg adjusted for the estimated null proportion pi_0."""
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p_vals = p_vals[sorted_indices]
    
    # Adaptive BH adjustment formula: critical threshold line is expanded to: (i / n) * (q / pi0)
    adjusted_critical_values = [(i + 1) / n * (target_q / pi0) for i in range(n)]
    
    significant_indices = np.where(sorted_p_vals <= adjusted_critical_values)[0]
    
    final_flags = np.zeros(n, dtype=int)
    if len(significant_indices) > 0:
        max_sig_idx = np.max(significant_indices)
        final_flags[sorted_indices[:max_sig_idx + 1]] = 1
        
    return final_flags

target_fdr = 0.05
adaptive_flags = apply_adaptive_fdr(p_values, pi0=pi0_hat, target_q=target_fdr)

# 4. Plot the Storey-Tibshirani Tuning and Curve Extrapolation Line
fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(lambdas, pi0_arr, color='blue', edgecolor='black', s=40, label=r'Empirical $\hat{\pi}_0(\lambda)$ points', zorder=3)
lambda_dense = np.linspace(0.05, 1.0, 100)
ax.plot(lambda_dense, fit_spline(lambda_dense), color='teal', lw=2, linestyle='-', label='Cubic Spline Interpolation')

# Highlight the final isolated intercept point at lambda=1.0
ax.scatter(1.0, pi0_hat, color='darkred', marker='X', s=120, zorder=5, label=f'Optimized Intercept $\hat{{\pi}}_0$ = {pi0_hat:.3f}')

ax.set_title(r'Storey-Tibshirani $\pi_0$ Estimation Protocol Loop' + f'\nEstimated Proportion of Regular Text Null States: {pi0_hat*100:.1f}%', fontsize=12, pad=12, fontweight='bold')
ax.set_xlabel(r'Tuning Parameter ($\lambda$ Boundary)', fontsize=11)
ax.set_ylabel(r'Proportion Value ($\hat{\pi}_0$)', fontsize=11)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.1])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.show()
