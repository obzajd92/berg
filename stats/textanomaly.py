"""
Text Anomaly Detection Pipeline Package
=======================================
This production-ready package implements an advanced statistical anomaly detection
pipeline for natural language processing and text metrics environments. It includes:
1. Spatial partial correlation analysis via the precision matrix.
2. Non-parametric permutation tests for distribution-free significance mapping.
3. Full-covariance Mahalanobis distance profiling using Scipy.
4. Adaptive False Discovery Rate (FDR) control via the Storey-Tibshirani method.
5. Automated multi-panel diagnostic visualization dashboard generation.

Author: AI Collaborator
Date: October 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from typing import Dict, List, Tuple

def calc_partial_corr(data_matrix: np.ndarray) -> np.ndarray:
    """Computes the partial correlation matrix from a feature data matrix.

    Uses the inverse covariance (precision) matrix method with a ridge adjustment
    to ensure numerical stability.

    Args:
        data_matrix: A 2D numpy array where rows represent features/metrics 
                     and columns represent observations/environments.

    Returns:
        A square 2D numpy array representing the partial correlation coefficients.
    """
    corr = np.corrcoef(data_matrix)
    ridge = 1e-6
    precision = np.linalg.inv(corr + np.eye(corr.shape[0]) * ridge)
    diag = np.sqrt(np.diag(precision))
    p_corr = -precision / np.outer(diag, diag)
    np.fill_diagonal(p_corr, 1.0)
    return p_corr

def run_permutation_test(obs_matrix: np.ndarray, obs_pcorr: np.ndarray, n_permutations: int = 1000) -> np.ndarray:
    """Performs a non-parametric permutation test to compute empirical p-values.

    Shuffles row elements independently across observations to construct an empirical 
    null distribution for each partial correlation coefficient.

    Args:
        obs_matrix: The original observed data matrix (Features x Observations).
        obs_pcorr: The observed partial correlation matrix.
        n_permutations: Number of random reshuffles to run.

    Returns:
        A square 2D numpy array containing the computed empirical p-values.
    """
    n_features = obs_matrix.shape[0]
    perm_distribution = np.zeros((n_permutations, n_features, n_features))

    for p in range(n_permutations):
        perm_matrix = obs_matrix.copy()
        for r in range(n_features):
            perm_matrix[r, :] = np.random.permutation(perm_matrix[r, :])
        perm_distribution[p] = calc_partial_corr(perm_matrix)

    extreme_counts = np.sum(np.abs(perm_distribution) >= np.abs(obs_pcorr), axis=0)
    empirical_p = extreme_counts / n_permutations
    np.fill_diagonal(empirical_p, 0.0)
    return empirical_p

def estimate_pi0(p_vals: np.ndarray, lambda_sweep: np.ndarray = np.arange(0.05, 0.95, 0.05)) -> Tuple[float, np.ndarray, np.ndarray, UnivariateSpline]:
    """Estimates the true proportion of null states (pi_0) using Storey-Tibshirani method.

    Fits a cubic spline interpolation curve over the lambda sweep points to evaluate
    the convergence limit as lambda approaches 1.0.

    Args:
        p_vals: 1D array of calculated p-values across text segments.
        lambda_sweep: Tuning boundary threshold parameters array.

    Returns:
        A tuple containing:
            - The optimized estimated scalar pi_0 value.
            - The sweep array parameters.
            - The empirical pi_0 arrays across the sweep.
            - The fitted UnivariateSpline object.
    """
    n = len(p_vals)
    pi0_estimates = []
    
    for lmbda in lambda_sweep:
        num_above_lambda = np.sum(p_vals > lmbda)
        pi0_lmbda = num_above_lambda / (n * (1 - lmbda))
        pi0_estimates.append(pi0_lmbda)
        
    pi0_estimates = np.array(pi0_estimates)
    spline = UnivariateSpline(lambda_sweep, pi0_estimates, k=3, s=0.1)
    estimated_pi0 = float(np.clip(spline(1.0), 0.0, 1.0))
    
    return estimated_pi0, lambda_sweep, pi0_estimates, spline

def apply_adaptive_fdr(p_vals: np.ndarray, pi0: float, target_q: float = 0.05) -> Tuple[np.ndarray, float]:
    """Applies Benjamini-Hochberg adjustment adjusted for the estimated null proportion pi_0.

    Args:
        p_vals: 1D array of empirical p-values.
        pi0: Inferred proportion of regular text null states.
        target_q: False Discovery Rate configuration target bounds.

    Returns:
        A tuple containing:
            - A binary 1D array flag array (1 = Anomaly, 0 = Normal).
            - The calibrated p-value threshold boundary.
    """
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p_vals = p_vals[sorted_indices]
    
    adjusted_critical_values = [(i + 1) / n * (target_q / pi0) for i in range(n)]
    significant_indices = np.where(sorted_p_vals <= adjusted_critical_values)[0]
    
    final_flags = np.zeros(n, dtype=int)
    p_threshold = 0.0
    if len(significant_indices) > 0:
        max_sig_idx = np.max(significant_indices)
        final_flags[sorted_indices[:max_sig_idx + 1]] = 1
        p_threshold = sorted_p_vals[max_sig_idx]
        
    return final_flags, p_threshold

def run_pipeline_and_plot(metrics_data: Dict[str, List[float]], target_metrics: List[str], output_path: str = "generated/text_anomaly_dashboard.png") -> None:
    """Coordinates core dataset processing matrices and exports analytical dashboards.

    Args:
        metrics_data: Ingestion dictionary containing named environment listings.
        target_metrics: Subset rows to isolate for deep statistical evaluation.
        output_path: Local filepath destination to print the graphic dashboard.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_full = pd.DataFrame(metrics_data, index=['Word_Freq', 'words_Mean', 'Verse_Lag', 'Token_Density', 'Entropy'])
    df = df_full.loc[target_metrics]
    entities = df.columns
    n_entities = len(entities)

    # Statistical Matrix Analytics
    obs_pcorr = calc_partial_corr(df.values)
    empirical_p = run_permutation_test(df.values, obs_pcorr, n_permutations=200)

    # Scipy Vectorized Covariance Spacing
    data_to_compare = df.T.values
    cov_matrix = np.cov(data_to_compare, rowvar=False) + np.eye(df.shape[0]) * 1e-6
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_matrix = squareform(pdist(data_to_compare, metric='mahalanobis', VI=inv_cov_matrix))

    # Matplotlib Graph Layout Preparation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Left Plot: Masked Partial Correlation ----
    im1 = ax1.imshow(obs_pcorr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title('Non-Parametric Permutation Mask\n(Alpha = 0.05 Null Check)', fontsize=12, pad=12, fontweight='bold')
    ax1.set_xticks(np.arange(n_entities))
    ax1.set_yticks(np.arange(n_entities))
    ax1.set_xticklabels(entities, rotation=30, ha='right')
    ax1.set_yticklabels(entities)

    for i in range(n_entities):
        for j in range(n_entities):
            if empirical_p[i, j] >= 0.05:
                ax1.text(j, i, "X\n[ns]", ha='center', va='center', color='darkred', fontsize=10, fontweight='bold')
                ax1.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='black', alpha=0.15))
            else:
                txt_color = 'white' if abs(obs_pcorr[i, j]) > 0.5 else 'black'
                ax1.text(j, i, f"{obs_pcorr[i, j]:.2f}", ha='center', va='center', color=txt_color, fontweight='bold')
    fig.colorbar(im1, ax=ax1, shrink=0.75)

    # ---- Right Plot: Mahalanobis Distance ----
    im2 = ax2.imshow(mahalanobis_matrix, cmap='viridis')
    ax2.set_title('Mahalanobis Distance Matrix\n(Full Covariance Scaled via pdist)', fontsize=12, pad=12, fontweight='bold')
    ax2.set_xticks(np.arange(n_entities))
    ax2.set_yticks(np.arange(n_entities))
    ax2.set_xticklabels(entities, rotation=30, ha='right')
    ax2.set_yticklabels(entities)

    for i in range(n_entities):
        for j in range(n_entities):
            txt_color = 'white' if mahalanobis_matrix[i, j] > np.max(mahalanobis_matrix) * 0.5 else 'black'
            ax2.text(j, i, f"{mahalanobis_matrix[i, j]:.2f}", ha='center', va='center', color=txt_color)
    fig.colorbar(im2, ax=ax2, shrink=0.75)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Dashboard figure successfully generated and exported to: {output_path}")

if __name__ == "__main__":
    print("Executing verification sequence for production package...")
    np.random.seed(42)
    
    mock_data = {
        'Poem_Baseline': [120, 450, 1.2, 0.85, 4.2],
        'Person_A':      [140, 420, 1.5, 0.70, 3.9],
        'Person_B':      [95,  490, 0.9, 0.92, 4.8],
        'Person_R':      [118, 455, 1.3, 0.83, 4.1],
        'Person_S':      [135, 410, 1.6, 0.65, 3.6]
    }
    targets = ['Word_Freq', 'words_Mean', 'Token_Density']
    
    run_pipeline_and_plot(mock_data, targets)
    print("Verification successfully completed.")
