# Code for Least-Norm Least-Squares Problem (Section 4.2)

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import time

from Regularisation_Methods_for_HVIs import (
    Scheduler,
    Algorithm,
    LipschitzMonotoneOperator, MaximallyMonotoneOperator,
    Problem
)

np.random.seed(42)

###################
# Run Experiments #
###################

# Dimensions to be run
dimensions = [(70, 100), (100, 200), (100, 500), (300, 500)]

# Rank of random matrix
r = 50

# Relaxation parameter in KM iteration
theta = 0.75

# Proximal Parameter (There must be 3 of them for plotting purposes)
alphas = [0.1, 1, 10]

# Results to be stored
results = [[None for _ in alphas] for __ in dimensions]

# Analytical solutions to be stored
analytical_solutions = [None for _ in dimensions]
matrices = [None for _ in dimensions]
vectors = [None for _ in dimensions]

for dim_idx, (m, n) in enumerate(dimensions):
    while True:
        ######################
        # Problem Definition #
        ######################
        Q1 = np.random.normal(0, 1, (m, r))
        Q2 = np.random.normal(0, 1, (r, n))
        Q = Q1.dot(Q2)

        # Clip singular values of Q
        U, s, Vh = np.linalg.svd(Q, full_matrices=False)
        Q = U @ np.diag(np.clip(s, a_min=0, a_max=10)) @ Vh

        # Generate a solution vector s with 20 nonzero entries
        s = np.random.uniform(0, 10, n)
        s[20:] = 0
        s = np.random.permutation(s)

        # Generate noise vector and compute b.
        nu = np.random.normal(0, 0.1, m)
        b = Q.dot(s) + nu

        # Store problem data
        analytical_solutions[dim_idx] = sp.linalg.pinv(Q) @ b
        matrices[dim_idx] = Q
        vectors[dim_idx] = b

        G = LipschitzMonotoneOperator(evaluate=lambda x: x, L=1)

        largest_eigenvalue = np.real(np.max(np.linalg.eig(Q.T @ Q)[0]))
        F = LipschitzMonotoneOperator(evaluate=lambda x: Q.T @ (Q @ x - b), L=largest_eigenvalue)

        proj_X = lambda x: np.clip(x, -1000, 1000)
        A = MaximallyMonotoneOperator(evaluate_resolvent=lambda x, gamma: proj_X(x))

        # It must hold true that the analytical solution is within the bounds. If not, regenerate problem
        if False in (proj_X(analytical_solutions[dim_idx]) == analytical_solutions[dim_idx]):
            print("Solution not in set")
            print(analytical_solutions[dim_idx])
            exit()
        else:
            break

    # Create problem instance with defined operators
    problem = Problem(leader=G, follower=A + F)

    # Solve the problem for each set of parameters
    for alpha_idx, alpha in enumerate(alphas):
        ########################
        # Algorithm Parameters #
        ########################

        max_iterations = 2000  # Number of iterations for the algorithm
        beta = Scheduler(0.55)  # Beta parameter for Tikhonov regularization
        epsilon = Scheduler(1, 1e-3 * alpha)  # Epsilon parameter for convergence criterion of sub-problem

        # Create instance of algorithm with defined parameters
        algorithm = Algorithm(problem, max_iterations, beta, epsilon)

        # Random initial point
        w0 = np.random.normal(0, 0.1, n)

        results[dim_idx][alpha_idx] = algorithm.solve(w0, theta, alpha, progress=tqdm)

fig_width, fig_height = 3, 1
fig, axs = plt.subplots(fig_height, fig_width, figsize=(10, 2.5), dpi=300, sharex=True, sharey=True)

for idx, (ax, alpha) in enumerate(zip(axs.flatten(), alphas)):
    for idx2, ((P, Q), res, sol) in enumerate(zip(dimensions, results, analytical_solutions)):
        # sol2 = results[idx2][idx][-1]
        ax.semilogy([np.linalg.norm(v - sol) for v in results[idx2][idx]], "--", label=rf"$(P,Q)=({P},{Q})$")

    # Add badge with value of alpha
    ax.text(0.93, 0.92, rf"$\alpha={alpha}$", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add labels to axes if they are on the border
    if idx // fig_width == fig_height - 1: ax.set_xlabel(r"Iteration counter ($n$)")
    if idx % fig_width == 0: ax.set_ylabel(r"Distance to solution")

    # Show grid
    ax.grid(True, linestyle='--', alpha=0.7)

handles, labels = axs.flatten()[0].get_legend_handles_labels()
plt.tight_layout()
fig.legend(handles, labels, loc='lower center', ncol=len(dimensions), bbox_to_anchor=(0.5, -0.1))

fig_name = f"LS_dist_{time.time()}"
plt.savefig(f"figures/{fig_name}.png", bbox_inches='tight')

fig_width, fig_height = 3, 1
fig, axs = plt.subplots(fig_height, fig_width, figsize=(10, 2.5), dpi=300, sharex=True, sharey=True)

for idx, (ax, alpha) in enumerate(zip(axs.flatten(), alphas)):
    for idx2, ((P, Q), res, sol) in enumerate(zip(dimensions, results, analytical_solutions)):
        ax.semilogy([np.linalg.norm(matrices[idx2] @ v - vectors[idx2]) ** 2 - np.linalg.norm(
            matrices[idx2] @ sol - vectors[idx2]) ** 2
                     for v in results[idx2][idx]], "--", label=rf"$(P,Q)=({P},{Q})$")

    # Add badge with value of alpha
    ax.text(0.93, 0.92, rf"$\alpha={alpha}$", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add labels to axes if they are on the border
    if idx // fig_width == fig_height - 1: ax.set_xlabel(r"Iteration counter ($n$)")
    if idx % fig_width == 0: ax.set_ylabel("Difference in lower-level" "\n" "function values")

    # Show grid
    ax.grid(True, linestyle='--', alpha=0.7)

handles, labels = axs.flatten()[0].get_legend_handles_labels()
plt.tight_layout()
fig.legend(handles, labels, loc='lower center', ncol=len(dimensions), bbox_to_anchor=(0.5, -0.1))

fig_name = f"LS_func_{time.time()}"
plt.savefig(f"figures/{fig_name}.png", bbox_inches='tight')
