# Code for Image Inpainting Problem (Section 4.3)

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import time
import pickle

from Regularisation_Methods_for_HVIs import (
    Scheduler,
    Algorithm,
    LipschitzMonotoneOperator, MaximallyMonotoneOperator,
    Problem
)

np.random.seed(42)

PERCENTAGE = 0.2
SIZE = 256

img = Image.open('image.jpeg').resize((SIZE, SIZE))
X_ref = np.asarray(ImageOps.grayscale(img)).astype("float32")
mask = np.random.binomial(1, 1 - PERCENTAGE, (SIZE, SIZE))
R = lambda X: X * mask
X_corrupt = R(X_ref)

######################
# Problem Definition #
######################

# Main problem: VI(G, Zer(F+A+B))
G = LipschitzMonotoneOperator(evaluate=lambda X: X, L=1)
F = LipschitzMonotoneOperator(evaluate=lambda X: R(R(X) - X_corrupt), L=1.0)

def soft_shrinkage(X, theta):
    def shrinkage(x, theta):
        if x >= theta: return x - theta
        elif x <= - theta: return x + theta
        else: return 0
    U, S, Vh = np.linalg.svd(X)
    S_shrinked = np.array([shrinkage(s, theta) for s in S])
    return U @ np.diag(S_shrinked) @ Vh
A = MaximallyMonotoneOperator(evaluate_resolvent=soft_shrinkage)
proj_X = lambda x: np.minimum(np.maximum(x, np.zeros_like(X_ref)), np.zeros_like(X_ref) + 255)
B = MaximallyMonotoneOperator(evaluate_resolvent=lambda x, gamma: proj_X(x))

# Create problem instance with defined operators
sigma = 50
problem = Problem(leader=G, follower=F+sigma*A+B)

########################
# Algorithm Parameters #
########################

max_iterations = 5000            # Number of iterations for the algorithm
beta = Scheduler(0.55)           # Beta parameter for Tikhonov regularization
epsilon = Scheduler(2, 2)        # Epsilon parameter for convergence criterion of sub-problem

theta = 0.75

###################
# Run Experiments #
###################

TIME = time.time()

ALPHAS = [0.1, 1, 10, 100]

Xs = [None] * len(ALPHAS)

def run_experiment(i):
    # Create instance of algorithm with defined parameters
    algorithm = Algorithm(problem, max_iterations, beta, epsilon)
    Xs[i] = algorithm.solve(X_corrupt, theta, ALPHAS[i], progress=tqdm, method="TOS")
    print(f"Number of iterations for alpha={ALPHAS[i]}: {algorithm.total_inner_iterations}")

if False:
    run_experiment(0)
    run_experiment(1)
    run_experiment(2)
    run_experiment(3)

    with open(f'saved/{TIME}.pkl', 'wb') as output:
        pickle.dump(Xs, output, pickle.HIGHEST_PROTOCOL)
else:
    TIME_SAVED = '1766140202.0866199'
    with open(f'saved/{TIME_SAVED}.pkl', 'rb') as input:
        Xs = pickle.load(input)

##########################
# Plot Lower-Level Curve #
##########################

plt.figure(figsize=(5, 3), dpi=300)

def plot_curve(i):
    def lowerlevel(X):
        return 0.5 * np.linalg.norm(R(X) - X_corrupt) + sigma * np.linalg.norm(X, 'nuc')
    plt.semilogy([lowerlevel(X) for X in Xs[i]], label=rf'$\alpha={ALPHAS[i]}$')

plot_curve(0)
plot_curve(1)
plot_curve(2)
plot_curve(3)

plt.xlabel(r'Iterations counter ($n$)')
plt.ylabel("Lower-level objective function")

plt.legend(loc='best')

fig_name = f"3_IP_Curves_{TIME}"
plt.savefig(f"figures/{fig_name}.png", bbox_inches='tight')

########################
# Plot Restored Images #
########################

fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=300)

axs[0,0].imshow(X_ref, cmap=cm.Greys_r, vmin=0, vmax=255)
axs[0,0].set_title("Original Image")
axs[0,0].set_axis_off()

axs[0,1].imshow(X_corrupt, cmap=cm.Greys_r, vmin=0, vmax=255)
axs[0,1].set_title("Corrupt Image")
axs[0,1].set_axis_off()

def plot_restored(i, ax):
    X_sol = Xs[i][-1]
    ax.imshow(X_sol, cmap=cm.Greys_r, vmin=0, vmax=255)
    ax.set_title(rf"Restored Image $\alpha={ALPHAS[i]}$")
    ax.set_axis_off()

plot_restored(0, axs[0,2])
plot_restored(1, axs[1,0])
plot_restored(2, axs[1,1])
plot_restored(3, axs[1,2])

fig_name = f"3_IP_Images_{TIME}"
plt.savefig(f"figures/{fig_name}.png", bbox_inches='tight')
