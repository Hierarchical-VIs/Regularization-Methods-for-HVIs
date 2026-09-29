import scipy as sp
import numpy as np


class AuxProblemSolver:
    # Maximum number of iterations for sub-problem
    MAX_SUB_ITERATIONS = 100

    def __init__(self, q, iterate, final_operator):
        self.q = q
        self.iterate = iterate
        self.final_operator = final_operator

        self.number_iterations = 0

    def solve_to_accuracy(self, initial, eps):
        # Perform sub-iterations to solve auxiliary problem
        z0 = initial

        for _ in range(self.MAX_SUB_ITERATIONS):
            self.number_iterations += 1
            z1 = self.iterate(z0)
            if np.linalg.norm(z1 - z0) <= eps: break
            z0 = z1
        else:
            print(f"WARNING: Maximum number of sub-iterations ({self.MAX_SUB_ITERATIONS}) reached")
        return self.final_operator(z0)
