import numpy as np
import pyswarms as ps

# Define the Rastrigin function
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

# Negated Rastrigin function for maximization
def rastrigin_neg(x):
    return -rastrigin(x)

# Set the parameters
D = 30  # Dimensions
Pop = 20  # Swarm size
Maxit = 100  # Max iterations
np.random.seed(12345)  # Seed for replicability

# Common options for all PSO runs
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Minimization using Global Best PSO
optimizer1 = ps.single.GlobalBestPSO(n_particles=Pop, dimensions=D, options=options, bounds=(-5.2 * np.ones(D), 5.2 * np.ones(D)))
cost1, pos1 = optimizer1.optimize(rastrigin, iters=Maxit)
print("pso::psoptim (Minimization):")
print(f"Best position: {np.round(pos1, 2)}")
print(f"Best fitness: {np.round(cost1, 2)}")

# Maximization (using negated Rastrigin) using Global Best PSO
optimizer2 = ps.single.GlobalBestPSO(n_particles=Pop, dimensions=D, options=options, bounds=(-5.2 * np.ones(D), 5.2 * np.ones(D)))
cost2, pos2 = optimizer2.optimize(rastrigin_neg, iters=Maxit)
print("psoptim::psoptim (Maximization):")
print(f"Best position: {np.round(pos2, 2)}")
print(f"Best fitness: {-np.round(cost2, 2)}")  # Negating back for the actual fitness

# Another run of Minimization using Global Best PSO for comparison
optimizer3 = ps.single.GlobalBestPSO(n_particles=Pop, dimensions=D, options=options, bounds=(-5.2 * np.ones(D), 5.2 * np.ones(D)))
cost3, pos3 = optimizer3.optimize(rastrigin, iters=Maxit)
print("PSopt (Minimization):")
print(f"Best position: {np.round(pos3, 2)}")
print(f"Best fitness: {np.round(cost3, 2)}")
