import pyswarms as ps
import numpy as np
import matplotlib.pyplot as plt

# Define the sphere function
def sphere(x):
    return np.sum(x ** 2, axis=1)

# Set the parameters
D = 2  # Dimensionality of the problem
options = {
    'c1': 0.5,  # Cognitive parameter
    'c2': 0.3,  # Social parameter
    'w': 0.9,   # Inertia weight
    'k': 5,     # Number of neighbors to consider
    'p': 2      # The Minkowski p-norm to use
}

# Create a PSO instance
optimizer = ps.single.GlobalBestPSO(n_particles=5, dimensions=D, options=options, bounds=(-5.2 * np.ones(D), 5.2 * np.ones(D)))

# Perform optimization
cost, pos = optimizer.optimize(sphere, iters=10)

# Output the fitness at each iteration
for i, cost in enumerate(optimizer.cost_history):
    print(f"Iteration {i+1}: Best Fitness = {cost}")

# Plot the optimization process
plt.figure(figsize=(5, 5))
plt.plot(optimizer.cost_history)
plt.xlabel('Iterations')
plt.ylabel('Best Fitness')
plt.title('Fitness Value at Each Iteration')
plt.show()

# Print the result
print("Best position : ", pos)
print("Best fitness : ", cost)
