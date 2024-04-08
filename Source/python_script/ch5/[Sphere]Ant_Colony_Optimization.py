import numpy as np
from scipy.optimize import differential_evolution

# Define the sphere function
def sphere(x):
    return np.sum(x ** 2)

# Define the bounds for each dimension
bounds = [(-5.2, 5.2)] * 30  # 30-dimensional problem

# Set the seed for replicability
np.random.seed(12345)

# Run the differential evolution algorithm
result = differential_evolution(sphere, bounds, maxiter=10, strategy='best1bin', popsize=64)

# Print the best solution
print("Best solution:", result.x)
print("Best objective value:", result.fun)
