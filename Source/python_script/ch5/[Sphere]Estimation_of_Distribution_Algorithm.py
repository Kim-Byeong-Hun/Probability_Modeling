import numpy as np
import matplotlib.pyplot as plt

# Define the sphere function
def sphere(x):
    return np.sum(x ** 2, axis=1)

# Define a simple EDA
class EDA:
    def __init__(self, pop_size, dimensions, max_gen, bounds):
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.max_gen = max_gen
        self.bounds = bounds
        self.best_sol = None
        self.best_eval = np.inf

    def run(self, objective_func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dimensions))
        for gen in range(self.max_gen):
            fitness = objective_func(population)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_eval:
                self.best_eval = fitness[best_idx]
                self.best_sol = population[best_idx]
            # Select the best half of the population
            selected_indices = np.argsort(fitness)[:self.pop_size // 2]
            selected = population[selected_indices]
            # Generate new samples by sampling from the selected population
            new_samples = selected[np.random.choice(selected.shape[0], self.pop_size - selected.shape[0])]
            population = np.vstack((selected, new_samples))
            if gen % 1 == 0:  # Modify if you want less frequent output
                print(f'Generation {gen+1}, best fitness {self.best_eval}')
        return population

# Parameters
D = 2
maxit = 10
LP = 5
np.random.seed(12345)

# Run EDA
eda = EDA(pop_size=LP, dimensions=D, max_gen=maxit, bounds=(-5.2, 5.2))
final_population = eda.run(sphere)

# Print results
print("Best solution:", eda.best_sol)
print("Best objective value:", eda.best_eval)

# Plotting the results (for LP=100 as an example)
LP = 100
eda = EDA(pop_size=LP, dimensions=D, max_gen=maxit, bounds=(-5.2, 5.2))
final_population = eda.run(sphere)

# Create plots for each dimension
for j in range(D):
    plt.figure(figsize=(7, 7))
    for i in range(maxit):
        if i == 0:
            population = np.random.uniform(-5.2, 5.2, (LP, D))
        else:
            # This should ideally be replaced with the actual population at each generation
            population = np.random.uniform(-5.2, 5.2, (LP, D))
        plt.scatter([i+1] * LP, population[:, j], s=19)
    plt.xlabel("Iterations")
    plt.ylabel(f"s_{j+1} value")
    plt.title(f"Evolution of parameter s_{j+1} over generations")
    plt.show()
