import numpy as np
# Define the fitness function to minimize
def fitness_function(x):
 return x**2 + 4*x + 4
# Genetic Algorithm parameters
population_size = 50
num_generations = 100
mutation_rate = 0.1
# Initialize the population with random solutions
population = np.random.uniform(-10, 10, size=(population_size,))
# Main Genetic Algorithm loop
for generation in range(num_generations):
 # Evaluate fitness of each individual in the population
 fitness_values = np.array([fitness_function(x) for x in population])
# Select parents for crossover
 parents = np.random.choice(population, size=population_size, p=fitness_values / 
np.sum(fitness_values))
 # Create new population through crossover and mutation
 children = []
 for _ in range(population_size):
  parent1 = np.random.choice(parents)
 parent2 = np.random.choice(parents)
 child = (parent1 + parent2) / 2 # Simple averaging crossover
 if np.random.rand() < mutation_rate:
  child += np.random.normal(scale=0.5)
 children.append(child)
 population = np.array(children)
# Find the best solution
best_solution = population[np.argmin([fitness_function(x) for x in population])]
print("Best Solution:", best_solution)
print("Minimum Value:", fitness_function(best_solution))