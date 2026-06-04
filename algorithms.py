import numpy as np
from mlp_utils import fitness_function


#Genetic Algorithm
def genetic_algorithm(initialization, fitness_function, selection, crossover, mutation, pop_size, n_iter, mutation_rate, total_weights, layer_sizes, model, X, y):

    population = initialization(pop_size, total_weights, layer_sizes)
    fitness = [fitness_function(ind, model, layer_sizes, X, y) for ind in population]
    history = []


    winner_index = np.argmax(fitness)
    best_solution, best_fitness = population[winner_index], fitness[winner_index]
    history.append(best_fitness)

    for i in range(n_iter):
        new_population = []

        while len(new_population) < pop_size:
            parent1,parent2 = selection(population, fitness), selection(population, fitness)
            child1,child2 = crossover(parent1,parent2)
            child1,child2 = mutation(child1,mutation_rate), mutation(child2,mutation_rate)
            new_population.extend([child1,child2])

        population = new_population[:pop_size] 
        fitness = [fitness_function(ind, model, layer_sizes, X, y) for ind in population]
        winner_index = np.argmax(fitness)

        #elitism
        if fitness[winner_index] > best_fitness:
            best_solution, best_fitness = population[winner_index], fitness[winner_index]

        history.append(best_fitness)

    return best_solution, history



#Differential Evolution
def differential_evolution(population_size, total_weights, generations, F, CR, model, layer_sizes, X, y, initialization, fitness_function):
    """
    Differential Evolution algorithm
    F: scaling factor (mutation)
    CR: crossover rate
    """

    # Initialize population

    population = initialization(population_size, total_weights, layer_sizes)
        
    fitness = [fitness_function(ind, model, layer_sizes, X, y) for ind in population]
    history = []

    for gen in range(generations):
        new_population = []
    
        for i in range(population_size):
            # Pick 3 random individuals different from i and from each other
            candidates = list(range(population_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)

            # Pick a random starting index
            j = np.random.randint(0, total_weights)

            # Crossover + Mutation
            trial = np.copy(population[i])
            for k in range(total_weights):
                if np.random.uniform() < CR or k == total_weights - 1:
                    # Mutant vector
                    trial[j] = population[c][j] + F * (population[a][j] - population[b][j])
                j = (j + 1) % total_weights

            # Selection — keep the one with better fitness
            trial_fitness = fitness_function(trial, model, layer_sizes, X, y)
            if trial_fitness >= fitness[i]:
                new_population.append(trial)
                fitness[i] = trial_fitness
            else:
                new_population.append(population[i])

        population = new_population
        best_fitness = max(fitness)
        print(f"Generation {gen + 1}: Best F1 = {best_fitness:.4f}")
        history.append(best_fitness)

    best_idx = np.argmax(fitness)
    return population[best_idx], fitness[best_idx], history