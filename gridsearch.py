import random
import pandas as pd
from algorithms import genetic_algorithm
from selections import initialize_population_uniform, initialize_population_he
from initializations import tournament_selection, split_rank_selection
from crossover import one_point_crossover, arithmetical_crossover, laplace_crossover
from mutations import gaussian_mutation, polynomial_mutation


results_gridsearch = pd.DataFrame(columns=['initialization', 'selection', 'crossover', 'mutation', 'mutation_rate', 'avg_fitness'])

initialization = [initialize_population_uniform, initialize_population_he]
selection      = [tournament_selection, split_rank_selection]
crossover      = [one_point_crossover, arithmetical_crossover, laplace_crossover]
mutation       = [gaussian_mutation, polynomial_mutation]
mutation_rate  = [0.1, 0.05, 0.01]

total_combinations = len(initialization) * len(selection) * len(crossover) * len(mutation) * len(mutation_rate)
current = 0

for ini in initialization:
    for sel in selection:
        for cro in crossover:
            for mut in mutation:
                for mut_rate in mutation_rate:
                    sum_result = 0
                    for n in range(5):
                        random.seed(n)
                        np.random.seed(n)
                        best_solution, history = genetic_algorithm(
                            initialization   = ini,
                            fitness_function = fitness_function,
                            selection        = sel,
                            crossover        = cro,
                            mutation         = mut,
                            pop_size         = 100,
                            n_iter           = 500,
                            mutation_rate    = mut_rate,
                            total_weights    = total_weights,
                            layer_sizes      = layer_sizes,
                            model            = mlp,
                            X                = X_train,
                            y                = y_train
                        )
                        sum_result += history[-1]

                    new_row = pd.DataFrame([{
                        'initialization': ini.__name__,
                        'selection':      sel.__name__,
                        'crossover':      cro.__name__,
                        'mutation':       mut.__name__,
                        'mutation_rate':  mut_rate,
                        'avg_fitness':    sum_result / 5
                    }])
                    results_gridsearch = pd.concat([results_gridsearch, new_row], ignore_index=True)
                    results_gridsearch.to_csv('ga_gridsearch.csv', index=False)

                    current += 1
                    print(f"[{current}/{total_combinations}]")
