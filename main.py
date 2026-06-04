import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mlp_utils import count_weights, fitness_function,plot_history
from algorithms import genetic_algorithm, differential_evolution
from selections import tournament_selection
from crossover import one_point_crossover
from mutations import gaussian_mutation
from initializations import initialize_population_uniform, initialize_population_he


if __name__=='__main__':
    data = pd.read_csv('parkinsons_preprocessed.csv')

    print(data.head())
    print(data.columns)

    X = data.drop("status", axis=1)
    y = data["status"]

    print(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)

    INPUT_SIZE   = X_train.shape[1]   #number of features of the data set(23 - 1 = 22)
    HIDDEN_SIZES = (5, )
    OUTPUT_SIZE  = 1    #target variable
    layer_sizes  = [INPUT_SIZE] + list(HIDDEN_SIZES) + [OUTPUT_SIZE]

    mlp = MLPClassifier(hidden_layer_sizes=HIDDEN_SIZES,    
                        activation="relu",
                        max_iter=1,
                        warm_start = False,
                        random_state=69)

    mlp.fit(X_train,y_train) #initialize weights

    #print(mlp.coefs_)  #weights
    #print(mlp.intercepts_)  #biases

    total_weights = count_weights(mlp)
    #print('\nTotal weights:\n',total_weights)

    gridsearch = pd.read_csv('ga_gridsearch.csv')
    best = gridsearch.loc[gridsearch['avg_fitness'].idxmax()]

    print(gridsearch.sort_values('avg_fitness', ascending=False).head())

# Best params from gridsearch: initialize_population_he, tournament_selection,
# one_point_crossover, gaussian_mutation, mutation_rate=0.1 → avg_fitness=0.9764
histories_ga, test_fitnesses_ga = [], []
for i in range(30):
    best_sol, history = genetic_algorithm(
        initialization=initialize_population_he,
        fitness_function=fitness_function,
        selection=tournament_selection,
        crossover=one_point_crossover,
        mutation=gaussian_mutation,
        pop_size=100,
        n_iter=500,
        mutation_rate=0.1,
        total_weights=total_weights,
        layer_sizes=layer_sizes,
        model=mlp,
        X=X_train, y=y_train
    )
    test_fit = fitness_function(best_sol, mlp, layer_sizes, X_test, y_test)
    test_fitnesses_ga.append(test_fit)
    histories_ga.append(history)
    print(f'GA: {i+1}/30 runs completed')

mean_history_ga = np.mean(histories_ga, axis=0)
plot_history(mean_history_ga, title="GA - Average Fitness over Generations (30 runs)")
print(f'GA - Mean Test F1: {np.mean(test_fitnesses_ga):.4f} ± {np.std(test_fitnesses_ga):.4f}')

# DE with uniform initialization — F=0.5, CR=0.9 (Storn & Price 1997)
de_histories_uniform, test_fitnesses_de_uniform = [], []
for i in range(30):
    best_sol, best_fit, history = differential_evolution(
        population_size=100,
        total_weights=total_weights,
        generations=500,
        F=0.5,
        CR=0.9,
        model=mlp,
        layer_sizes=layer_sizes,
        X=X_train, y=y_train,
        initialization=initialize_population_uniform,
        fitness_function=fitness_function
    )
    test_fit = fitness_function(best_sol, mlp, layer_sizes, X_test, y_test)
    test_fitnesses_de_uniform.append(test_fit)
    de_histories_uniform.append(history)
    print(f'DE Uniform: {i+1}/30 runs completed')

# DE with He initialization
de_histories_he, test_fitnesses_de_he = [], []
for i in range(30):
    best_sol, best_fit, history = differential_evolution(
        population_size=100,
        total_weights=total_weights,
        generations=500,
        F=0.5,
        CR=0.9,
        model=mlp,
        layer_sizes=layer_sizes,
        X=X_train, y=y_train,
        initialization=initialize_population_he,
        fitness_function=fitness_function
    )
    test_fit = fitness_function(best_sol, mlp, layer_sizes, X_test, y_test)
    test_fitnesses_de_he.append(test_fit)
    de_histories_he.append(history)
    print(f'DE He: {i+1}/30 runs completed')

mean_de_uniform = np.mean(de_histories_uniform, axis=0)
mean_de_he = np.mean(de_histories_he, axis=0)
plot_history(mean_de_uniform, mean_de_he,
             labels=["Uniform Initialization", "He Initialization"],
             title="DE - Average Fitness over Generations (30 runs)")
print(f'DE Uniform - Mean Test F1: {np.mean(test_fitnesses_de_uniform):.4f} ± {np.std(test_fitnesses_de_uniform):.4f}')
print(f'DE He     - Mean Test F1: {np.mean(test_fitnesses_de_he):.4f} ± {np.std(test_fitnesses_de_he):.4f}')

# GA vs DE comparison (best DE initialization)
best_de_histories = mean_de_he if np.mean(test_fitnesses_de_he) >= np.mean(test_fitnesses_de_uniform) else mean_de_uniform
plot_history(mean_history_ga, best_de_histories,
             labels=["GA", "DE"],
             title="GA vs DE - Average Fitness over Generations (30 runs)")