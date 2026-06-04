import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mlp_utils import count_weights, fitness_function,plot_history
from algorithms import genetic_algorithm, differential_evolution
from selections import tournament_selection
from crossover import one_point_crossover
from mutations import gaussian_mutation, uniform_mutation
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
    histories = []
    for _ in range(30):
        _,history = genetic_algorithm(
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
        print(f'GA: {_+1}/30 runs completed')
        histories.append(history)

    mean_history_ga = np.mean(histories, axis=0)
    plot_history(mean_history_ga, title="GA - Average Fitness over Generations (30 runs)")

    # DE with uniform initialization
    de_histories_uniform = []
    for _ in range(30):
        _, __, history = differential_evolution(
            population_size=100,
            total_weights=total_weights,
            generations=500,
            F=0.8,
            CR=0.9,
            model=mlp,
            layer_sizes=layer_sizes,
            X=X_train, y=y_train,
            initialization=initialize_population_uniform,
            fitness_function=fitness_function
        )
        print(f'DE Uniform: {_+1}/30 runs completed')
        de_histories_uniform.append(history)

    # DE with He initialization
    de_histories_he = []
    for _ in range(30):
        _, __, history = differential_evolution(
            population_size=100,
            total_weights=total_weights,
            generations=500,
            F=0.8,
            CR=0.9,
            model=mlp,
            layer_sizes=layer_sizes,
            X=X_train, y=y_train,
            initialization=initialize_population_he,
            fitness_function=fitness_function
        )
        print(f'DE He: {_+1}/30 runs completed')
        de_histories_he.append(history)

    mean_de_uniform = np.mean(de_histories_uniform, axis=0)
    mean_de_he = np.mean(de_histories_he, axis=0)
    plot_history(mean_de_uniform, mean_de_he,
                labels=["Uniform Initialization", "He Initialization"],
                title="DE - Average Fitness over Generations (30 runs)")