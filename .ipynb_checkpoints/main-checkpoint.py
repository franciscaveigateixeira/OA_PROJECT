import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mlp_utils import count_weights,generate_solution,vector_to_weights, fitness_function, initialize_population_uniform


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

    mlp = MLPClassifier(hidden_layer_sizes=HIDDEN_SIZES,   #inputs -> 16 neurons -> 8 neurons -> 1 output
                        activation="relu",
                        max_iter=1,
                        warm_start = False,
                        random_state=69)

    mlp.fit(X_train,y_train) #initialize weights

    #print(mlp.coefs_)  #weights
    #print(mlp.intercepts_)  #biases

    total_weights = count_weights(mlp)
    #print('\nTotal weights:\n',total_weights)

    solution = generate_solution(total_weights)
    print(solution)

    #fitness = fitness_function(solution, mlp, layer_sizes, X_train, y_train)
    #print('Fitness:', fitness)
    print(initialize_population_uniform(2,total_weights))

    for i in range(10):
        solution = generate_solution(total_weights)

        fitness = fitness_function(
            solution,
            mlp,
            layer_sizes,
            X_train,
            y_train
        )

        print(f"Solution {i + 1}: {fitness}")
