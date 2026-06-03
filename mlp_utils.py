import random
import numpy as np
import matplotlib.pyplot as plt

def count_weights(model):
    total = 0

    for matrix in model.coefs_:
        total += matrix.size

    for bias in model.intercepts_:
        total += bias.size

    return total

def generate_solution(total_weights):
    """
    Generates a random weight vector of size total_weights with values in [-1, 1]
    """
    return np.random.uniform(-1, 1, total_weights)

def vector_to_weights(vector, layer_sizes):
    """
    Reshape a numpy array into the coefs and intercepts again
    """
    coefs, intercepts = [],[]
    idx = 0

    for i in range(len(layer_sizes) - 1):
        rows, cols = layer_sizes[i], layer_sizes[i + 1]

        w = vector[idx : idx + rows * cols].reshape(rows, cols)
        coefs.append(w)
        idx += rows * cols

        b = vector[idx : idx + cols]
        intercepts.append(b)
        idx += cols

    return coefs, intercepts


#------ FITNESS FUNCTION ------

def fitness_function(solution, model, layer_sizes, X, y):
    """
    Compute the F1 score
    """
    y = np.array(y)

    coefs, intercepts = vector_to_weights(solution, layer_sizes)

    model.coefs_ = coefs
    model.intercepts_ = intercepts

    predictions = model.predict(X)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    # Count TP, FP, FN 
    for i in range(len(y)):
        if predictions[i] == 1 and y[i] == 1:
            # True Positive: correctly predicted Parkinson
            true_positive += 1
        elif predictions[i] == 1 and y[i] == 0:
            # False Positive: predicted Parkinson, but actually healthy
            false_positive += 1
        elif predictions[i] == 0 and y[i] == 1:
            # False Negative: predicted healthy, but actually Parkinson
            false_negative += 1

    # Calculate precision: TP / (TP + FP)
    if (true_positive + false_positive) == 0:
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)

    # Calculate recall: TP / (TP + FN)
    if (true_positive + false_negative) == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)

    # Calculate F1-score: 2 * (precision * recall) / (precision + recall)
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    # Return the F1-score as fitness
    return f1_score

#------ GENETIC OPERATORS ------

def initialize_population_uniform(population_size, total_weights, layer_sizes, generate = generate_solution):

    population = []

    for i in range(population_size):

        solution = generate(total_weights)

        population.append(solution)

    return population

def initialize_population_he(population_size, total_weights, layer_sizes):
    population = []

    for _ in range(population_size):
        vector = np.array([])
        for i in range(len(layer_sizes) - 1):
            n_inlayer = layer_sizes[i] #number of neurons in the current layer
            std = np.sqrt(2 / n_inlayer)
            
            #weights are initialized with normal distribution
            w = np.random.normal(0, std, n_inlayer * layer_sizes[i + 1])
            #bias are initialized with zeros
            b = np.zeros(layer_sizes[i + 1])
            
            vector = np.concatenate([vector, w, b])

        population.append(vector)

    return population


def one_point_crossover(parent1, parent2):
    k = random.randint(1, len(parent1) - 1)

    child1 = np.concatenate([parent1[:k], parent2[k:]])
    child2 = np.concatenate([parent2[:k], parent1[k:]])

    return child1, child2


#we use arithmetical instead of geometric,because weights could
#be negative,and since geometric uses exponents it would not work

def arithmetical_crossover(parent1, parent2):
    # alpha: random vector with values in [0, 1]
    alpha = np.random.uniform(0, 1, len(parent1))


    child1 = (parent1 * alpha) + ((1 - alpha) * parent2)
    child2 = (parent2 * alpha) + ((1 - alpha) * parent1)

    return child1, child2

#The arithmetical crossover with alpha = 0.5 is a clear example
#of an exploitative crossover operator. Contrarily, this
#operator will show exploration for alpha > 1 or alpha < 0

#othercrossover:
#-Blend Crossover (BLX-α)

#Laplace crossover

def laplace_crossover(parent1, parent2, a, b):
    u = np.random.uniform(0, 1, len(parent1))
   # 2. Calcular o vetor beta seguindo estritamente a Equação (3) da imagem
    # np.where(condição, valor_se_verdadeiro, valor_se_falso)
    beta = np.where(u <= 0.5, 
                    a - b * np.log(u), 
                    a + b * np.log(u))
    child1=parent1+beta*np.abs(parent1-parent2)

    child2=parent2+beta*np.abs(parent1-parent2)
   
    return child1, child2


#plot

def plot_history(history , title =" Fitness over Iterations "):
    plt.plot(history)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.show()
