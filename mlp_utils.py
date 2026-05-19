import random
import numpy as np

def count_weights(model):
    total = 0

    for matrix in model.coefs_:
        total += matrix.size

    for bias in model.intercepts_:
        total += bias.size

    return total

def generate_solution(total_weights):
    solution = []

    for i in range(total_weights):
        value = random.uniform(-1, 1)
        solution.append(value)

    return np.array(solution)

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

def fitness_function( solution, model, layer_sizes, X, y):

    y = np.array(y)
    coefs, intercepts = vector_to_weights( solution, layer_sizes)

    model.coefs_ = coefs
    model.intercepts_ = intercepts

    predictions = model.predict(X)

    #Accuracy
    correct = 0

    for i in range(len(y)):
        if y[i] == predictions[i]:
            correct += 1

    accuracy = correct / len( y )

    #Recall (Parkinson = 1 )
    true_positive = 0
    false_negative = 0

    for i in range(len(y)):
        #True Parkinson
        if y[i] == 1:
            #correct prediction
            if predictions[1] == 1:
                true_positive += 1
            #predicted healthy instead
            else:
                false_negative += 1

    if (true_positive + false_negative) == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)

    fitness = 0.5 * accuracy + 0.5 * recall

    return fitness

#------ GENETIC OPERATORS ------

def initialize_population_uniform(population_size, total_weights):

    population = []

    for i in range(population_size):

        solution = generate_solution(total_weights)

        population.append(solution)

    return population

#CROSSOVERS

def one_point_crossover(parent1, parent2):
    k = random.randint(1, len(parent1) - 1)

    child1 = np.concatenate([parent1[:k], parent2[k:]])
    child2 = np.concatenate([parent2[:k], parent1[k:]])

    return child1, child2

#dochat
def multi_point_crossover(parent1, parent2, n_points=2):
    size = len(parent1)
    # pick n_points unique cut points (not first or last index)
    points = sorted(random.sample(range(1, size), n_points))

    child1, child2 = parent1.copy(), parent2.copy()

    # swap segments between every other pair of cut points
    swap = False
    prev = 0
    for point in points + [size]:
        if swap:
            child1[prev:point] = parent2[prev:point]
            child2[prev:point] = parent1[prev:point]
        swap = not swap
        prev = point

    return child1, child2


def geometric_semantic_crossover(parent1, parent2):
    # TR: random vector with values in [0, 1]
    TR = np.random.uniform(0, 1, len(parent1))

    # T_XO = (T1 * TR) + ((1 - TR) * T2)
    child = (parent1 * TR) + ((1 - TR) * parent2)

    return child

#oqueedtouatetarfazer
def multi_point_crossover(parent1, parent2, n_points=2):
    size = len(parent1)
    # pick n_points unique cut points (not first or last index)
    points = sorted(random.sample(range(1, size), n_points))

