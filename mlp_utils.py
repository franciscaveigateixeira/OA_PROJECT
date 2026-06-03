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


#plot

def plot_history(history , title =" Fitness over Iterations "):
    plt.plot(history)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.show()
