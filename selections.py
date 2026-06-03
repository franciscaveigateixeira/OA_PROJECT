<<<<<<< HEAD
def tournament_selection(population, fitnesses, tournament_size=3):

=======
import random
import numpy as np

def tournament_selection(population, fitnesses, tournament_size=3,maximize=True):
>>>>>>> 91e89e7b0c50f533535d8affaabcff2eac576659

    sample = random.choices([i for i in range(len(population))],k = tournament_size)
    sample_fitnesses = [fitnesses[i] for i in sample]

    winner_index = sample[np.argmax(sample_fitnesses)]

    return population[winner_index]

def split_rank_selection(population, fitnesses):

    pop_size = len(population)
    fitnesses = np.array(fitnesses)

    sorted_indices = np.argsort(fitnesses)

    probabilities = np.zeros(pop_size)

    for idx in range(pop_size):
    i = idx + 1  
        
        orig_idx = sorted_indices[idx]
        
        if i <= pop_size / 2:
            prob = (12 * i) / (5 * pop_size * (pop_size + 2))
        else:
            prob = (28 * i) / (5 * pop_size * (3 * pop_size + 2))
            
        probabilities[orig_idx] = prob

    #garantees that the sum is exactly equal to 1   
    probabilities /= np.sum(probabilities)
    
    #choosing two parents using roullete sample
    chosen_indices = np.random.choice(pop_size, size=2, replace=False, p=probabilities)
    
    return population[chosen_indices[0]], population[chosen_indices[1]]