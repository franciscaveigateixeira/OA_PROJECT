import numpy as np
import random
import math


##gaussian mutation
def gaussian_mutation(individual, mutation_rate, sigma=0.1):
    mutant = np.copy(individual).astype(float)
    for i in range(len(mutant)):
        if np.random.uniform() < mutation_rate:
            mutant[i] += np.random.normal(0, sigma)
    return mutant

##polynomial mutation
def polynomial_mutation(individual, mutation_rate, eta_m=20, lb=-1.0, ub=1.0):
    mutant = np.copy(individual).astype(float)
    for i in range(len(mutant)):
        if np.random.uniform() < mutation_rate:
            x = mutant[i]

            # how far is x from each bound, normalised to [0, 1]
            # if x = -1 (at lower bound): delta1 = 0
            # if x =  0 (at center):      delta1 = 0.5
            # if x =  1 (at upper bound): delta1 = 1
            delta1 = (x - lb) / (ub - lb)
            delta2 = (ub - x) / (ub - lb)

            r = np.random.uniform()

            if r <= 0.5:
                # left side: perturbation goes downward (towards lb)
                # (1 - delta1)^(eta_m+1) shrinks when x is near lb,
                # so the step is naturally smaller near the lower bound
                delta_q = (2*r + (1 - 2*r) * (1 - delta1)**(eta_m + 1))**(1/(eta_m + 1)) - 1
            else:
                # right side: perturbation goes upward (towards ub)
                # same logic: step shrinks automatically near the upper bound
                delta_q = 1 - (2*(1 - r) + 2*(r - 0.5) * (1 - delta2)**(eta_m + 1))**(1/(eta_m + 1))

            # scale delta_q back to the actual weight range and apply
            # clip is just a safety net, the formula already respects bounds
            mutant[i] = np.clip(x + delta_q * (ub - lb), lb, ub)
    return mutant

