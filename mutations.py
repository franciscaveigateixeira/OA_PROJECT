
##uniform mutation
def uniform_mutation(individual, mutation_rate, lb=-1.0, ub=1.0):
    """
    Uniform Mutation for real-valued vectors.
 
    Adapted from the binary version (Algorithm 9.6, where bits are negated):
    instead of flipping a bit, each gene is replaced by a new random value
    drawn uniformly from [lb, ub].
 
    For each gene j:
        if U(0,1) <= pm:
            x_j = U(lb, ub)
 
    Parameters
    ----------
    individual    : numpy array — the weight vector to mutate
    mutation_rate : float — probability of mutating each gene (pm)
    lb            : float — lower bound of the weight range (default -1.0)
    ub            : float — upper bound of the weight range (default  1.0)
 
    Returns
    -------
    mutant : numpy array — the (possibly) mutated individual
    """
    mutant = np.copy(individual)
    for j in range(len(mutant)):
        if np.random.uniform(0, 1) <= mutation_rate:
            mutant[j] = np.random.uniform(lb, ub)
    return mutant

##gaussian mutation
def gaussian_mutation(individual, mutation_rate, sigma=0.1):
    """
    Standard Gaussian mutation: each gene is perturbed with probability
    mutation_rate by adding noise ~ N(0, sigma).
    """
    mutant = np.copy(individual)
    for i in range(len(mutant)):
        if np.random.uniform() < mutation_rate:
            mutant[i] += np.random.normal(0, sigma)
    return mutant