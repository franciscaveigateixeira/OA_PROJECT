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