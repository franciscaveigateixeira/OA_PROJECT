def tournament_selection(population, fitnesses, tournament_size=3,maximize=True):

    sample = random.choices([i for i in range(len(population))],k = tournament_size)
    sample_fitnesses = [fitnesses[i] for i in sample]

    winner_index = sample[np.argmax(sample_fitnesses)] if maximize else sample[np.argmin(sample_fitnesses)]

    return population[winner_index]