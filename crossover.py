import random
import numpy as np

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
