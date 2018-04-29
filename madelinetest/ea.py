import numpy as np
import pandas as pd
import config as c
import random



from ann import ANN
from deap import base,creator,tools,algorithms

fileName = "training.csv"


lat=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=0)
lon=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=1)
time=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=2)
call_dur=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=3)
call_type=np.genfromtxt(fileName, dtype=int, delimiter=",",skip_header=1, usecols=4)
response_time=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=5)


inputs = np.column_stack((lat,lon,time,call_dur))
# print inputs
outputs = np.column_stack((call_type,response_time))
# print outputs

num_inputs = c.nnet['n_inputs']
num_h_nodes = c.nnet['n_h_neurons']
num_h_layers = c.nnet['n_h_layers']
num_outputs = c.nnet['n_outputs']

num_in_weights = (num_inputs+1)*(num_h_nodes)
num_h_weights = (num_h_nodes)*(num_h_nodes+1)*(num_h_layers-1)
num_out_weights = (num_h_nodes+1)*(num_outputs)
num_weights = num_in_weights + num_h_weights + num_out_weights

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Individuals
toolbox.register("attribute", random.uniform, -20, 20)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=num_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# evaluate function
def evaluate(ind):
    ann = ANN(ind)
    error = 0.0;
    for i in range(0,inputs.shape[0]):
        out=ann.evaluate(inputs[i])
        error = error + ((out[0]-outputs[i][0])**2) + ((out[1]-outputs[i][1])**2)
    return error,

toolbox.register("evaluate",evaluate)


# ea parameters: get from config file
n_gen = c.ga['n_gen']
pop_size = c.ga['pop_size']
prob_xover = c.ga['prob_xover']
prob_mut = c.ga['prob_mut']


# pick ea functions
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma = 5.0, indpb=1)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=pop_size)

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop,fitnesses):
    ind.fitness.values = fit

output = ""

# generation loop
for g in range(1, n_gen):
    top=tools.selBest(pop,1)[0]
    output += "gen " + str(g-1) + " best fit: "
    output += str(top.fitness.values[0])
    output += "\n"

    # Selection:
    offspring = map(toolbox.clone, toolbox.select(pop, len(pop)))

    # Apply variations (xover and mutation), Ex: algorithms.varAnd(?, ?, ?, ?)
    offspring = algorithms.varAnd(offspring, toolbox, prob_xover, prob_mut)
    pop =  tools.selBest(pop, 2) + offspring

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop,fitnesses):
        ind.fitness.values = fit
    # output += " fitnesses: " + str(fitnesses) + "\n"



top=tools.selBest(pop,1)[0]
output += "final best fit: "
output += str(top.fitness.values[0])
for ind in pop:
    if top.fitness.values[0] == ind.fitness.values:
        output += str(ind)
        break

with open("output.txt", "w") as file:
    file.write(output)
