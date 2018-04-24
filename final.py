import random
import math
import numpy as np

#import sys
#sys.path.append('home/zhou/FRI/MT1_Spring2018_upload/MT1_Spring2018_upload/MT1/MT1/standalone_game')

#import config
from ANN import ANN

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

num_inputs = 4 #config.nnet['n_inputs']
num_hidden_nodes =2 # config.nnet['n_h_neurons']
num_outputs = 2 #config.nnet['n_outputs']

fileName="data_clean.csv"

lat=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=2)
lon=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=3)
time=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=4)
call_dur=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=5)
call_type=np.genfromtxt(fileName, dtype=int, delimiter=",",skip_header=1, usecols=1)
response_time=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=6)


inputs = np.column_stack((lat,lon,time,call_dur))

outputs = np.column_stack((call_type,response_time))




creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Prepare your individuals below.
# Let's assume that you have a one-hidden layer neural network with 2 hidden nodes:
# You would need to define a list of floating numbers of size: 16 (10+6)
toolbox.register("attr_real", random.uniform,-1,1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, 16) #the x value of cutinfo
popsize=30
toolbox.register("population", tools.initRepeat, list, toolbox.individual, popsize)

def evalMin(individual):
    sum=0
    a=ANN(num_inputs, num_hidden_nodes, num_outputs, individual)
    for i in range(inputs.shape[0]):
    	y=a.evaluate(inputs[i])
    	for j in range(len(y)):
        	sum=sum+(y[j]-outputs[i][j])**2
    return sum,

toolbox.register("evaluate", evalMin)

# Define your selection, crossover and mutation operators below:

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian,mu=0.0 , sigma=0.2 , indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define EA parameters: n_gen, pop_size, prob_xover, prob_mut:
# You can define them in the "config.py" file too.

pop = toolbox.population()
cxpb=0.07
mutpb=0.1
ngen=100
# Create initial population (each individual represents an agent or ANN):
for ind in pop:
    # ind (individual) corresponds to the list of weights
    # ANN class is initialized with ANN parameters and the list of weights
    ann = ANN(num_inputs, num_hidden_nodes, num_outputs, ind)

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(ngen):
   
        
    # Start creating the children (or offspring)
        
    # First, Apply selection:
    offspring = toolbox.select(pop,k=popsize)
        
    # Apply variations (xover and mutation), Ex: algorithms.varAnd(?, ?, ?, ?)
    offspring = algorithms.varAnd(offspring,toolbox,cxpb,mutpb)

    # Repeat the process of fitness evaluation below. You need to put the recently
    # created offspring-ANN's into the game (Line 55-69) and extract their fitness values:
    # One way of implementing elitism is to combine parents and children to give them equal chance to compete:
    # For example: pop[:] = pop + offspring
    # Otherwise you can select the parents of the generation from the offspring population only: pop[:] = offspring
    pop[:]=offspring

	
    for ind in pop:
        # ind (individual) corresponds to the list of weights
        # ANN class is initialized with ANN parameters and the list of weights
        ann = ANN(num_inputs, num_hidden_nodes, num_outputs, ind)

    fitnesses = list(map(toolbox.evaluate, pop))
    sum=0 
    bestIn=0

    for i in range(len(pop)):
#	print "indi",i,"\n","fitness: ", fitnesses[i][0], pop[i]
	   if fitnesses[i][0]==min(fitnesses)[0]:
	   		bestIn=i
	   sum=sum+fitnesses[i][0]
	
    avg=sum/popsize
    print min(fitnesses)[0], "\t","with individual: ", pop[i], "\t", avg
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        