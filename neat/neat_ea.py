import numpy as np
import random
import neat

fileName = "training.csv"


lat=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=0)
lon=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=1)
time=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=2)
call_dur=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=3)
call_type=np.genfromtxt(fileName, dtype=int, delimiter=",",skip_header=1, usecols=4)
response_time=np.genfromtxt(fileName, dtype=float, delimiter=",",skip_header=1, usecols=5)


inputs = np.column_stack((lat,lon,time,call_dur))
print inputs
outputs = np.column_stack((call_type,response_time))
print outputs



# evaluate function
def evaluate(genomes,config):
    for genome_id, genome in genomes:
        error = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        for i,o in zip(inputs, outputs):
            nnet_out = net.activate(i)
            error = error + (nnet_out[0]-o[0])**2 + (nnet_out[1]-o[1])**2
        genome.fitness = -error

config_file = "config"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,config_file)

# create population
pop = neat.Population(config)

# add a stats reporter?
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.Checkpointer(5))

# run for 150 generations
winner = pop.run(evaluate, 100)

print('\nBest genome:\n{!s}'.format(winner))
