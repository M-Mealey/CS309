# config file

# configuration for neural network
nnet = dict(
    n_inputs = 4,
    n_h_neurons = 12,
    n_outputs = 1
)

# configuration for genetic algorithm
ga = dict(
    n_gen       = 300,
    pop_size    = 30,
    prob_xover  = 0.2,
    prob_mut    = 0.2,
    mut_ind     = 0.1
)
