# config file

# configuration for neural network
nnet = dict(
    n_inputs = 4,
    n_h_neurons = 3,
    n_h_layers = 3,
    n_outputs = 2
)

# configuration for genetic algorithm
ga = dict(
    n_gen       = 50,
    pop_size    = 10,
    prob_xover  = 0.3,
    prob_mut    = 0.3
)

# some metadata for cleaning
params = dict(
    lat_min     = 37.70862411,
    lat_max     = 37.83166623,
    lon_min     = -122.5136484,
    lon_max     = -122.3651383
)
