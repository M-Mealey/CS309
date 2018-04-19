import config
import numpy as np
#Comment

num_inputs = config.nnet['n_inputs']
num_hidden_nodes = config.nnet['n_h_neurons']
num_hidden_layers = config.nnet['n_h_layers']
num_outputs = config.nnet['n_outputs']

num_in_weights = (num_inputs+1)*(num_hidden_nodes)
num_h_weights = (num_hidden_nodes)*(num_hidden_nodes+1)*(num_hidden_layers-1)
num_out_weights = (num_hidden_nodes+1)*(num_outputs)
num_weights = num_in_weights + num_h_weights + num_out_weights

class ANN(object):
    def __init__(self,weights):
        self.weights = weights
        self.input_weights = np.array(weights[:num_in_weights]).reshape(num_hidden_nodes,num_inputs+1)
        self.hidden_weights = np.array(weights[num_in_weights:num_in_weights+num_h_weights]).reshape(num_hidden_nodes,num_hidden_nodes+1,num_hidden_layers-1)
        self.output_weights = np.array(weights[num_in_weights+num_h_weights:]).reshape(num_outputs,num_hidden_nodes+1)

    def activation(self,x):
        return 1/(a+np.exp(-x))

    def evaluate(self,inputs):
        inputs = np.array(inputs)
        inputs = np.insert(inputs, 0, 1)
        layer1out = np.dot(self.input_weights,inputs)
        a = [1]
        for i in range(0,num_hidden_nodes):
            a.append( activation(layer1out[i]))
        h_out = np.array(a)

        # evaluate the hidden layers
        for l in range(0,num_hidden_layers-1):
            layer_out = np.dot(self.hidden_weights[l],h_out)
            a = [1]
            for i in range(0,num_hidden_nodes):
                a.append( activation(layer_out[i]) )
            h_out = np.array(a)
        
        # evaluate output layer
        outputs = np.dot(self.output_weights, h_out)
        
        # not really sure what to return here?
        return outputs.tolist()

