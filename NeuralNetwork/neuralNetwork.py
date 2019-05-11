# -----------------------------------------------------------------------------
# Neural network class itself.
# Source code is a property of https://github.com/makeyourownneuralnetwork.
# -----------------------------------------------------------------------------
import numpy as np
import scipy.special as sp

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, wih = None, who = None):
        # set number of nodes inside input, hidden and output layers.
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate.
        self.lr = learningrate

        self.lastTrainEpochs = 0

        # matrices of weight coefficients.
        # wih - between input and hidden layers,
        # who - between hidden and output layers.
        # uncomment this to use default weights initialization.
        """
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        """
        if wih is None:
            self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        else:
            self.wih = wih
        
        if who is None:
            self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        else:
            self.who = who
        self.activation_function = lambda x: sp.expit(x)
        self.inverse_activation_function = lambda x: sp.logit(x)

    # neural network training.
    def train(self, inputs_list, targets_list):
        inputs_array = np.array(inputs_list, ndmin=2)
        targets = np.array(targets_list, ndmin=2).T

        # get here hidden and final outputs.
        query_outputs = self.query(inputs_list)
        hidden_outputs = query_outputs[0]
        final_outputs = query_outputs[1]

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # update weight coefficients of links between hidden and output layers. 
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
            np.transpose(hidden_outputs))

        # update weight coefficients of links between input and hidden layers. 
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            inputs_array)

    # neural network querying.
    def query(self, inputs_list):
        # convert inputs list to two-dimensional array.
        inputs = np.array(inputs_list, ndmin=2).T

        # calc input signals for hidden layer.
        hidden_inputs = np.dot(self.wih, inputs)
        # calc output signals for hidden layer.
        hidden_outputs = self.activation_function(hidden_inputs)

        # calc input signals for output layer.
        final_inputs = np.dot(self.who, hidden_outputs)
        # calc output signals for output layer.
        final_outputs = self.activation_function(final_inputs)

        # return tuple of hidden and final outputs.
        return (hidden_outputs, final_outputs)

    # neural network backquarying.
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array.
        final_outputs = np.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer.
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer.
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 - 0.99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer.
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer.
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 - 0.99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs