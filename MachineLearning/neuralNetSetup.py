import numpy as np
import nnfs
import math
from nnfs.datasets import spiral_data
nnfs.init()


#this set of code was used as a basis to understand layers in FNN

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activatoin_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


#the first number is effectively the input shape (i.e. the number of items per 1D list) 
#the second number is effectively the output shape (i.e. the number of items per 1D list after operation)
# layer1 = Layer_Dense(2, 5)
# activation1 = Activatoin_ReLu()

# layer1.forward(X)
# activation1.forward(layer1.forward)
# print(layer1.output)



'''
This was general builder code to get a better understanding of what operations are done within a NN

inputs = [
    [1,2,3,2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]
          

weights = [  
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]



#This starts a new layer by adding a new weight and bias set, we use the outputs from the first layer as our inputs
weights2 = [  
    [0.1, -0.14, 0.5],
    [-0.5, 0.12,-0.33],
    [-0.44, 0.73, -0.13]
]

biases2 = [-1, 2, -0.5]



# the first element passed is how it will be indexed and returned
layer1_output = np.dot(inputs, np.array(weights).T) + biases


#as mentionned, using layer1_outputs as our inputs, and our weights2 and biases2 respectively
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
print(layer1_output)
print(layer2_output)
'''
'''
This code will be done with numpy and other librairies, but here it is as a proof of concept
layer_outputs = [] #output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 #output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
    
    
print(layer_outputs)



inputs = [1,2,3,2.5]

weights = [  
        [0.2,0.8,-0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]

biases = [2, 3, 0.5]



'''