import numpy as np
import math
from autograd import grad


# function: data handler

class Neuron:
    def __init__(self, num_weights):
        # initialize weights from normal distribution (H.E. intiailization)
        mean = 0
        variance = 2 / num_weights
        # square root of variance gives standard deviation
        self.weights = np.random.normal(mean, math.sqrt(variance), num_weights)
        # initialize bias as 0
        self.bias = 0
    
    def __call__(self, inputs):
       self.inputs = inputs
       weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
       return weighted_sum
    
    
    def backward(self):
        gradients = []
        h = 0.001
        for i, w in enumerate(self.weights):
            ws = self.weights[:]
            ws[i] += h
            weight_sum = sum(w * x for w, x in zip(ws, self.inputs)) + self.bias
            pr_weighted_sum = sum(w * x for w, x in zip(self.weights, self.inputs)) + self.bias
            gradients.append((weight_sum - pr_weighted_sum) / h)
        
        # one dimensional array - the gradients of each weight
        return gradients
        
    

class Layer:
    def __init__(self, activation, num_neurons, num_previous_neurons):
        self.neurons = [Neuron(num_previous_neurons) for i in range(num_neurons)]
        self.activation = activation
        self.activation_derivative = grad(activation)
        self.outputs = []
        self.pre_activation = []
    
    def __call__(self, inputs):
        weighted_sums = [n(inputs) for n in self.neurons]
        self.pre_activation = weighted_sums
        out = [self.activation(s) for s in weighted_sums]
        self.outputs = out
        return out
    

    def backward(self):
        # gradient of the layers output with respect to the output before the activation function
        output_preactivation_g = self.activation_derivative(self.pre_activation)[0]
        
        
        # gradient of the weighted sum with respect to each of the weights
        # two dimensional array - the gradients of each weight of each neuron in the layer
        weight_sum_weight_g = [n.backward() for n in self.neurons]

        o = [[output_preactivation_g * w for w in nw] for nw in weight_sum_weight_g]
        return o

# class: Optimizer ??

class MultilayerPerceptron:
    # layer sizes: sizes of the non-input layers
    # activations: activation fucntions for non-input layers
    def __init__(self, layer_sizes, num_inputs, activations, loss_function):
        # non-input layers
        self.layers = []
        self.outputs = []
        self.loss_function = loss_function
        self.loss_derivative = grad(loss_function)
        for i, num_neurons in enumerate(layer_sizes):
            # make previous layer size the number of inputs if this is the first hidden layer
            previous_layer_size = num_inputs if i == 0 else len(self.layers[-1].neurons)
            self.layers.append(Layer(activations[i], num_neurons, previous_layer_size))

    def __call__(self, inputs):
        # manually perfmorm forward pass for input layer
        next_inputs = []
        for n in self.layers[0].neurons:
            next_inputs.append(n(inputs))

        # loop through layers to continue forward pass
        for l in self.layers:
            # next layer inputs = current layer outputs
            next_inputs = l(next_inputs)

        # after the loop, next inputs will be the outputs of the output layer
        self.outputs = next_inputs
        return next_inputs


    def backward(self, actual_outputs):
        loss = self.loss_function(self.outputs, actual_outputs)

        # gradient of the loss function with respect to the outputs
        loss_output_g = self.loss_derivative(self.outputs, actual_outputs)[0]
        layer_grads = self.layers[-1].backward()
        o = []
        for i in layer_grads:
            t = []
            for j in i:
                t.append(j * loss_output_g)
            o.append(t)
        print(o)




    # def train(self, x, y):
    #     for input, label in zip(x, y):
    #         out = self.forward(input)
    #         loss = self.loss_function(out, label)

            




# mock dataset
dataset = {
    'X': [[i] for i in range(100)],
    'Y': [[-7.55], [1.7], [0.46], [12.87], [3.23], [9.0], [13.76], [5.5], [8.81], [10.42], [9.54], [15.16], [2.82], [15.1], [9.01], [24.64], [7.73], [10.26], [21.15], [25.27], [22.38], [13.04], [26.19], [16.25], [29.1], [20.3], [28.7], [35.27], [26.09], [35.0], [39.65], [31.43], [37.67], [42.35], [35.52], [32.1], [32.9], [37.37], [36.36], [43.16], [43.46], [36.63], [51.89], [37.38], [47.2], [43.84], [50.8], [53.6], [50.64], [39.8], [48.79], [52.59], [56.62], [56.07], [51.17], [48.43], [62.9], [65.31], [67.9], [60.12], [52.94], [64.39], [60.18], [55.91], [69.97], [62.98], [59.4], [59.85], [76.19], [77.52], [65.18], [78.6], [73.91], [67.58], [77.51], [68.71], [82.41], [79.29], [80.03], [72.38], [75.47], [71.64], [85.3], [87.85], [84.11], [82.8], [90.73], [88.83], [92.51], [91.63], [82.8], [81.69], [89.7], [85.67], [90.61], [99.3], [96.85], [100.66], [107.01], [102.59]]
}

# linear activation function
linear_activation = lambda x: x

# MSE loss function
def mean_squared_error_loss(inputs, actuals):
    mse = 0
    for input, actual in zip(inputs, actuals):
        mse += (input - actual) ** 2
    return mse

mlp = MultilayerPerceptron(layer_sizes=[4, 5, 1], num_inputs=1, activations=[linear_activation for i in range(3)], loss_function=mean_squared_error_loss)
mlp([1])
mlp.backward([-1])
# mlp.train(dataset['X'], dataset['Y'])