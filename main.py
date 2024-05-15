import numpy as np
from autograd import elementwise_grad



# function: data handler
        
    

class Layer:
    def __init__(self, activation, num_neurons, num_previous_neurons):
        # initialize weights
        weights = np.random.normal(0, 2 / num_previous_neurons, num_neurons * num_previous_neurons)

        # weight structure: 
        # each row corresponds to the weights of a neuron
        # each column corresponds to the weights of neurons connected to a neuron in the previous layer
        self.weights = np.reshape(weights, (num_neurons, num_previous_neurons))

        # initialize biases as 0, one for each neuron
        self.biases = np.zeros((num_neurons, 1))

        # initialize gradients of each weight as 0
        self.gradients = np.zeros((num_neurons, num_previous_neurons))

        self.activation = activation

        self.activation_derivatives = np.zeros((num_neurons, 1))

        self.neuron_grads = np.zeros((num_neurons, 1))

        self.inputs = np.zeros((num_neurons, 1))

        
        # data needed for backpropagation
    
    def __call__(self, inputs):
        self.inputs = np.array(inputs)

        # multiply the weights that connect to each previous neuron with the neurons value
        weighted_sum = np.dot(self.weights, inputs) + self.biases

        self.preactivations = weighted_sum

        # store the derivatives of the activation function with respect to the weighted sum
        # weighted sum is in n, 1 shape
        self.activation_derivatives = elementwise_grad(self.activation)(weighted_sum)

        # run outputs through activation function
        # weighted sum is in n, 1 shape
        activations = self.activation(weighted_sum)

    
        return activations
    


# class: Optimizer ??

class MultilayerPerceptron:
    # layer sizes: sizes of the non-input layers
    # activations: activation fucntions for non-input layers
    def __init__(self, num_inputs, loss_function):
        # non-input layers
        self.layers = []
        self.loss_function = loss_function
        
        self.num_inputs = num_inputs

    def add_layer(self, activation, num_neurons):
        previous_layer_size = self.num_inputs if len(self.layers) == 0 else len(self.layers[-1].weights)
        self.layers.append(Layer(activation, num_neurons, previous_layer_size))

    def __call__(self, inputs):
        # manually perfmorm forward pass for input layer
        # next_inputs = []
        # for n in self.layers[0].neurons:
        #     next_inputs.append(n(inputs))
        next_inputs = inputs

        # loop through layers to continue forward pass
        for l in self.layers:
            # next layer inputs = current layer outputs
            next_inputs = l(next_inputs)

        # after the loop, next inputs will be the outputs of the output layer
        return next_inputs


    def backward(self, outputs, actual_outputs):
        # backprpop to output layer manually
        # each layers input/output shape: (n, 1)

        # grad of the loss func w.r.t the outputs of the last layer before activation function
        output_n_grad = elementwise_grad(self.loss_function)(outputs, actual_outputs) * self.layers[-1].activation_derivatives

        # gradients of the weighted sums
        self.layers[-1].neruon_grads = output_n_grad

        # gradients of the weights
        self.layers[-1].gradients = np.dot(output_n_grad, self.layers[-1].inputs.T)


        # loop through layers backwards
        layer_ind = len(self.layers) - 1
        # stopping loop at 2nd layer (1st hidden layer) because input layer doesn't have weights and biases to optimize
        while layer_ind > 0:
            
            # gradient of the preactivations of the current layer w.r.t the preactivations in the previous layer  
            # x = np.dot(self.layers[layer_ind].weights, self.layers[layer_ind - 1].activation_derivatives)
            x = self.layers[layer_ind].weights * self.layers[layer_ind - 1].activation_derivatives.T
            # output -> matrix with shape (num_current_neurons, num_previous neurons)


            # previous layer neuron gradients = x * current layer neuron gradients
            # output -> matrix with shape (num_prev_neruons, 1)
            n_grads = np.dot(x.T , self.layers[layer_ind].neuron_grads)

            
            self.layers[-1].neuron_gradients = n_grads

            # previous layer weight gradients = previous layer inputs * previous layer neuron gradients
            self.layers[layer_ind - 1].gradients = np.dot(n_grads, self.layers[layer_ind - 1].inputs.T)
        
            layer_ind -= 1
        
    def gradient_descent(self, learning_rate):
        for l in self.layers:
            l.weights -= (l.gradients * learning_rate)
                


    def train(self, x, y, learning_rate, epochs):
        for _ in range(epochs):
            x_train, y_train = x[: round(len(x) * 0.70)], y[: round(len(y) * 0.70)]
            x_test, y_test = x[round(len(x) * 0.70):], y[round(len(y) * 0.70):]
            
            for input, output in zip(x_train, y_train):
                pred = self.__call__(input)
                self.backward(pred, output)
                self.gradient_descent(learning_rate=learning_rate)
            
            loss = []
            for input, output in zip(x_test, y_test):
                pred = self.__call__(input)
                loss.append(self.loss_function(pred, output))

            print('Loss:' + str(np.average(np.array(loss))))


        
     




# mock dataset
dataset = {
    'X': [[[i]] for i in range(100)],
    'Y': [[[-7.55]], [[1.7]], [[0.46]], [[12.87]], [[3.23]], [[9.0]], [[13.76]], [[5.5]], [[8.81]], [[10.42]], [[9.54]], [[15.16]], [[2.82]], [[15.1]], [[9.01]], [[24.64]], [[7.73]], [[10.26]], [[21.15]], [[25.27]], [[22.38]], [[13.04]], [[26.19]], [[16.25]], [[29.1]], [[20.3]], [[28.7]], [[35.27]], [[26.09]], [[35.0]], [[39.65]], [[31.43]], [[37.67]], [[42.35]], [[35.52]], [[32.1]], [[32.9]], [[37.37]], [[36.36]], [[43.16]], [[43.46]], [[36.63]], [[51.89]], [[37.38]], [[47.2]], [[43.84]], [[50.8]], [[53.6]], [[50.64]], [[39.8]], [[48.79]], [[52.59]], [[56.62]], [[56.07]], [[51.17]], [[48.43]], [[62.9]], [[65.31]], [[67.9]], [[60.12]], [[52.94]], [[64.39]], [[60.18]], [[55.91]], [[69.97]], [[62.98]], [[59.4]], [[59.85]], [[76.19]], [[77.52]], [[65.18]], [[78.6]], [[73.91]], [[67.58]], [[77.51]], [[68.71]], [[82.41]], [[79.29]], [[80.03]], [[72.38]], [[75.47]], [[71.64]], [[85.3]], [[87.85]], [[84.11]], [[82.8]], [[90.73]], [[88.83]], [[92.51]], [[91.63]], [[82.8]], [[81.69]], [[89.7]], [[85.67]], [[90.61]], [[99.3]], [[96.85]], [[100.66]], [[107.01]], [[102.59]]]
}


# linear activation function
# x: matrix of shape (n, 1)
def linear_activation(x):
    return x


# MSE loss function
# inputs are of shape (n, 1)
def mean_squared_error_loss(inputs, actuals):
    mse = (inputs - actuals) ** 2
    # mean along the rows outputs an array of shape (1, )
    mse = mse.mean(0)
    return mse[0]

mlp = MultilayerPerceptron(num_inputs=1, loss_function=mean_squared_error_loss)
mlp.add_layer(activation=linear_activation, num_neurons=4)
mlp.add_layer(activation=linear_activation, num_neurons=5)
mlp.add_layer(activation=linear_activation, num_neurons=1)


pred = mlp([[-1]])
mlp.train(dataset['X'], dataset['Y'], learning_rate=0.01, epochs=100)