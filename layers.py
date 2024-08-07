import autograd.numpy as np
import numpy.typing as npt
import typing
import math
from autograd import elementwise_grad

class LSTM_Layer:
    def __init__(self, num_units: int, input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.num_units = num_units
        self.num_features = input_shape[1]
        self.num_frames = input_shape[0]

        # intializes weights using glorot initialization
        high = math.sqrt( 6 / (self.num_features + self.num_units))
        low = -high

        # J: number of features
        # K: number of outputs/units
        # T: number of timesteps
        # B: batch size

        # input dimentions -> (B, T, J)

        self.params = [ 
                # 4 input weight matrices, one for each gate
                # input weights -> (K, J)
                np.random.uniform(low, high, ((num_units, self.num_features))),
                np.random.uniform(low, high, ((num_units, self.num_features))),
                np.random.uniform(low, high, ((num_units, self.num_features))),
                np.random.uniform(low, high, ((num_units, self.num_features))),
                # 4 recurrent weight matrices, one for each gate
                # recurrent weights -> (K, K)
                np.random.uniform(low, high, ((num_units, num_units))),
                np.random.uniform(low, high, ((num_units, num_units))),
                np.random.uniform(low, high, ((num_units, num_units))),
                np.random.uniform(low, high, ((num_units, num_units))),
                # 4 bias matrices, one for each gate
                # biases -> (K)
                np.random.uniform(low, high, ((num_units))),
                np.random.uniform(low, high, ((num_units))),
                np.random.uniform(low, high, ((num_units))),
                np.random.uniform(low, high, ((num_units)))
        ]

        # gate outputs -> (B, K)

        # intermediate values needed for backpropagation
        self.caches = []
    
        

    @staticmethod
    def sigmoid(input: npt.NDArray) -> npt.NDArray:
        return np.array([1 / (1 + np.exp(-x)) for x in input])
    
    @staticmethod
    def tanh(input: npt.NDArray) -> npt.NDArray:
        return np.array([  (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  for x in input])
    


    # inputs shape: (batch_size, num timeframes, num features)
    # output shape: (batch_size, num units)
    def __call__(self, inputs: npt.NDArray, training: bool = False) -> npt.NDArray:
        batch_size = inputs.shape[0]

        # intialize state
        hidden_state = np.zeros((batch_size, self.num_units))
        cell_state = np.zeros((batch_size, self.num_units))
        self.caches = []

        # iterating throught the timesteps
        for t in range(inputs.shape[1]):
            # cache value
            if training: c = [cell_state, hidden_state]

            # the input
            timestep = inputs[:, t, :]

            w_f, w_i, w_c, w_o, u_f, u_i, u_c, u_o, b_f, b_i, b_c, b_o = self.params

            # forget gate for remembering only a certain part of the cell state
            f_preactivation = np.dot(timestep, w_f.T) + np.dot(hidden_state, u_f.T) + b_f
            forget = self.sigmoid(f_preactivation)

            # input gate for determining what percent of the potential cell state to add to the cell state
            i_preactivation = np.dot(timestep, w_i.T) + np.dot(hidden_state, u_i.T) + b_i
            input = self.sigmoid(i_preactivation)

            # cell gate for determining the potential cell state value
            c_preactivation = np.dot(timestep, w_c.T) + np.dot(hidden_state, u_c.T) + b_c
            cell = self.tanh(c_preactivation)

            # output gate for getting the new hidden state from the new cell state
            o_preactivation = np.dot(timestep, w_o.T) + np.dot(hidden_state, u_o.T) + b_o
            output = self.sigmoid(o_preactivation)

            # shapes ^ : (batch size, num units)

            # update cell state and hidden state
            cell_state = forget * cell_state + input * cell
            hidden_state = self.tanh(cell_state) * output

            # store intermediate values needed for backward pass
            if training:
                c.extend([input, cell, output, cell_state, inputs, f_preactivation, i_preactivation, c_preactivation, o_preactivation])
                self.caches.append(c)

        
        return hidden_state
    
    
    # returns gradient of the loss w.r.t the weights, biases, and input of the layer
    def backward(self, dl_dh: npt.NDArray) -> tuple:    

        tanh_derivative = elementwise_grad(self.tanh)
        sigmoid_derivative = elementwise_grad(self.sigmoid)
        
        # initialize gradients, the gradients for all the time frames will be added up 
        w_f_grad, w_i_grad, w_c_grad, w_o_grad, u_f_grad, u_i_grad, u_c_grad, u_o_grad, b_f_grad, b_i_grad, b_c_grad, b_o_grad = (np.zeros_like(p) for p in self.params)
        batch_size = self.caches[0][6].shape[0]
        timestep_grads = np.zeros((batch_size, self.num_features))

        # looping through timesteps reversed
        for t in reversed(range(len(self.caches))):
            prev_c, prev_h, i, c, o, new_c, inputs, f_pre, i_pre, c_pre, o_pre = self.caches[t]
            
            # gradient of the loss w.r.t the output gate activation
            dl_do = dl_dh * self.tanh(new_c)

            # gradient of the loss w.r.t the cell state
            dl_dcs = dl_dh * tanh_derivative(new_c) * o

            # gradient of the loss w.r.t the forget gate activation
            dl_df = dl_dh * dl_dcs * prev_c

            # gradient of the loss w.r.t the input gate activation
            dl_di = dl_dh * dl_dcs * c

            # gradient of the loss w.r.t the cell gate activation
            dl_dc = dl_dh * dl_dcs * i

            
            # gradients of the loss w.r.t the preactivations of each gate
            # shape: (batch size, num units)
            forget_preactivation_grad = dl_df * sigmoid_derivative(f_pre)
            input_preactivation_grad = dl_di * sigmoid_derivative(i_pre)
            cell_preactivation_grad = dl_dc * tanh_derivative(c_pre)
            output_preactivation_grad = dl_do * sigmoid_derivative(o_pre)

            # gradients of the loss w.r.t the input xt in each gate
            # shape: (num features)
            dfpre_dxt = np.dot(np.sum(forget_preactivation_grad, axis=0), self.params[0])
            dipre_dxt = np.dot(np.sum(input_preactivation_grad, axis=0), self.params[1])
            dcpre_dxt = np.dot(np.sum(cell_preactivation_grad, axis=0), self.params[2])
            dopre_dxt = np.dot(np.sum(output_preactivation_grad, axis=0), self.params[3])

            # gradient of the loss w.r.t the input xt
            x_t_grad = dfpre_dxt + dipre_dxt + dcpre_dxt + dopre_dxt
            timestep_grads = timestep_grads + x_t_grad


            # input at current timeframe shape: (batch size, num features)
            # previous hidden state shape: (batch size, num units)
            current_input = inputs[:, t, :]

            # the gradient of the loss w.r.t each of the weights and biases
            w_i_grad = w_i_grad + np.dot(input_preactivation_grad.T, current_input)
            w_c_grad = w_c_grad + np.dot(cell_preactivation_grad.T, current_input)
            w_o_grad = w_o_grad + np.dot(output_preactivation_grad.T, current_input)

            u_f_grad = u_f_grad + np.dot(forget_preactivation_grad.T, prev_h)
            u_i_grad = u_i_grad + np.dot(input_preactivation_grad.T, prev_h)
            u_c_grad = u_c_grad + np.dot(cell_preactivation_grad.T, prev_h)
            u_o_grad = u_o_grad + np.dot(output_preactivation_grad.T, prev_h)    

            # the preactivation grads are flattened across batches giving them shape (num units)
            b_f_grad = b_f_grad + np.sum(forget_preactivation_grad, axis=0)
            b_i_grad = b_i_grad + np.sum(input_preactivation_grad, axis=0)
            b_c_grad = b_c_grad + np.sum(cell_preactivation_grad, axis=0)
            b_o_grad = b_o_grad + np.sum(output_preactivation_grad, axis=0)
        
        return (w_f_grad, w_i_grad, w_c_grad, w_o_grad, u_f_grad, u_i_grad, u_c_grad, u_o_grad, b_f_grad, b_i_grad, b_c_grad, b_o_grad), timestep_grads

class Dense_Layer:
    def __init__(self, num_inputs: int, num_neurons: int, activation: typing.Callable) -> None:
        # glorot initialization
        high = math.sqrt( 6 / (num_inputs + num_neurons))
        low = -high
        weights = np.random.uniform(low, high, ((num_neurons, num_inputs)))
        biases = np.random.uniform(low, high, ((num_neurons))) 
        self.params = [weights, biases]

        self.activation = activation

        # intermediate value used for backward pass
        self.preactivation = None
        self.input = None
    

    def __call__(self, inputs: npt.NDArray, training: bool = False) -> typing.Any:
        weights, biases = self.params
        self.input = inputs
        self.preactivation = np.dot(inputs, weights.T) + biases 
        return self.activation(self.preactivation)

    # returns the gradient of the loss w.r.t the weights, biases, and input
    def backward(self, dl_do: npt.NDArray) -> tuple:
        # gradient of the output of the layer w.r.t the preactivation of the layer
        do_dp = elementwise_grad(self.activation)(self.preactivation)
        
        # gradient of the output of the layer w.r.t the weights, biases, and input
        wgrad = np.dot((do_dp * dl_do).T, self.input)
        bgrad = np.sum(do_dp * dl_do, axis=0, keepdims=True)
        igrad = np.dot(do_dp * dl_do, self.params[0])

        return (wgrad, bgrad), igrad