import autograd.numpy as np
import numpy.typing as npt
import typing
import math
from autograd import elementwise_grad
import pandas as pd



def create_lstm_layer(num_units, input_shape):
    # initialize matrices for parameters for forward and backward pass
    def initialize_params():
        num_features = input_shape[1]
        # J: number of features
        # K: number of outputs/units
        # T: number of timesteps
        # B: batch size

        # input dimentions -> (B, T, J)
        
        # gate outputs -> (B, K)

        weight_shapes = [
            # input weights -> (K, J)
            (num_units, num_features), (num_units, num_features), (num_units, num_features), (num_units, num_features),
            # recurrent weights -> (K, K)
            (num_units, num_units), (num_units, num_units), (num_units, num_units), (num_units, num_units),
            # biases -> (K)
            (num_units), (num_units), (num_units), (num_units)
        ]
        
        high = math.sqrt( 6 / (num_features + num_units))
        low = -high
        return [np.random.uniform(low, high, (shape)) for shape in weight_shapes]

    sigmoid = lambda input: np.array([1 / (1 + np.exp(-x)) for x in input])
    tanh = lambda input: np.array([  (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  for x in input])

    # assumes batch processing
    def lstm_layer(input, params, **kwargs):
        # previous hidden state and previous cell state
        prev_hidden_state = kwargs.get('previous_hidden_state', np.zeros((input.shape[1], num_units)))
        prev_cell_state = kwargs.get('previous_cell_state', np.zeros((input.shape[1], num_units)))

        w_f, w_i, w_c, w_o, u_f, u_i, u_c, u_o, b_f, b_i, b_c, b_o = params

        # take first timestep
        timestep = input[:, 0, :]

        # forget gate for remembering only a certain part of the cell state
        f_preactivation = np.dot(timestep, w_f.T) + np.dot(prev_hidden_state, u_f.T) + b_f
        forget = sigmoid(f_preactivation)

        # input gate for determining what percent of the potential cell state to add to the cell state
        i_preactivation = np.dot(timestep, w_i.T) + np.dot(prev_hidden_state, u_i.T) + b_i
        input = sigmoid(i_preactivation)

        # cell gate for determining the potential cell state value
        c_preactivation = np.dot(timestep, w_c.T) + np.dot(prev_hidden_state, u_c.T) + b_c
        cell = tanh(c_preactivation)

        # output gate for getting the new hidden state from the new cell state
        o_preactivation = np.dot(timestep, w_o.T) + np.dot(prev_hidden_state, u_o.T) + b_o
        output = sigmoid(o_preactivation)

        # shapes ^ : (batch size, num units)


        # update cell state and hidden state
        cell_state = forget * prev_cell_state + input * cell
        hidden_state = tanh(cell_state) * output

        # returns cache values needed for backpropagation
        cache = lambda: [prev_hidden_state, prev_cell_state, input, cell, output, cell_state, timestep, f_preactivation, i_preactivation, c_preactivation, o_preactivation]

        # if this was the last timestep
        if input.shape[1] == 1:
            return hidden_state, [cache]
        # if not continue the recursion
        next_timestep_output = lstm_layer(input[:, 1:, :], previous_hidden_state=hidden_state, previous_cell_state=cell_state)
        out = next_timestep_output[0], cache + next_timestep_output[1]
        return out
    
    def backward(dl_dh, params, cache):
        tanh_derivative = elementwise_grad(tanh)
        sigmoid_derivative = elementwise_grad(sigmoid)
        
        caches = [c() for c in cache]

        # all shapes below: (batch_size, num_units)

        # each cache value can be indexed by the timeframe
        prev_h, prev_c, i, c, o, cell_state, x, f_pre, i_pre, c_pre, o_pre = list(map(lambda c: [t[c] for t in caches], range(len(caches[0]))))

        # gradient of the loss w.r.t the output gate activation for each timeframe
        dl_do = list(map(lambda c: dl_dh * tanh(c), cell_state))

        # gradient of the loss w.r.t the cell state for each timeframe
        dl_dcs = list(map(lambda x: dl_dh * tanh_derivative(x[0]) * x[1], zip(cell_state, o)))

        # gradient of the loss w.r.t the forget gate activation for each timeframe
        dl_df = list(map(lambda x: dl_dh * x[0] * x[1], zip(dl_dcs, prev_c)))

        # gradient of the loss w.r.t the input gate activation for each timeframe
        dl_di = list(map(lambda x: dl_dh * x[0] * x[1], zip(dl_dcs, c)))

        # gradient of the loss w.r.t the cell gate activation for each timeframe
        dl_dc = list(map(lambda x: dl_dh * x[0] * x[1], zip(dl_dcs, i)))

        # gradient of the loss w.r.t the preactivation of each gate for each timestep
        forget_preactivation_grad = list(map(lambda x: x[0] * sigmoid_derivative(x[1]), zip(dl_df, f_pre)))
        input_preactivation_grad = list(map(lambda x: x[0] * sigmoid_derivative(x[1]), zip(dl_di, i_pre)))
        cell_preactivation_grad = list(map(lambda x: x[0] * tanh_derivative(x[1]), zip(dl_dc, c_pre)))
        output_preactivation_grad = list(map(lambda x: x[0] * sigmoid_derivative(x[1]), zip(dl_do, o_pre)))

        # all shapes above this: (batch size, num units)

        w_f_grad = np.array(list(map(lambda x: np.dot(x[0].T, x[1]), zip(forget_preactivation_grad, x)))).sum(0)
        w_i_grad = np.array(list(map(lambda x: np.dot(x[0].T, x[1]), zip(input_preactivation_grad, x)))).sum(0)
        w_c_grad = np.array(list(map(lambda x: np.dot(x[0].T, x[1]), zip(cell_preactivation_grad, x)))).sum(0)
        w_o_grad = np.array(list(map(lambda x: np.dot(x[0].T, x[1]), zip(output_preactivation_grad, x)))).sum(0)

        u_f_grad = np.array(list(map(lambda x: np.dot(x[0].T, x[1]), zip(forget_preactivation_grad, prev_h)))).sum(0)
        u_i_grad = np.array(list(map(lambda x: np.dot(x[0].T, x[1]), zip(input_preactivation_grad, prev_h)))).sum(0)
        u_c_grad = np.array(list(map(lambda x: np.dot(x[0].T, x[1]), zip(cell_preactivation_grad, prev_h)))).sum(0)
        u_o_grad = np.array(list(map(lambda x: np.dot(x[0].T, x[1]), zip(output_preactivation_grad, prev_h)))).sum(0)

        b_f_grad = np.array(list(map(lambda x: np.sum(x, axis=0), forget_preactivation_grad))).sum(0)
        b_i_grad = np.array(list(map(lambda x: np.sum(x, axis=0), input_preactivation_grad))).sum(0)
        b_c_grad = np.array(list(map(lambda x: np.sum(x, axis=0), cell_preactivation_grad))).sum(0)
        b_o_grad = np.array(list(map(lambda x: np.sum(x, axis=0), output_preactivation_grad))).sum(0)

        # gradients of the loss w.r.t the input xt in each gate for each timeframe
        # shape: (num features)
        dfpre_dx = list(map(lambda x: np.dot(np.sum(x, axis=0), params[0]), forget_preactivation_grad))
        dipre_dx = list(map(lambda x: np.dot(np.sum(x, axis=0), params[1]), input_preactivation_grad))
        dcpre_dx = list(map(lambda x: np.dot(np.sum(x, axis=0), params[2]), cell_preactivation_grad))
        dopre_dx = list(map(lambda x: np.dot(np.sum(x, axis=0), params[3]), output_preactivation_grad))

        # gradient of the loss w.r.t the input xt for each timeframe
        x_grad = list(map(lambda i: dfpre_dx[i] + dipre_dx[i] + dcpre_dx[i] + dopre_dx[i], range(len(caches))))

        return (w_f_grad, w_i_grad, w_c_grad, w_o_grad, u_f_grad, u_i_grad, u_c_grad, u_o_grad, b_f_grad, b_i_grad, b_c_grad, b_o_grad), x_grad


    return lstm_layer, backward, initialize_params


def create_dense_layer(num_inputs, num_neurons, activation):
    def initialize_params():
        high = math.sqrt( 6 / (num_inputs + num_neurons))
        low = -high
        weights = np.random.uniform(low, high, ((num_neurons, num_inputs)))
        biases = np.random.uniform(low, high, ((num_neurons)))
        return weights, biases

    def dense_layer(input, params):
        weights, biases = params
        preactivation = np.dot(input, weights.T) + biases

        # returns extra information needed for backpropagation
        cache: lambda: (input, preactivation, weights)

        return activation(preactivation), cache
    
    def backward(dl_do, cache):
        input, preactivation, weights = cache()
        # gradient of the output of the layer w.r.t the preactivation of the layer
        do_dp = elementwise_grad(activation)(preactivation)
        
        # gradient of the output of the layer w.r.t the weights, biases, and input
        wgrad = np.dot((do_dp * dl_do).T, input)
        bgrad = np.sum(do_dp * dl_do, axis=0, keepdims=True)
        igrad = np.dot(do_dp * dl_do, weights)

        return (wgrad, bgrad), igrad
    
    return dense_layer, backward, initialize_params
    