import autograd.numpy as np
import numpy.typing as npt
import math
from autograd import elementwise_grad, grad

class Layer:

    # input_shape: the shape of the inputs -> (num_timeframes, num features)
    def __init__(self, num_units: int, input_shape: tuple[int]) -> None:
        self.num_units = num_units
        self.num_features = input_shape[1]
        self.num_frames = input_shape[0]

        # intializes weights using orthogonal initialization
        # (glorot/xavier uniform initialization for now)
        high = math.sqrt( 6 / (self.num_features + self.num_units))
        low = -high

        # J: number of features
        # K: number of outputs/units
        # T: number of timesteps

        # input dimentions -> (T, J)

        # 4 input weight matrices, one for each gate
        # input weights -> (K, T)
        
        self.w_f = np.random.uniform(low, high, ((num_units, self.num_features)))
        self.w_i = np.random.uniform(low, high, ((num_units, self.num_features)))
        self.w_c = np.random.uniform(low, high, ((num_units, self.num_features)))
        self.w_o = np.random.uniform(low, high, ((num_units, self.num_features)))

        # 4 recurrent weight matrices, one for each gate
        # recurrent weights -> (K, K)
        self.u_f = np.random.uniform(low, high, ((num_units, num_units)))
        self.u_i = np.random.uniform(low, high, ((num_units, num_units)))
        self.u_c = np.random.uniform(low, high, ((num_units, num_units)))
        self.u_o = np.random.uniform(low, high, ((num_units, num_units)))


        # 4 bias matrices, one for each gate
        # biases -> (K)
        self.b_f = np.random.uniform(low, high, ((num_units)))
        self.b_i = np.random.uniform(low, high, ((num_units)))
        self.b_c = np.random.uniform(low, high, ((num_units)))
        self.b_o = np.random.uniform(low, high, ((num_units)))

        # gate outputs -> (K)

        # cell state -> (K)
        self.cell_state = np.zeros((num_units))

        # hidden state -> (K)
        self.hidden_state = np.zeros((num_units))

        # intermediate values needed for backpropagation
        # storing the following in order:
        # 1. previous cell state
        # 2. previous hidden state
        # 3. forget gate activation
        # 4. input gate activation
        # 5. cell gate activation
        # 6. output gate activation
        # 7. new cell state
        # 8. new hidden state
        # 9. input
        self.caches = []
    
        

    @staticmethod
    def sigmoid(input: npt.NDArray) -> npt.NDArray:
        return np.array([1 / (1 + np.exp(-x)) for x in input])
    
    @staticmethod
    def tanh(input: npt.NDArray) -> npt.NDArray:
        return np.array([  (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  for x in input])
       

    # inputs shape: (num timeframes, num features)
    # output shape: (num timeframes, num units)
    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        if inputs.shape != (self.num_frames, self.num_features):
            raise Exception('Incorrect input shape') 

        outputs = []

        for timestep in inputs:
            c = [self.cell_state, self.hidden_state]

            # forget gate for remembering only a certain part of the cell state
            f_preactivation = np.dot(self.w_f, timestep) + np.dot(self.u_f, self.hidden_state) + self.b_f
            forget = self.sigmoid(f_preactivation)

            # input gate for determining what percent of the potential cell state to add to the cell state
            i_preactivation = np.dot(self.w_i, timestep) + np.dot(self.u_i, self.hidden_state) + self.b_i
            input = self.sigmoid(i_preactivation)

            # cell gate for determining the potential cell state value
            c_preactivation = np.dot(self.w_c, timestep) + np.dot(self.u_c, self.hidden_state) + self.b_c
            cell = self.tanh(c_preactivation)

            # output gate for getting the new hidden state from the new cell state
            o_preactivation = np.dot(self.w_o, timestep) + np.dot(self.u_o, self.hidden_state) + self.b_o
            output = self.sigmoid(o_preactivation)

            # shapes ^ : (num units)

            # update cell state and hidden state
            self.cell_state = forget * self.cell_state + input * cell
            self.hidden_state = self.tanh(self.cell_state) * output

            # store intermediate values needed for backward pass
            c.extend([forget, input, cell, output, self.cell_state, self.hidden_state, inputs, f_preactivation, i_preactivation, c_preactivation, o_preactivation])
            self.caches.append(c)

            outputs.append(self.hidden_state)

        
        return np.array(outputs)
    
    
    # pred/actual shape: (num timesteps, num units)
    # returns tuple of gradients 
    def backward(self, pred: npt.NDArray, actual: npt.NDArray) -> tuple:
        loss_func = lambda p, a: ((p - a) ** 2).mean(0)[0]
        # loss = loss_func(pred, actual)
        # find the gradient of the loss function w.r.t its input (the hidden states)
        dl_dh = grad(loss_func)(pred, actual)
        
        tanh_derivative = elementwise_grad(self.tanh)
        sigmoid_derivative = elementwise_grad(self.sigmoid)
        
        # initialize gradients, the gradients for all the time frames will be added up
        w_f_grad = np.zeros((self.num_units, self.num_features))
        w_i_grad = np.zeros((self.num_units, self.num_features))
        w_c_grad = np.zeros((self.num_units, self.num_features))
        w_o_grad = np.zeros((self.num_units, self.num_features))

        u_f_grad = np.zeros((self.num_units, self.num_units))
        u_i_grad =np.zeros((self.num_units, self.num_units))
        u_c_grad =np.zeros((self.num_units, self.num_units))
        u_o_grad = np.zeros((self.num_units, self.num_units))
        
        b_f_grad = np.zeros((self.num_units))
        b_i_grad = np.zeros((self.num_units))
        b_c_grad = np.zeros((self.num_units))
        b_o_grad = np.zeros((self.num_units))

        # looping through timesteps reversed
        for t in reversed(range(len(self.caches))):
            prev_c, prev_h, f, i, c, o, new_c, new_h, inputs, f_pre, i_pre, c_pre, o_pre = self.caches[t]
            
            # gradient of the hidden states w.r.t the output gate activations
            dh_do = self.tanh(new_c)

            # gradient of the hidden states w.r.t the cell state 
            dh_dcs = o * tanh_derivative(new_c)

            # gradient of the hidden states w.r.t the inputs gate activations
            dh_di = dh_dcs * c
            
            # gradient of the hidden states w.r.t the forget gate activations
            dh_df = dh_dcs * prev_c
            
            # gradient of the hidden states w.r.t the cell gate activations
            dh_dc = dh_dcs * i
            

            # gradients of the loss w.r.t the preactivations of each gate
            # shape: (num units)
            # [..., None] just converts the 1 dimensional vectors into arrays of shape (n, 1)
            # it is needed for dot products later
            forget_preactivation_grad = (dh_df * sigmoid_derivative(f_pre) * dl_dh[t])[..., None]
            input_preactivation_grad = (dh_di * sigmoid_derivative(i_pre) * dl_dh[t])[..., None]
            cell_preactivation_grad = (dh_dc * tanh_derivative(c_pre) * dl_dh[t])[..., None]
            output_preactivation_grad = (dh_do * sigmoid_derivative(o_pre) * dl_dh[t])[..., None]

            # input at current timeframe shape: (num features)
            # previous hidden state shape: (num units)
            w_f_grad = w_f_grad + np.dot(forget_preactivation_grad, inputs[t])
            w_i_grad = w_i_grad + np.dot(input_preactivation_grad, inputs[t])
            w_c_grad = w_c_grad + np.dot(cell_preactivation_grad, inputs[t])
            w_o_grad = w_o_grad + np.dot(output_preactivation_grad, inputs[t])

            u_f_grad = u_f_grad + np.dot(forget_preactivation_grad, prev_h[None, ...])
            u_i_grad = u_i_grad + np.dot(input_preactivation_grad, prev_h[None, ...])
            u_c_grad = u_c_grad + np.dot(cell_preactivation_grad, prev_h[None, ...])
            u_o_grad = u_o_grad + np.dot(output_preactivation_grad, prev_h[None, ...])
            
            b_f_grad = b_f_grad + forget_preactivation_grad
            b_i_grad = b_i_grad + input_preactivation_grad
            b_c_grad = b_c_grad + cell_preactivation_grad
            b_o_grad = b_o_grad + output_preactivation_grad
            
        return w_f_grad, w_i_grad, w_c_grad, w_o_grad, u_f_grad, u_i_grad, u_c_grad, u_o_grad, b_f_grad, b_i_grad, b_c_grad, b_o_grad
            


# mock dataset
dataset = {
    'X': [[i] for i in range(100)],
    'Y': [[-7.55], [1.7], [0.46], [12.87], [3.23], [9.0], [13.76], [5.5], [8.81], [10.42], [9.54], [15.16], [2.82], [15.1], [9.01], [24.64], [7.73], [10.26], [21.15], [25.27], [22.38], [13.04], [26.19], [16.25], [29.1], [20.3], [28.7], [35.27], [26.09], [35.0], [39.65], [31.43], [37.67], [42.35], [35.52], [32.1], [32.9], [37.37], [36.36], [43.16], [43.46], [36.63], [51.89], [37.38], [47.2], [43.84], [50.8], [53.6], [50.64], [39.8], [48.79], [52.59], [56.62], [56.07], [51.17], [48.43], [62.9], [65.31], [67.9], [60.12], [52.94], [64.39], [60.18], [55.91], [69.97], [62.98], [59.4], [59.85], [76.19], [77.52], [65.18], [78.6], [73.91], [67.58], [77.51], [68.71], [82.41], [79.29], [80.03], [72.38], [75.47], [71.64], [85.3], [87.85], [84.11], [82.8], [90.73], [88.83], [92.51], [91.63], [82.8], [81.69], [89.7], [85.67], [90.61], [99.3], [96.85], [100.66], [107.01], [102.59]]
}

# num units should correspond to the number of output features on the last layer
lstm = Layer(num_units=4, input_shape=(10, 1))
# 1 features in each timeframe
# 1 units
# 10 timesteps
pred = lstm(np.array(dataset['X'][:10]))
grads = lstm.backward(pred, np.array(dataset['Y'][10]))