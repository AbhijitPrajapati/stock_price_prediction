import autograd.numpy as np
import numpy.typing as npt
import typing
import pandas as pd
import math
from autograd import elementwise_grad, grad

class LSTM_Layer:

    def __init__(self, num_units: int, num_timeframes: int, num_features: int, batch_size: int, loss: typing.Callable) -> None:
        self.num_units = num_units
        self.num_features = num_features
        self.num_frames = num_timeframes
        self.batch_size = batch_size
        self.loss = loss

        # intializes weights using orthogonal initialization
        # (glorot/xavier uniform initialization for now)
        high = math.sqrt( 6 / (self.num_features + self.num_units))
        low = -high

        # J: number of features
        # K: number of outputs/units
        # T: number of timesteps
        # B: batch size

        # input dimentions -> (B, T, J)

        self.params = [ 
                # 4 input weight matrices, one for each gate
                # input weights -> (K, T)
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

        # cell state -> (B, K)
        self.cell_state = np.zeros((batch_size, num_units))

        # hidden state -> (B, K)
        self.hidden_state = np.zeros((batch_size, num_units))

        # intermediate values needed for backpropagation
        self.caches = []
    
        

    @staticmethod
    def sigmoid(input: npt.NDArray) -> npt.NDArray:
        return np.array([1 / (1 + np.exp(-x)) for x in input])
    
    @staticmethod
    def tanh(input: npt.NDArray) -> npt.NDArray:
        return np.array([  (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  for x in input])
    

    # inputs shape: (batch_size, num timeframes, num features)
    # output shape: (batch_size, num timeframes, num units)
    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        # reset state and cache
        self.hidden_state = np.zeros((self.batch_size, self.num_units))
        self.cell_state = np.zeros((self.batch_size, self.num_units))
        self.caches = []

        if inputs.shape != (self.batch_size, self.num_frames, self.num_features):
            raise Exception(f'Incorrect input shape: {inputs.shape} should be {(self.batch_size, self.num_frames, self.num_features)}') 


        # iterating throught the timesteps
        for t in range(inputs.shape[1]):
            # cache value
            c = [self.cell_state, self.hidden_state]

            # the input
            timestep = inputs[:, t, :]

            w_f, w_i, w_c, w_o, u_f, u_i, u_c, u_o, b_f, b_i, b_c, b_o = self.params

            # forget gate for remembering only a certain part of the cell state
            f_preactivation = np.dot(timestep, w_f.T) + np.dot(self.hidden_state, u_f.T) + b_f
            forget = self.sigmoid(f_preactivation)

            # input gate for determining what percent of the potential cell state to add to the cell state
            i_preactivation = np.dot(timestep, w_i.T) + np.dot(self.hidden_state, u_i.T) + b_i
            input = self.sigmoid(i_preactivation)

            # cell gate for determining the potential cell state value
            c_preactivation = np.dot(timestep, w_c.T) + np.dot(self.hidden_state, u_c.T) + b_c
            cell = self.tanh(c_preactivation)

            # output gate for getting the new hidden state from the new cell state
            o_preactivation = np.dot(timestep, w_o.T) + np.dot(self.hidden_state, u_o.T) + b_o
            output = self.sigmoid(o_preactivation)

            # shapes ^ : (batch size, num units)

            # update cell state and hidden state
            self.cell_state = forget * self.cell_state + input * cell
            self.hidden_state = self.tanh(self.cell_state) * output

            # store intermediate values needed for backward pass
            c.extend([input, cell, output, self.cell_state, inputs, f_preactivation, i_preactivation, c_preactivation, o_preactivation])
            self.caches.append(c)

        
        return self.hidden_state
    
    
    # pred/actual shape: (batch size, num units)
    # returns tuple of gradients 
    def backward(self, pred: npt.NDArray, actual: npt.NDArray) -> tuple:
        # find the gradient of the loss function w.r.t its input (the hidden states)
        dl_dh = elementwise_grad(self.loss)(pred, actual)
        

        tanh_derivative = elementwise_grad(self.tanh)
        sigmoid_derivative = elementwise_grad(self.sigmoid)
        
        # initialize gradients, the gradients for all the time frames will be added up 
        w_f_grad, w_i_grad, w_c_grad, w_o_grad, u_f_grad, u_i_grad, u_c_grad, u_o_grad, b_f_grad, b_i_grad, b_c_grad, b_o_grad = (np.zeros_like(p) for p in self.params)

        # looping through timesteps reversed
        for t in reversed(range(len(self.caches))):
            prev_c, prev_h, i, c, o, new_c, inputs, f_pre, i_pre, c_pre, o_pre = self.caches[t]
            
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
            # shape: (batch size, num units)
            forget_preactivation_grad = (dh_df * sigmoid_derivative(f_pre) * dl_dh)
            input_preactivation_grad = (dh_di * sigmoid_derivative(i_pre) * dl_dh)
            cell_preactivation_grad = (dh_dc * tanh_derivative(c_pre) * dl_dh)
            output_preactivation_grad = (dh_do * sigmoid_derivative(o_pre) * dl_dh)

            # input at current timeframe shape: (batch size, num features)
            # previous hidden state shape: (batch size, num units)
            current_input = inputs[:, t, :]

            w_f_grad = w_f_grad + np.dot(forget_preactivation_grad.T, current_input)
            w_i_grad = w_i_grad + np.dot(input_preactivation_grad.T, current_input)
            w_c_grad = w_c_grad + np.dot(cell_preactivation_grad.T, current_input)
            w_o_grad = w_o_grad + np.dot(output_preactivation_grad.T, current_input)

            u_f_grad = u_f_grad + np.dot(forget_preactivation_grad.T, prev_h)
            u_i_grad = u_i_grad + np.dot(input_preactivation_grad.T, prev_h)
            u_c_grad = u_c_grad + np.dot(cell_preactivation_grad.T, prev_h)
            u_o_grad = u_o_grad + np.dot(output_preactivation_grad.T, prev_h)
            
            b_f_grad = b_f_grad + forget_preactivation_grad
            b_i_grad = b_i_grad + input_preactivation_grad
            b_c_grad = b_c_grad + cell_preactivation_grad
            b_o_grad = b_o_grad + output_preactivation_grad
            
        return w_f_grad, w_i_grad, w_c_grad, w_o_grad, u_f_grad, u_i_grad, u_c_grad, u_o_grad, b_f_grad, b_i_grad, b_c_grad, b_o_grad
    
    # turns data into batches
    def create_batches(self, data: npt.NDArray) -> tuple[npt.NDArray]:
        # the sequence length is one more than the number of timeframes because the sequences will later be divided into x and y
        sequence_length = self.num_frames + 1
        # create consequtive sequences from the long list of values 
        sequences = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i+sequence_length]
            sequences.append(seq)
        sequences = np.array(sequences)

        # some sequences need to be removed in order for the batches to be the same size
        num_batches = len(sequences) // self.batch_size
        sequences = sequences[: num_batches * self.batch_size]
        
        # the x values will be the preceeding sequence of values
        # the y values will be the single value after the preceeding sequence of values 
        x = sequences[:, :-1]
        y = sequences[:, -1]

        # reshape the sequences
        return x.reshape((num_batches, self.batch_size, self.num_frames, self.num_features)), y.reshape((num_batches, self.batch_size, self.num_features))
    
    # normalizes data
    # data shape: (num frames, num features)
    @staticmethod
    def min_max_normalization(data: npt.NDArray) -> tuple:
        # the data may have multiple features, so the min and max values for each column are needed
        min_vals = data.min(axis=0)
        max_vals_minus_min = data.max(axis=0) - min_vals

        return (data - min_vals) / max_vals_minus_min, (min_vals, max_vals_minus_min)
    
    # inverse normalization
    # data shape: (num frames, num features)
    # inverse_parameters: the parameters from the normalization that will be used
    @staticmethod
    def inverse_min_max_normalization(data: npt.NDArray, inverse_parameters: tuple[int]) -> npt.NDArray:
        min_vals, max_vals_minus_min = inverse_parameters
        return data * max_vals_minus_min + min_vals

    
    # trains model
    # implements adam optimizer
    # data: matrix of shape (timeframes, features) which will be split into batches later
    def train(self, data: npt.NDArray, learning_rate: float, epochs: int):

        data, inverse_parameters = self.min_max_normalization(data)
        
        x, y = self.create_batches(data)


        # hyperparameters
        epsilon = 10 ** -8
        beta_1 = 0.9
        beta_2 = 0.999

        # starting values
        # 12 lists correspond to the 12 parameters being optimized
        m = [np.zeros_like(p) for p in self.params]
        v = [np.zeros_like(p) for p in self.params]

        for epoch in range(1, epochs + 1):
            # shuffle data
            p = np.random.permutation(len(x))
            x = x[p]
            y = y[p]

            for batch in range(len(x)):
                xb = x[batch]
                yb = y[batch]
                
                pred = self(xb)

                lr_t = learning_rate * (np.sqrt(1 - beta_2**epoch) / (1 - beta_1**epoch))


                grads = self.backward(pred, yb)
                for i in range(len(grads)):
                    m[i] = beta_1 * m[i] + (1 - beta_1) * grads[i]
                    v[i] = beta_2 * v[i] + (1 - beta_2) * (grads[i] ** 2)

                    m_hat = m[i] / (1 - beta_1**epoch)
                    v_hat = v[i] / (1 - beta_2**epoch)

                    # update params
                    self.params[i] = self.params[i] - lr_t * m_hat / (np.sqrt(v_hat) + epsilon)

            print(f'Epoch: {epoch}\nLoss: {self.loss(pred, yb)}')


url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
data = pd.read_csv(url)['Close']

mse = lambda p, a: ((p - a) ** 2).mean()

# num_units: the dimensionality of the output of the lstm
# num_timeframes: the number of timeframes that the lstm will use to predict the next one
# num features: the number of features that the lstm is predicting at each timeframe
# batch_size: the model will process this amount of input-output-pairs at once before updating the gradients
lstm = LSTM_Layer(num_units=64, num_timeframes=20, num_features=1, batch_size=32, loss=mse)
lstm.train(data, 0.001, 100)

# need to implement:
# validation set in training
# run using gpu instead of cpu with bigger dataset
# implement a dataloader if needed