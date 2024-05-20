import numpy as np

class Layer:

    # input_size: the number of features
    # num_frames: the number of time frames
    # input to layer will be of shape (num_timesteps, num_features)
    def __init__(self, num_features, num_units):
        self.num_units = num_units
        self.num_features = num_features

        # intializes weights in shape using orthogonal initialization
        # (H.E initialization for now)
        varience = 0.1

        
        # initialize weights and biases

        # J: number of features
        # K: number of outputs/units
        # T: number of timesteps

        # input dimentions -> (T, J)

        # 4 input weight matrices, one for each gate
        # input weights -> (K, T)

        # 4 recurrent weight matrices, one for each gate
        # recurrent weights -> (K, K)

        # 4 bias matrices, one for each gate
        # biases -> (K, 1)

        # gate outputs -> (K, J)

        # cell state -> (K, J)
        # hidden state -> (K, J)


        self.cell_state = np.zeros((num_units))
        
        self.hidden_state = np.zeros((num_units))

        

    # input: vector
    @staticmethod
    def sigmoid(input):
        return np.array([1 / (1 + np.exp(-x)) for x in input])
    
    # input: vector
    @staticmethod
    def tanh(input):
        return np.array([  (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  for x in input])


    # inputs shape: (num timeframes, num features)
    def __call__(self, timesteps):

        # for inputs in timesteps:
            # forget = sigmoid(forget_input_w * inputs   +   forget_recurrent_w * hidden_state   +   forget_b)

            # input = sigmoid(input_input_w * inputs   +   input_recurrent_w * hidden_state   +   input_b)

            # cell = tanh(cell_input_w * inputs   +   cell_recurrent_w * hidden_state   +   cell_b)

            # output = sigmoid(output_input_w * inputs   +   output_recurrent_w * hidden_state   +   output_b)

            # cell_state = forget * cell_state   +   input * cell
            # hidden_state = tanh(cell_state) * output
        
        pass


lstm = Layer(2, 4)
# 2 features in each timeframe
# 4 units
# 5 timesteps
print(lstm([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))