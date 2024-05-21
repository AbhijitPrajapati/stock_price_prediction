import numpy as np
import numpy.typing as npt
import math

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
    
        

    @staticmethod
    def sigmoid(input: npt.NDArray) -> npt.NDArray:
        return np.array([1 / (1 + np.exp(-x)) for x in input])
    
    @staticmethod
    def tanh(input: npt.NDArray) -> npt.NDArray:
        return np.array([  (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  for x in input])


    # inputs shape: (num timeframes, num features)
    # output shape: (num timeframes, num units)
    def __call__(self, inputs: npt.NDArray) -> npt.NDArray:
        outputs = []

        for timestep in inputs:
            # forget gate for remembering only a certain part of the cell state
            forget = self.sigmoid(np.dot(self.w_f, timestep) + np.dot(self.u_f, self.hidden_state) + self.b_f)

            # input gate for determining what percent of the potential cell state to add to the cell state
            input = self.sigmoid(np.dot(self.w_i, timestep) + np.dot(self.u_i, self.hidden_state) + self.b_i)

            # cell gate for determining the potential cell state value
            cell = self.tanh(np.dot(self.w_c, timestep) + np.dot(self.u_c, self.hidden_state) + self.b_c)

            # output gate for getting the new hidden state from the new cell state
            output = self.sigmoid(np.dot(self.w_o, timestep) + np.dot(self.u_o, self.hidden_state) + self.b_o)

            # update cell state and hidden state
            self.cell_state = forget * self.cell_state + input * cell
            self.hidden_state = self.tanh(self.cell_state) * output

            outputs.append(self.hidden_state)

            # shapes ^ : (num units)
        
        return np.array(outputs)
        


lstm = Layer(num_units=4, input_shape=(5, 2))
# 2 features in each timeframe
# 4 units
# 5 timesteps
print(lstm(np.array([[1, 1], 
                     [2, 2], 
                     [3, 3], 
                     [4, 4], 
                     [5, 5]])))