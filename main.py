import numpy as np

class Layer:

    # input_size: the size of the input of one timeframe
    # num_frames: the number of time frames
    def __init__(self, input_size, num_frames):
        self.num_frames = num_frames


        # intializes weights in shape using orthogonal initialization
        # (H.E initialization for now)
        self.weights = {
            # weights for using the inputs to determine what percent of the cell state should be remembered
            'forget': np.random.normal(0, 0.1, (num_frames, input_size)),
            # weights for using the hidden state to determine what percent of the cell state should be remembered
            'forget_recurrent': np.random.normal(0, 0.1, (num_frames, num_frames)),
            # weights for using the inputs to determine the candidate cell state
            'candidate_cell_state':  np.random.normal(0, 0.1, (num_frames, input_size)),
            # weights for using the hidden state to determine the candidate cell state
            'candidate_cell_state_recurrent': np.random.normal(0, 0.1, (num_frames, num_frames)),
            # weights for using the input to determine what percent of the candidate cell state should be added
            'cell_state':  np.random.normal(0, 0.1, (num_frames, input_size)),
            # weights for using the hidden state to determine what percent of the candidate cell state should be added
            'cell_state_recurrent': np.random.normal(0, 0.1, (num_frames, num_frames)),
            # weights for using the input to detemine the new short term memory
            'output':  np.random.normal(0, 0.1, (num_frames, input_size)),
            # weights for using the input to detemine the new short term memory
            'output_recurrent': np.random.normal(0, 0.1, (num_frames, num_frames))
        }
        
        self.biases = {
            # bias for forget gate
            'forget': np.zeros((num_frames)),
            # biases for cell state computation
            'candidate_cell_state': np.zeros((num_frames)),
            'cell_state': np.zeros((num_frames)),
            # bias for output
            'output': np.zeros((num_frames))
        }


        self.cell_state = np.zeros((num_frames))
        
        self.hidden_state = np.zeros((num_frames))

        # shape will be (num_frames, num_frames) because the different rows keep track of the hidden states of each of the units
        self.outputs = []

    # input: vector
    def sigmoid(input):
        return [1 / (1 + np.exp(-x)) for x in input]
    
    # input: vector
    def tanh(input):
        return [  (np.exp(x) - np(-x)) / (np.exp(x) + np.exp(-x))  for x in input]


    def forget_gate(self, input):
        # get the percentage of the cell state to be remembered
        input_forget = np.dot( self.weights['forget'], input )
        hidden_forget = np.dot( self.weights['forget_recurrent'], self.hidden_state )
        forget = input_forget + hidden_forget + self.biases['forget']
        self.cell_state *= forget
    
    def input_gate(self, input):
         # get a potential cell state
        input_candidate = np.dot( self.weights['candidate_cell_state'], input)
        hidden_candidate = np.dot( self.weights['candidate_cell_state_recurrent'], self.hidden_state )
        candidate = input_candidate + hidden_candidate + self.biases['candidate_cell_state']
        candidate = self.tanh(candidate)

        # get the percentage of the potential cell state to be added to the cell state
        input_cell_state = np.dot( self.weights['cell_state'], input )
        hidden_cell_state = np.dot( self.weights['cell_state_recurrent'], self.hidden_state )
        percent = input_cell_state + hidden_cell_state + self.biases['cell_state']
        percent = self.sigmoid(percent)

        self.cell_state += (candidate * percent)

    def output_gate(self, inputs):
        # get percentage of short term memory to make as new short term memory
        input_output = np.dot( self.weights['output'], inputs)
        hidden_output = np.dot( self.weights['output_recurrent'], self.hidden_state )
        percent = input_output + hidden_output + self.biases['output']
        percent = self.sigmoid(percent)

        # get potential output
        potential = self.tanh(self.cell_state)

        # update
        self.hidden_state = potential * percent
        self.outputs.append(self.hidden_state)



    # inputs shape: (num features, num time frames)
    def __call__(self, inputs):
        for input in inputs:

            # forget gate
            self.forget_gate(input)

            # input gate
            self.input_gate(input)

            # output gate
            self.output_gate(input)
        

        # output should be of shape (batch size, hidden dim)

        return self.outputs

class LSTM:
    def __init__(self, input_size):
        pass

    def add_layer(self, ):
        pass