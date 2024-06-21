from layers import LSTM_Layer, Dense_Layer
import pandas as pd
import numpy as np
import typing
import numpy.typing as npt
from autograd import elementwise_grad

# turns sorted sequential data into batches
# returns x_train, y_train, x_test, y_test
def chronological_batching(data: npt.NDArray, batch_size: int, num_frames: int) -> tuple[npt.NDArray]:
    num_features = 1 if len(data.shape) == 1 else data.shape[1]

    # the sequence length is one more than the number of timeframes because the sequences will later be divided into x and y
    sequence_length = num_frames + 1
    # create consequtive sequences from the long list of values 
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        sequences.append(seq)
    sequences = np.array(sequences)   

    # some sequences need to be removed in order for the batches to be the same size
    num_batches = len(sequences) // batch_size
    sequences = sequences[: num_batches * batch_size]

    # reshape the sequences
    # the x values will be the preceeding sequence of values
    # the y values will be the single value after the preceeding sequence of values 
    x = sequences[:, :-1].reshape((num_batches, batch_size, num_frames, num_features))
    y = sequences[:, -1].reshape((num_batches, batch_size, num_features))
    
    return x, y


# normalizes data
# data shape: (num frames, num features)
def min_max_normalization(data: npt.NDArray) -> tuple:
    # the data may have multiple features, so the min and max values for each column are needed
    min_vals = data.min(axis=0)
    max_vals_minus_min = data.max(axis=0) - min_vals

    return (data - min_vals) / max_vals_minus_min, (min_vals, max_vals_minus_min)

# inverse normalization
# data shape: (num frames, num features)
# inverse_parameters: the parameters from the normalization that will be used
def inverse_min_max_normalization(data: npt.NDArray, inverse_parameters: tuple[int]) -> npt.NDArray:
    min_vals, max_vals_minus_min = inverse_parameters
    return data * max_vals_minus_min + min_vals


def predict(model: typing.Iterable, input: npt.NDArray) -> npt.NDArray:
    input, inverse_params = min_max_normalization(input)

    pred = input
    for layer in model:
        pred = layer(pred)
    
    pred = inverse_min_max_normalization(pred, inverse_params)

    return pred

# trains model
# implements adam optimizer
# batch_size: the model will process this amount of input-output-pairs at once before updating the gradients
def train(model: typing.Iterable, x: npt.NDArray, y: npt.NDArray, learning_rate: float, epochs: int, loss: typing.Callable, 
          x_test: npt.NDArray = None, y_test: npt.NDArray = None) -> None:
    # hyperparameters
    epsilon = 10 ** -8
    beta_1 = 0.9
    beta_2 = 0.999


    for epoch in range(1, epochs + 1):

        lr_t = learning_rate * (np.sqrt(1 - beta_2**epoch) / (1 - beta_1**epoch))

        for batch in range(len(x)):
            xb = x[batch]
            yb = y[batch]    
            
            # foreward pass
            pred = xb
            for layer in model:
                pred = layer(pred, training=True)

            # backward pass
            # gradient of the loss w.r.t the output of the current layer 
            dl_do = elementwise_grad(loss)(pred, yb)
            for layer in reversed(model):
                # gradients w.r.t the parameters and the inputs
                pgrads, igrads = layer.backward()

                # process grads of current layer
                for i in range(len(pgrads)):
                    m = np.zeros_like(layer.params[i])
                    v = np.zeros_like(layer.params[i])
                    m = beta_1 * m + (1 - beta_1) * pgrads[i]
                    v = beta_2 * v + (1 - beta_2) * (pgrads[i] ** 2)

                    m_hat = m / (1 - beta_1**epoch)
                    v_hat = v / (1 - beta_2**epoch)

                    # update params
                    layer.params[i] = layer.params[i] - lr_t * m_hat / (np.sqrt(v_hat) + epsilon)
                
                dl_do = dl_do * igrads
            
        # validation
        vpred = x_test[batch]
        for layer in model:
            vpred = layer(vpred)
        l = loss(vpred, y_test[batch])
        print(f'Epoch: {epoch}\nLoss: {loss}')
    

url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
data = pd.read_csv(url)['Close']

mse = lambda p, a: ((p - a) ** 2).mean()

linear_activation = lambda x: x

# num_units: the dimensionality of the output of the lstm
# num_timeframes: the number of timeframes that the lstm will use to predict the next one
# num features: the number of features that the lstm is predicting at each timeframe
lstm = LSTM_Layer(num_units=64, input_shape=(20, 1))

# num inputs: number of inputs to the layer
# num neurons: number of outputs to the layer
# activation: the activation function
dense = Dense_Layer(num_inputs=64, num_neurons=1, activation=linear_activation)

model = [lstm, dense]


data, _ = min_max_normalization(data)

x, y = chronological_batching(data, 32, 20)

split = round(0.8 * x.shape[0])
x_train, y_train, x_test, y_test = x[:split], y[:split], x[split:], y[split:]

train(model, x_train, y_train, 0.001, 100, loss=mse, x_test=x_test, y_test=y_test)