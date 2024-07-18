from layers2 import create_lstm_layer, create_dense_layer
import pandas as pd
import numpy as np

# data shape: (num frames, num features)
def min_max_normalization(data):
    # the data may have multiple features, so the min and max values for each column are needed
    min_vals = data.min(axis=0)
    max_vals_minus_min = data.max(axis=0) - min_vals
    inverse = lambda data: data * max_vals_minus_min + min_vals
    return (data - min_vals) / max_vals_minus_min, inverse

def main():
    url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
    data = pd.read_csv(url)['Close']

    mse = lambda p, a: ((p - a) ** 2).mean()

    linear_activation = lambda x: x

    lstm_layer, lstm_backward, lstm_init_params = create_lstm_layer(num_units=64, input_shape=(20, 1))
    lstm_params = lstm_init_params()

    dense_layer, dense_backward, dense_init_params = create_dense_layer(num_inputs=64, num_neurons=1, activation=linear_activation)
    dense_params = dense_init_params()

    inp = np.array(data[-40:]).reshape((2, 20, 1))

    lstm_out, lstm_back_cache = lstm_layer(inp, lstm_params)
    
    dense_out, dense_back_cache = dense_layer(lstm_out, dense_params)

    print(dense_out)



if __name__ == '__main__':
    main()