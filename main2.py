from layers2 import create_lstm_layer, create_dense_layer
from autograd import elementwise_grad
import pandas as pd
import numpy as np

def min_max_normalization(data):
    # the data may have multiple features, so the min and max values for each column are needed
    min_vals = data.min(axis=0)
    max_vals_minus_min = data.max(axis=0) - min_vals
    inverse = lambda data: data * max_vals_minus_min + min_vals
    return (data - min_vals) / max_vals_minus_min, inverse

# turns long string of timeframe values into batches 
def chronological_batching(data, batch_size, num_timeframes):
    num_features = 1 if len(data.shape) == 1 else data.shape[1]

    # the sequence length is one more than the number of timeframes because the sequences will later be divided into x and y
    sequence_length = num_timeframes + 1
    # create consequtive sequences from the long list of values 
    sequences = np.array(list(map(lambda i: data[i:i+sequence_length], range(len(data) - sequence_length))))  

    # some sequences need to be removed in order for the batches to be the same size
    num_batches = len(sequences) // batch_size
    cut_sequences = sequences[: num_batches * batch_size]

    # reshape the sequences
    # the x values will be the preceeding sequence of values
    # the y values will be the single value after the preceeding sequence of values 
    x = cut_sequences[:, :-1].reshape((num_batches, batch_size, num_timeframes, num_features))
    y = cut_sequences[:, -1].reshape((num_batches, batch_size, num_features))
    
    return x, y

# train_test_validation_split: length of three describing the percent starting point for each part
# negative one for not including validation
def preprocess_data(data, batch_size, num_timeframes, train_test_validation_split):
    normalized, _ = min_max_normalization(data)
    batched_x, batched_y = chronological_batching(normalized, batch_size, num_timeframes)

    train_start, test_start, validation_start = list(map(lambda i: None if train_test_validation_split[i] == -1 else round(len(batched_x) * train_test_validation_split[i]), 
                                                         range(len(train_test_validation_split))))
    x_train, y_train = batched_x[train_start:test_start], batched_y[train_start:test_start]
    x_test, y_test = batched_x[test_start:validation_start], batched_y[test_start:validation_start]
    if validation_start:
        x_valid, y_valid = batched_x[validation_start:], batched_y[validation_start:]
        return x_train, y_train, x_test, y_test, x_valid, y_valid
    return x_train, y_train, x_test, y_test



# runs after each batch and returns optimized parameters
def adam_optimizer(params, grads, epoch=1, learning_rate=0.001, **kwargs):
    epsilon = 10 ** -8
    beta_1 = 0.9
    beta_2 = 0.999

    m = kwargs.get('m', [np.zeros_like(g) for g in grads])
    v = kwargs.get('v', [np.zeros_like(g) for g in grads])

    first_moment_func = lambda i: beta_1 * m[i] + (1 - beta_1) * grads[i]
    second_moment_func = lambda i: beta_2 * m[i] + (1 - beta_2) * grads[i] ** 2
    new_m = list(map(first_moment_func, range(len(m))))
    new_v = list(map(second_moment_func, range(len(v))))

    bias_corrected_first_movement_func = lambda i: new_m[i] / (1 - beta_1 ** epoch)
    bias_corrected_second_movement_func = lambda i: new_v[i] / (1 - beta_2 ** epoch)
    m_hat = list(map(bias_corrected_first_movement_func, range(len(new_m))))
    v_hat = list(map(bias_corrected_second_movement_func, range(len(new_v))))

    update_func = lambda i: params[i] - (learning_rate / (np.sqrt(v_hat[i]) + epsilon)) * m_hat[i]
    new_params = list(map(update_func, range(len(params))))
    return new_params, (m, v)



def main():
    url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
    data = np.array(pd.read_csv(url)['Close'])

    mse = lambda p, a: ((p - a) ** 2).mean()

    linear_activation = lambda x: x

    lstm_layer, lstm_backward, lstm_init_params = create_lstm_layer(num_units=64, input_shape=(20, 1))
    lstm_params = lstm_init_params()

    dense_layer, dense_backward, dense_init_params = create_dense_layer(num_inputs=64, num_neurons=1, activation=linear_activation)
    dense_params = dense_init_params()

    x_train, y_train, x_test, y_test = preprocess_data(data, 32, 20, train_test_validation_split=(0, 0.7, -1))

    # proof of concept below

    x, y = x_train[0], y_train[0]

    lstm_out, lstm_back_cache = lstm_layer(x, lstm_params)
    
    dense_out, dense_back_cache = dense_layer(lstm_out, dense_params)

    # gradient of the loss w.r.t the output of the lstm
    dl_do = elementwise_grad(mse)(dense_out, y)

    dense_param_grads, lstm_out_grads = dense_backward(dl_do, dense_back_cache)

    lstm_param_grads, x_grads = lstm_backward(lstm_out_grads, lstm_back_cache)

    b1optimized, (m, v) = adam_optimizer(lstm_params + dense_params, lstm_param_grads + dense_param_grads)
    b1lstm_params, b1dense_params = b1optimized[:len(lstm_params)], b1optimized[len(dense_params):]

    


if __name__ == '__main__':
    main()