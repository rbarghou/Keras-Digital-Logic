import keras
from keras.layers import SimpleRNN, Dense
from keras.models import Sequential

import numpy as np

import tensorflow as tf


def create_oscillator_data_set(time_steps, n_samples, min_duration, wavelength=1, np_seed=None):
    """
    Create a data set of input outputs where each input has one stretch of high values.
     Each output has a corresponding period of alternating high-low patterns.
    :param time_steps:
    :param n_samples:
    :param min_duration:
    :param wavelength:
    :param np_seed:
    :return (X, Y): both of shapes (n_samples, n_time_steps)
    """
    if np_seed is not None:
        np.random.seed(np_seed)

    _X = []
    _Y = []
    for _ in range(n_samples):
        on, off = (0, 0)
        while abs(off - on) < min_duration:
            on, off = sorted(np.random.choice(range(time_steps + 1), 2))

        _X_sample = -np.ones((time_steps, 1))
        _X_sample[on:off, 0] = 1.0

        _Y_sample = -np.ones((time_steps, 1))
        _Y_sub = _Y_sample[on:off, 0]
        _Y_sub = _Y_sub ** ((np.array(range(off - on)) / wavelength).astype(np.int))
        _Y_sample[on:off, 0] = _Y_sub

        _X.append(_X_sample)
        _Y.append(_Y_sample)

    _X = np.array(_X)
    _Y = np.array(_Y)

    return _X, _Y


def construct_model(n_neurons,
                    time_steps,
                    recurrent_activation="tanh",
                    final_layer_activation="tanh",
                    final_layer_recurrent=False,
                    tf_seed=None,
                    clear_session=True):
    if tf_seed is not None:
        tf.set_random_seed(tf_seed)

    if clear_session:
        keras.backend.clear_session()

    model = Sequential()
    model.add(SimpleRNN(n_neurons,
                        activation=recurrent_activation,
                        return_sequences=True,
                        input_shape=(time_steps, 1)))
    if final_layer_recurrent:
        model.add(SimpleRNN(1, activation=final_layer_activation, return_sequences=True))
    else:
        model.add(Dense(1, activation=final_layer_activation))

    return model
