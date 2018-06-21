import gc

import json

import keras
from keras.callbacks import Callback
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


class RNNOExperimentNLJSONLogger(Callback):
    """
    Logs the loss, configuration and weights of an RNNO every epoch.

    Writes this data in NLJSON format.
    """

    class ModeNotValid(Exception):
        pass

    def __init__(self, open_file, **extra_data):
        if open_file.closed or open_file.mode not in "wa":
            raise RNNOExperimentNLJSONLogger.ModeNotValid(
                "You must pass in an file open for writing or appending"
            )
        self.file = open_file
        self.extra_data = extra_data
        super(RNNOExperimentNLJSONLogger, self).__init__()

    def write_record(self, **kwargs):
        weights = self.model.get_weights()
        weights = [weight_array.tolist() for weight_array in weights]
        record = {
            "weights": weights,
            "params": self.params,
            "extra_data": self.extra_data,
        }
        record.update(kwargs)
        self.file.write("{}\n".format(json.dumps(record)))

    def on_train_begin(self, logs=None):
        self.write_record(**logs)

    def on_epoch_end(self, epoch, logs=None):
        self.write_record(**logs)


class ThresholdStopper(Callback):

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("acc")
        if acc > .9995:
            self.model.stop_training = True


def run_experimental_condition(
        file_path,
        time_steps=128,
        n_samples=1024,
        min_duration=3,
        wavelength=1,
        np_seed=None,
        n_neurons=1,
        recurrent_activation="tanh",
        final_layer_activation="tanh",
        final_layer_recurrent=False,
        tf_seed=None,
        clear_session=True,
        optimizer="adam",
        loss="mse",
        num_epochs=400):
    """
    :param file_path:
    :param time_steps:
    :param n_samples:
    :param min_duration:
    :param wavelength:
    :param np_seed:
    :param n_neurons:
    :param recurrent_activation:
    :param final_layer_activation:
    :param final_layer_recurrent:
    :param tf_seed:
    :param clear_session:
    :param optimizer:
    :param loss:
    :param num_epochs:
    :return:
    """
    model = construct_model(
        n_neurons,
        time_steps,
        recurrent_activation,
        final_layer_activation,
        final_layer_recurrent,
        tf_seed,
        clear_session
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )

    _X, _Y = create_oscillator_data_set(
        time_steps,
        n_samples,
        min_duration,
        wavelength,
        np_seed
    )

    with open(file_path, "w") as log_file:
        logger = RNNOExperimentNLJSONLogger(
            log_file,
            n_neurons=n_neurons,
            wavelength=wavelength,
            np_seed=np_seed,
            tf_seed=tf_seed,
            optimizer=optimizer,
            loss=loss,
            recurrent_activation=recurrent_activation,
            final_layer_recurrent=final_layer_recurrent,
            final_layer_activation=final_layer_activation,
        )
        stopper = ThresholdStopper()
        model.fit(_X, _Y, epochs=num_epochs, callbacks=[logger, stopper])

    keras.backend.clear_session()
    del model
    gc.collect()
