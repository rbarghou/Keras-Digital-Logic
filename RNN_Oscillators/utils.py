import numpy as np


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
