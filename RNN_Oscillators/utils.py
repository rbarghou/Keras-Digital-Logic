import numpy as np


def create_dataset(time_steps, n_samples, min_duration, np_seed=None):
    if np_seed:
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
        _Y_sub = _Y_sub ** range(off - on)
        _Y_sample[on:off, 0] = _Y_sub

        _X.append(_X_sample)
        _Y.append(_Y_sample)

    _X = np.array(_X)
    _Y = np.array(_Y)

    return _X, _Y
