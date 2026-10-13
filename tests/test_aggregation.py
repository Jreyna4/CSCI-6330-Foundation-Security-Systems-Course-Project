import numpy as np
import pytest

from aggregation import aggregate, fedavg, median, trimmed_mean


def make_updates(values, sizes=None):
    """One-layer 'models' whose single weight array is filled with `value`."""
    sizes = sizes or [1] * len(values)
    return [([np.full((2, 3), v, dtype=np.float32)], n) for v, n in zip(values, sizes)]


def test_fedavg_weights_by_client_size():
    out = fedavg(make_updates([0.0, 10.0], sizes=[3, 1]))
    np.testing.assert_allclose(out[0], 2.5)


def test_trimmed_mean_ignores_one_outlier_with_four_clients():
    out = trimmed_mean(make_updates([1.0, 2.0, 3.0, 1000.0]), beta=0.25)
    np.testing.assert_allclose(out[0], 2.5)  # mean of the middle two


def test_trimmed_mean_is_coordinate_wise():
    a = [np.array([0.0, 9.0])]
    b = [np.array([1.0, 1.0])]
    c = [np.array([2.0, 2.0])]
    d = [np.array([9.0, 0.0])]
    out = trimmed_mean([(a, 1), (b, 1), (c, 1), (d, 1)], beta=0.25)
    np.testing.assert_allclose(out[0], [1.5, 1.5])


def test_trimmed_mean_rejects_trimming_everything():
    with pytest.raises(ValueError):
        trimmed_mean(make_updates([1.0, 2.0, 3.0, 4.0]), beta=0.5)


def test_trimmed_mean_warns_when_nothing_is_trimmed():
    with pytest.warns(UserWarning):
        trimmed_mean(make_updates([1.0, 2.0, 3.0, 4.0]), beta=0.2)


def test_median():
    out = median(make_updates([1.0, 2.0, 100.0]))
    np.testing.assert_allclose(out[0], 2.0)


def test_aggregate_dispatch():
    updates = make_updates([1.0, 2.0, 3.0, 1000.0])
    out = aggregate(updates, {"aggregation": "trimmed_mean", "trim_beta": 0.25})
    np.testing.assert_allclose(out[0], 2.5)
    with pytest.raises(ValueError):
        aggregate(updates, {"aggregation": "nope"})
