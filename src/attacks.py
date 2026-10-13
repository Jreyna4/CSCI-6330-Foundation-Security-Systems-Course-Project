"""Simulated label-flipping attack (threat model from the proposal).

A malicious client relabels a fraction of its attack rows as benign before
training. The flipped set is drawn once per run (seeded) and used in every
round. Only the attacker's local copy of the labels changes; evaluation always
uses clean labels.
"""

import numpy as np


def flip_labels(y, flip_fraction, target_id, rng):
    """Return (poisoned copy of y, number of flipped rows)."""
    y = y.copy()
    candidates = np.flatnonzero(y != target_id)
    n_flip = int(round(flip_fraction * len(candidates)))
    idx = rng.choice(candidates, size=n_flip, replace=False)
    y[idx] = target_id
    return y, n_flip
