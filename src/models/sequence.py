"""Causal fixed-length rolling windows for streaming feature dictionaries."""

from collections import deque
from numbers import Real


class RollingSequence:
    """Build fixed-length sequences ending at the most recently appended row.

    The first ``sequence_length - 1`` appends return ``None``. Values must be
    numeric; categorical encodings are intentionally left to a later phase.
    """

    def __init__(self, sequence_length=10, feature_names=None):
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        self.sequence_length = int(sequence_length)
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self._rows = deque(maxlen=self.sequence_length)

    def append(self, features):
        if self.feature_names is None:
            self.feature_names = tuple(features.keys())
            if not self.feature_names:
                raise ValueError("features must contain at least one value")
        if set(features) != set(self.feature_names):
            raise ValueError("feature keys changed within the stream")

        row = []
        for name in self.feature_names:
            value = features[name]
            if not isinstance(value, Real):
                raise TypeError(f"feature {name!r} must be numeric")
            row.append(float(value))
        self._rows.append(tuple(row))
        if len(self._rows) < self.sequence_length:
            return None
        # Return an immutable snapshot. Later appends cannot alter this
        # prediction/training example or introduce future observations.
        return tuple(self._rows)

    @property
    def n_features(self):
        return len(self.feature_names or ())
