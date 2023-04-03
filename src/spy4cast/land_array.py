from typing import Tuple

import numpy as np
import numpy.typing as npt


class LandArray:
    """Array that manages land as `nan` values

    Note
    ----
    Assumes that land is any point that has any nan values on any time

    Paramenters
    -----------
    data : array_like
        Space x Time array with `nan` on land data

    Attributes
    ----------
    values : array_like
        Complete data

    """

    def __init__(self, values: npt.NDArray[np.float_]) -> None:
        self.values = values
        self._land_indices = np.nonzero(np.isnan(values).any(1))[0]
        self._not_land_indices = np.nonzero(~np.isnan(values).any(1))[0]
        self._not_land_values = values[self._not_land_indices, :]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the values matrix"""
        return self.values.shape

    @property
    def not_land_values(self) -> npt.NDArray[np.float_]:
        """Data ordered but jumps over land data-points (`nan`)."""
        return self._not_land_values

    @property
    def land_indices(self) -> npt.NDArray[np.int_]:
        """Indices of the land data points"""
        return self._land_indices

    @property
    def not_land_indices(self) -> npt.NDArray[np.int_]:
        """Indices of the NOT land data points"""
        return self._not_land_indices
