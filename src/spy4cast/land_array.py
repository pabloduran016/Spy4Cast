from typing import Tuple, cast

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

    values: npt.NDArray[np.float_]
    _land_mask: npt.NDArray[np.bool_]

    def __init__(self, values: npt.NDArray[np.float_]) -> None:
        self.values = values
        self._land_mask = np.isnan(values).any(1)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the values matrix"""
        return self.values.shape

    @property
    def not_land_values(self) -> npt.NDArray[np.float_]:
        """Data ordered but jumps over land data-points (`nan`)."""
        return cast(npt.NDArray[np.float_], self.values[~self._land_mask])

    @property
    def land_mask(self) -> npt.NDArray[np.bool_]:
        """Mask of the land data points"""
        return self._land_mask

    def update_land(self, land_mask: npt.NDArray[np.bool_]) -> None:
        self._land_mask = land_mask

