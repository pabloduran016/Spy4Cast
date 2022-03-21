from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from spy4cast._procedure import _Procedure
from spy4cast.preprocess import Preprocess


class MCA(_Procedure):
    RUY: npt.NDArray[np.float32]
    RUY_sig: npt.NDArray[np.float32]
    SUY: npt.NDArray[np.float32]
    SUY_sig: npt.NDArray[np.float32]
    RUZ: npt.NDArray[np.float32]
    RUZ_sig: npt.NDArray[np.float32]
    SUZ: npt.NDArray[np.float32]
    SUZ_sig: npt.NDArray[np.float32]
    Us: npt.NDArray[np.float32]
    Vs: npt.NDArray[np.float32]
    scf: npt.NDArray[np.float32]
    alpha: float

    def __init__(
        self,
        z: Preprocess,
        y: Preprocess,
        nm: int, alpha: float
    ):
        raise NotImplementedError

    def plot(
        self,
        flags: int = 0,
        dir: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    @classmethod
    def load(self, prefix: str, dir: str = '.') -> None:
        raise NotImplementedError

    def save(self, prefix: str, dir: str = '.') -> None:
        raise NotImplementedError


def _index_regression(
    data: npt.NDArray[np.float32],
    index: npt.NDArray[np.float32],
    alpha: float
) -> Tuple[npt.NDArray[np.float32], ...]:
    raise NotImplementedError
