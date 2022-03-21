from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt

from spy4cast._procedure import _Procedure
from spy4cast.preprocess import Preprocess


class Crossvalidation(_Procedure):
    zhat: npt.NDArray[np.float32]
    scf: npt.NDArray[np.float32]
    r_z_zhat_t: npt.NDArray[np.float32]
    p_z_zhat_t: npt.NDArray[np.float32]
    r_z_zhat_s: npt.NDArray[np.float32]
    p_z_zhat_s: npt.NDArray[np.float32]
    r_uv: npt.NDArray[np.float32]
    p_uv: npt.NDArray[np.float32]
    us: npt.NDArray[np.float32]
    alpha: float

    def __init__(
        self,
        y: Preprocess,
        z: Preprocess,
        nm: int,
        alpha: float,
        multiprocessing: bool = False
    ):
        raise NotImplementedError

    def _crossvalidate_year(
        self,
        year: int,
        z: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        nt: int,
        ny: int,
        yrs: npt.NDArray[np.int32],
        nm: int,
        alpha: float
    ) -> Tuple[npt.NDArray[np.float32], ...]:
        raise NotImplementedError

    def plot(
        self,
        flags: int = 0,
        dir: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    def plot_zhat(
        self,
        year: int,
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
