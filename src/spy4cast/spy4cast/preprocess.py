from typing import Optional

from ..dataset import Dataset
from .._procedure import _Procedure


class Preprocess(_Procedure):
    def __init__(
        self,
        ds: Dataset,
        order: Optional[int] = None,
        period: Optional[int] = None
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
