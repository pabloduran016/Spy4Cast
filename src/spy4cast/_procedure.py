from abc import ABC, abstractmethod
from typing import Optional


class _Procedure(ABC):
    @abstractmethod
    def plot(
        self,
        flags: int = 0,
        dir: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, prefix: str, dir: str = '.') -> None:
        raise NotImplementedError
