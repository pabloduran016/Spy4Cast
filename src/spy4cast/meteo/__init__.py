from enum import Enum
from typing import Tuple


class _PlotType(Enum):
    MAP = 'map'
    TS = 'ts'

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        return 'map', 'ts'


from .anom import Anom
from .clim import Clim


__all__ = [
    'Anom',
    'Clim',
]
