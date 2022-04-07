from enum import Enum
from typing import Tuple


class _PlotType(Enum):
    MAP = 'map'
    TS = 'ts'

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        return 'map', 'ts'

def _get_type(val: str) -> _PlotType:
    if val not in _PlotType.values():
        raise TypeError(
            f'Exected type to be one of '
            f'{{{", ".join(_PlotType.values())}}}, '
            f'but got `{val}`'
        )
    return _PlotType(val)


from .anom import Anom
from .clim import Clim


__all__ = [
    'Anom',
    'Clim',
]
