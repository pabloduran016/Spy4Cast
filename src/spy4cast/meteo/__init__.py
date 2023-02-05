from enum import Enum
from typing import Tuple


class PlotType(Enum):
    """Enum for different types of plots in meteo"""

    MAP = 'map'
    TS = 'ts'

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        return tuple(x.value for x in cls)


def _get_type(val: str) -> PlotType:
    if val not in PlotType.values():
        raise TypeError(
            f'Exected type to be one of '
            f'{{{", ".join(PlotType.values())}}}, '
            f'but got `{val}`'
        )
    return PlotType(val)


from .anom import Anom
from .clim import Clim


__all__ = [
    'PlotType',
    'Anom',
    'Clim',
]
