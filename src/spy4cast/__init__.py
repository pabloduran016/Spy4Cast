from .dataset import Dataset
from .stypes import Region, Month
from ._functions import region2str, season2str
from ._procedure import plot_map, plot_ts, add_cyclic_point_to_data, get_central_longitude_from_region, get_xlim_from_region

__all__ = [
    'Region',
    'Month',
    'Dataset',
    'set_silence',
    'Settings',
    'region2str',
    'season2str',
    'plot_map',
    'plot_ts',
    'add_cyclic_point_to_data',
    'get_central_longitude_from_region',
    'get_xlim_from_region',
]


class Settings:
    """Stores the settings that can be modified by the user"""
    silence: bool = False
    """Bool that if set to `True` indicates the program to
    don't output information about the process"""


def set_silence(b: bool) -> None:
    """Set the silence for the output"""
    if type(b) != bool:
        raise TypeError(f'Expected bool got {type(b)}')
    Settings.silence = b
