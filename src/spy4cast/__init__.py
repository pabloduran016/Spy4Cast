from .dataset import Dataset
from .stypes import Slise, Month

__all__ = [
    'Slise',
    'Month',
    'Dataset',
    'set_silence'
]


class Settings:
    """Stores the settings that can be modified by the user"""
    silence: bool = True
    """Bool that if set to `True` indicates the program to
    don't output information about the process"""


def set_silence(b: bool) -> None:
    """Set the silence for the output"""
    if type(b) != bool:
        raise TypeError(f'Expected bool got {type(b)}')
    Settings.silence = b
