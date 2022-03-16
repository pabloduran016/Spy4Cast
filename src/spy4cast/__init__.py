from .plotters import ClimerTS, AnomerTS, AnomerMap, ClimerMap
from .spy4caster import Spy4Caster
from .stypes import Slise, Month, RDArgs, RDArgsDict, F
from .read_data import ReadData


__all__ = [
    'ClimerTS',
    'AnomerTS',
    'AnomerMap',
    'ClimerMap',
    'Spy4Caster',
    'Slise',
    'Month',
    'RDArgs',
    'RDArgsDict',
    'F',
    'ReadData',
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
