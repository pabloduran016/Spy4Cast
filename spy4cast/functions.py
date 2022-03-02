import json
import os
import datetime
from typing import Optional, Dict, Any
from .stypes import Month, Slise
from time import perf_counter


__all__ = [
    'set_silence',
    'time_from_here',
    'time_to_here',

    'slise2str',
    'mon2str',
    'str2mon',
    'log_error',
    'debugprint',
]


# Array with the month indices
MONTH_TO_STRING = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
"""Array to translate from Month to string"""

VALID_MONTHS = list(filter(lambda x: not x.startswith('_'), Month.__dict__.keys()))
"""Array indicating valid months that can be passed to `str2mon`"""


class Settings:
    """Stores the settings that can be modified by the user"""
    silence: bool = True
    """Bool that if set to `True` indicates the program to dont output information about the process"""


def set_silence(b: bool) -> None:
    """Set the silence for the output"""
    if type(b) != bool:
        raise TypeError(f'Expected bool got {type(b)}')
    Settings.silence = b


_prev: Optional[float] = None
def time_from_here() -> None:
    """Function that is supposed to use in conjunctin with `time_to_here` to time program parts

    Example
    -------
        >>> print('Iterating through the first 100 million ints and adding their square to an array ', end='')
        ... arr = []
        ... time_from_here()
        ... for i in range(100_000_000):
        ...     arr.append(i ** 2)
        ...
        ... print(f'took: {time_to_here():.02f} seconds')
        Iterating through the first 100 million ints and adding their square to an array took: 30.27 seconds
    """
    global _prev
    _prev = perf_counter()
    return


def time_to_here() -> float:
    """Function that is supposed to use in conjunctin with `time_from_here` to time program parts

    Example
    -------
        >>> print('Iterating through the first 100 million ints and adding their square to an array ', end='')
        ... arr = []
        ... time_from_here()
        ... for i in range(100_000_000):
        ...     arr.append(i ** 2)
        ...
        ... print(f'took: {time_to_here():.02f} seconds')
        Iterating through the first 100 million ints and adding their square to an array took: 30.27 seconds
    """
    global _prev
    if _prev is None:
        raise ValueError('Expected to call time_from_here() before calling time_to_here()')
    rv = perf_counter() - _prev
    _prev = None
    return rv


def slise2str(slise: Slise) -> str:
    sufixes: Dict[str, str] = {'lat_min': '', 'lat_max': '', 'lon_min': '', 'lon_max': ''}
    values: Dict[str, float] = {'lat_min': slise.lat0, 'lat_max': slise.latf,
                                'lon_min': slise.lon0, 'lon_max': slise.lonf}
    for key in {'lat_min', 'lat_max'}:
        if values[key] >= 0:
            sufixes[key] = 'ºN'
        else:
            sufixes[key] = 'ºS'
        values[key] = abs(values[key])
    for key in {'lon_min', 'lon_max'}:
        if values[key] >= 0:
            sufixes[key] = 'ºE'
        else:
            sufixes[key] = 'ºW'
        values[key] = abs(values[key])
    return f'{values["lat_min"]}{sufixes["lat_min"]}, {values["lon_min"]}{sufixes["lon_min"]} - ' \
           f'{values["lat_max"]}{sufixes["lat_max"]}, {values["lon_max"]}{sufixes["lon_max"]}'


def mon2str(month: Month) -> str:
    """
    Function to return a month string depending on a month value
    :param month: Month from enum
    """
    if not 1 <= month <= 12:
        raise ValueError(f'Expected month number from 0 to 12, got {month}')
    # print('[WARNING] <month_to_string()> Changing month indexes!!!!!!!!!!!!!')
    return MONTH_TO_STRING[month - 1]


def str2mon(month: str) -> Month:
    """
    Function to return a month index depending on its string
    :param month: Value of the month. Minimum first three letters
    """
    month = month.upper()[:3]
    if not hasattr(Month, month):
        raise ValueError(f'Month not known, got {month}, valid values: {VALID_MONTHS}')
    return Month['JAN']


def log_error(string: str, path: Optional[str] = None) -> None:
    """
    Function to log an error in the errors foler on y_%y/m_%m/errors_%d.txt
    :param string: String to be logged
    :param path: Path to the erros folder, defult is website/log/errors
    :return:
    """
    timestamp = datetime.datetime.today()
    full_path = f'website/log/errors/{timestamp.year}/{timestamp.month}/' if path is None else path
    if not os.path.exists(full_path):
        acc = ''
        for i in full_path.split():
            acc += i
            if not os.path.exists(acc):
                os.mkdir(acc)
        assert acc == full_path, f'Expected full path to be the same as acc, got {acc=} and {full_path=}'
    print(f'[INFO] <log_error()> Logging in file with path: {os.path.join(full_path, f"errors_{timestamp.day}.txt")}')
    with open(os.path.join(full_path, f'errors_{timestamp.day}.txt'), 'a') as f:
        f.write(string + '\n')


def debugprint(*msgs: str, **kws: Any) -> None:
    if not Settings.silence:
        print(*msgs, **kws)

