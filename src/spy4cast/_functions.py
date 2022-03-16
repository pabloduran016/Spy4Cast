"""
Collection of functions used across the api and for the users convenience
"""
from typing import Optional, Dict, Any
from .stypes import Month, Slise
from time import perf_counter


__all__ = [
    'time_from_here',
    'time_to_here',

    'slise2str',
    'mon2str',
    'str2mon',
    'debugprint',
]


# Array with the month indices
MONTH_TO_STRING = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]
"""Array to translate from Month to string"""

VALID_MONTHS = list(filter(
      lambda x: not x.startswith('_'), Month.__dict__.keys()))
"""Array indicating valid months that can be passed to `str2mon`"""


_prev: Optional[float] = None


def time_from_here() -> None:
    """Function that is supposed to use in conjunctin with `time_to_here`
    to time program parts

    Example
    -------
        >>> print(
        ...     'Iterating through the first 100 million ints and adding their'
        ...     ' square to an array ', end=''
        ... )
        ... arr = []
        ... time_from_here()
        ... for i in range(100_000_000):
        ...     arr.append(i ** 2)
        ...
        ... print(f'took: {time_to_here():.02f} seconds')
        Iterating through the first 100 million ints and adding their square
        to an array took: 30.27 seconds
    """
    global _prev
    _prev = perf_counter()
    return


def time_to_here() -> float:
    """Function that is supposed to use in conjunctin with `time_from_here`
    to time program parts

    Example
    -------
        >>> print(
        ...     'Iterating through the first 100 million ints and adding their'
        ...     ' square to an array ', end=''
        ... )
        ... arr = []
        ... time_from_here()
        ... for i in range(100_000_000):
        ...     arr.append(i ** 2)
        ...
        ... print(f'took: {time_to_here():.02f} seconds')
        Iterating through the first 100 million ints and adding their square
        to an array took: 30.27 seconds
    """
    global _prev
    if _prev is None:
        raise ValueError(
            'Expected to call time_from_here() before calling time_to_here()'
        )
    rv = perf_counter() - _prev
    _prev = None
    return rv


def slise2str(slise: Slise) -> str:
    """Transforms a Slise into a string

    Parameters
    ----------
        slise : Slise
            Slise that you wish to transform into a string

    Returns
    -------
        str
            Slise fromatted using N (north), W (west), S (south) and E (east)
            for the spacial slice and the first letter of each month of the
            season for the time slice.

    Example
    -------
        >>> s = Slise(-10, 10, -100, -80, Month.JAN, Month.FEB, 1870, 2020)
        >>> slise2str(s)
        '10ºS, 100ºW - 10ºN, 80ºW'

    See Also
    --------
    stypes.Slise
    """
    sufixes: Dict[str, str] = {
        'lat_min': '', 'lat_max': '', 'lon_min': '', 'lon_max': ''
    }
    values: Dict[str, float] = {
        'lat_min': slise.lat0, 'lat_max': slise.latf,
        'lon_min': slise.lon0, 'lon_max': slise.lonf
    }
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
    region = f'{values["lat_min"]}{sufixes["lat_min"]}, ' \
             f'{values["lon_min"]}{sufixes["lon_min"]} - ' \
             f'{values["lat_max"]}{sufixes["lat_max"]}, ' \
             f'{values["lon_max"]}{sufixes["lon_max"]}'

    if slise.monthf >= slise.month0:
        season = ''.join(
            Month(x).name[0] for x in range(slise.month0, slise.monthf + 1)
        )
    else:
        season = ''.join(
            Month(x).name[0] for x in range(slise.month0, Month.DEC + 1)
        ) + ''.join(Month(x).name[0] for x in range(1, slise.monthf + 1))

    return f'{season} ({region})'


def mon2str(month: Month) -> str:
    """Function that turns `Month` enum value into a string

    Parameters
    ----------
        month : Month

    Returns
    -------
        str
            String belonging to MONTH_TO_STRING that orresponds to the month
            inputed

    See Also
    --------
    Month, MONTH_TO_STRING, str2month
    """
    if not 1 <= month <= 12:
        raise ValueError(f'Expected month number from 1 to 12, got {month}')
    # print('[WARNING] <month_to_string()> Changing month indexes!!!!!!!!!!!!')
    return MONTH_TO_STRING[month - 1]


def str2mon(month: str) -> Month:
    """Function that turns a string ingo a `Month` enum value

    Parameters
    ----------
        month : str

    Returns
    -------
        Month
            Month belonging to `Month` enum that
            corresponds to the month inputed

    See Also
    --------
    Month, MONTH_TO_STRING, mon2str
    """
    month = month.upper()[:3]
    if not hasattr(Month, month):
        raise ValueError(
            f'Month not known, got {month}, '
            f'valid values: {VALID_MONTHS}'
        )
    return Month[month]


def debugprint(*msgs: str, **kws: Any) -> None:
    """Function that only prints if `Settings.silence` is True

    See Also
    --------
    Settings
    """
    from . import Settings
    if not Settings.silence:
        print(*msgs, **kws)
