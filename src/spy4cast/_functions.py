"""
Collection of functions used across the api and for the users convenience
"""
import sys
from typing import Optional, Dict, Any, Union
from .stypes import Month, Region
from time import perf_counter


__all__ = [
    'time_from_here',
    'time_to_here',

    'region2str',
    'season2str',
    'mon2str',
    'str2mon',
    'debugprint',

    '_warning',
    '_error',
    '_debuginfo',
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


def time_from_here() -> float:
    """Function that is supposed to use in conjunctin with `time_to_here`
    to time program parts

    Returns
    -------
        here : float
            Timestamp calcualted with time.perf_counter that can be passed to time_to_here()

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
    return _prev


def time_to_here(here: Optional[float] = None) -> float:
    """Function that is supposed to use in conjunction with `time_from_here`
    to time program parts

    Parameters
    ----------
        here : optional, float
            If passed it will calculate the time since `here`. If not it
            will used a global variable that is set when `time_from_here` is called

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
    now = perf_counter()
    if here is not None:
        return now - here

    global _prev
    if _prev is None:
        raise ValueError(
            'Expected to call time_from_here() before calling time_to_here() or passing a tiemstamp'
        )
    rv = now - _prev
    _prev = None
    return rv


def region2str(region: Region) -> str:
    """Transforms a Region into a string with 2 decimals for the spatial dimension

    Parameters
    ----------
        region : Region
            Region that you wish to transform into a string

    Returns
    -------
        str
            Region fromatted using N (north), W (west), S (south) and E (east)
            for the spacial slice and the first letter of each month of the
            season for the time slice.

    Example
    -------
        >>> s = Region(-10, 10, -100, -80, Month.JAN, Month.FEB, 1870, 2020)
        >>> region2str(s)
        'JF (10.00ºS, 10.00ºN - 100.00ºW, 80.00ºW)'

    See Also
    --------
    stypes.Region
    """
    sufixes: Dict[str, str] = {
        'lat_min': '', 'lat_max': '', 'lon_min': '', 'lon_max': ''
    }
    values: Dict[str, float] = {
        'lat_min': region.lat0, 'lat_max': region.latf,
        'lon_min': region.lon0, 'lon_max': region.lonf
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
    region_str = f'{values["lat_min"]:.02f}{sufixes["lat_min"]}, ' \
             f'{values["lat_max"]:.02f}{sufixes["lat_max"]} - ' \
             f'{values["lon_min"]:.02f}{sufixes["lon_min"]}, ' \
             f'{values["lon_max"]:.02f}{sufixes["lon_max"]}'

    if region.monthf >= region.month0:
        season = ''.join(
            Month(x).name[0] for x in range(region.month0, region.monthf + 1)
        )
    else:
        season = ''.join(
            Month(x).name[0] for x in range(region.month0, Month.DEC + 1)
        ) + ''.join(Month(x).name[0] for x in range(1, region.monthf + 1))

    return f'{season} ({region_str})'


def mon2str(month: Union[Month, int]) -> str:
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


def season2str(month0: Union[Month, int], monthf: Union[Month, int]) -> str:
    """Converts season defined by initial and final month to a season string:
    (Month.JUL, Month.SEP) -> 'JAS'

    Parameters
    ----------
        month0 : Month
            Intial month of the season
        monthf : Month
            Final month of the season

    Returns
    -------
        season : str
            String created with the first letter of the season
    """
    if monthf < month0:
        monthf += 12
    rv = ""
    for i in range(month0, monthf + 1):
        index = (i-1) % 12
        rv += MONTH_TO_STRING[index][0]
    return rv


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


def debugprint(*msgs: Union[str, int], **kws: Any) -> None:
    """Function that only prints if `Settings.silence` is True

    See Also
    --------
    Settings
    """
    from . import Settings
    if not Settings.silence:
        print(*msgs, **kws)

def _warning(msg: str) -> None:
    """Print a warning into the stderr"""
    print(f'[WARNING] {msg}', file=sys.stderr)

def _error(msg: str) -> None:
    """Print an error into the stderr"""
    print(f'[ERROR] {msg}', file=sys.stderr)

def _debuginfo(msg: str, end: Optional[str] = None) -> None:
    """Print an info message into the stdout in debug"""
    debugprint(f'[INFO] {msg}', end=end)
