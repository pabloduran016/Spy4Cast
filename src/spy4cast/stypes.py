import builtins
from typing import Union, Tuple, Dict, Optional, TypeVar, cast, Any, List

import numpy as np
import pandas as pd
import datetime
from enum import auto, IntEnum
from dataclasses import dataclass
import numpy.typing as npt


__all__ = [
    'Region',
    'Color',
    'T_FORMAT',
    'TimeStamp',
    'Month',
]

Color = Tuple[float, float, float]
"""Color type. Tuple of 3 floats"""
T_FORMAT = '%d/%m/%Y %H:%M:%S'
"""Time stamp format used internally"""
TimeStamp = Union[pd.Timestamp, datetime.datetime]
"""TimeStamp type. Union of standard library `datetime.datetime` and `pandas.Timestamp`"""


T = TypeVar('T')
def _document_enum(cls: T) -> T:
    try:
        import enum_tools
        return cast(T, enum_tools.documentation.document_enum(cast(Any, cls)))
    except ImportError:
        return cls


@_document_enum
class Month(IntEnum):
    """Enumeration for Months.

    Useful to use together with `Region`

    Example
    -------
        >>> Region(-20, 20, -100, -60, Month.DEC, Month.FEB, 1870, 1910)
    """
    JAN = auto()
    FEB = auto()
    MAR = auto()
    APR = auto()
    MAY = auto()
    JUN = auto()
    JUL = auto()
    AUG = auto()
    SEP = auto()
    OCT = auto()
    NOV = auto()
    DEC = auto()


@dataclass
class Region:
    """Dataclass to create a `Region`

    A region is the way of slicing datasets that is convenient to the user and the API

    Note
    ----
        The developers of this API are aware that the correct spelling for `slice` is with a `c`.
        However, `slice` is a built-in function in python and in the start of the development of this poject was better to use
        `region`. This is the reason why this class is spelled with `s` (it wouldn't conflict with `slice` right now because
        it is capitalised, but it looks good enough, right?)
    """

    lat0: Union[float, int]
    """Minimum latitude"""
    latf: Union[float, int]
    """Maximum latitude"""
    lon0: Union[float, int]
    """Minimum longitud"""
    lonf: Union[float, int]
    """Maximum longitud"""
    month0: Union[Month, int]
    """Starting month of the season to select (included)"""
    monthf: Union[Month, int]
    """Ending month of the season to select (included)"""
    year0: int
    """Starting year of the period to select (included)"""
    yearf: int
    """Ending year of the period to select (included)"""
    sy: Optional[int] = None  # Should not be used
    """DEPRECATED. Selected year used in methodologies like anom where you can only plot a given year"""

    @classmethod
    def default(cls, month0: int = Month.JAN, monthf: int = Month.DEC, year0: int = 0, yearf: int = 2000,
                sy: Optional[int] = None) -> 'Region':
        """Alternartive constructor that creates a `Region` that has the largest spatial region and season possible

        It is useful if you want to create fast a `Region` where you conly want to modify the initial and final year for example

        Returns
        -------
            Region(-90, 90, -180, 180, month0, monthf, year0, yearf, sy)"""
        return Region(-90, 90, -180, 180, month0, monthf, year0, yearf, sy)

    @classmethod
    def from_numpy(cls, arr: npt.NDArray[Union[np.float32, np.uint]]) -> 'Region':
        """Alternative constructor for region:
            lat0, latf, lon0, lonf, month0, monthf, year0, yearf, sy
        """
        attrs: List[float] = [x for x in arr]
        if len(attrs) != 9:
            raise TypeError(f'Invalid dimensions for array expected 9 fields, got {len(attrs)}')

        lat0, latf, lon0, lonf, month0, monthf, year0, yearf, sy = attrs

        return Region(
            lat0=float(lat0),
            latf=float(latf),
            lon0=float(lon0),
            lonf=float(lonf),
            month0=Month(int(month0)),
            monthf=Month(int(monthf)),
            year0=int(year0),
            yearf=int(yearf),
            sy=(int(sy) if not np.isnan(sy) else None),
        )

    def as_numpy(self) -> npt.NDArray[Union[np.float32, np.uint]]:
        """Converts region to np array with fields:
            lat0, latf, lon0, lonf, month0, monthf, year0, yearf, sy
        """
        return np.array([
            self.lat0,
            self.latf,
            self.lon0,
            self.lonf,
            self.month0,
            self.monthf,
            self.year0,
            self.yearf,
            (self.sy if self.sy is not None else np.nan)
        ])


CLS = TypeVar('CLS')


def _document_dataclass(cls: CLS) -> CLS:
    if not getattr(builtins, '__sphinx_build__', False):
        # print('return normal')
        return cls
    # print('return modified')
    for attr, typ in cls.__annotations__.items():
        setattr(cls, attr, None)
    return cls


_document_dataclass(Region)
# class RegionDict(TypedDict):
#     latitude_min: Union[float, int]
#     latitude_max: Union[float, int]
#     longitude_min: Union[float, int]
#     longitude_max: Union[float, int]
#     initial_month: int
#     final_month: int
#     initial_year: int
#     final_year: int




