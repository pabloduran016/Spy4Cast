import builtins
import sys
from typing import Union, TypedDict, Tuple, Dict, Optional
import pandas as pd
import datetime
from enum import auto, IntEnum, IntFlag
from dataclasses import dataclass
import enum_tools


__all__ = ['Slise', 'Color', 'T_FORMAT', 'TimeStamp', 'Month', 'F', 'RDArgs', 'RDArgsDict']


Color = Tuple[float, float, float]
"""Color type. Tuple of 3 floats"""
T_FORMAT = '%d/%m/%Y %H:%M:%S'
"""Time stamp format used internally"""
TimeStamp = Union[pd.Timestamp, datetime.datetime]
"""TimeStamp type. Union of standard library `datetime.datetime` and `pandas.Timestamp`"""

@enum_tools.documentation.document_enum
class Month(IntEnum):
    """Enumaretion for Months.

    Useful to use together with `Slise`

    Example
    -------
        >>> Slise(-20, 20, -100, -60, Month.DEC, Month.FEB, 1870, 1910)
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


def document_dataclass(cls):
    if not getattr(builtins, '__sphinx_build__', False):
        # print('return normal')
        return cls
    # print('return modified')
    for attr, typ in cls.__annotations__.items():
        setattr(cls, attr, None)

    return cls


@document_dataclass
@dataclass
class Slise:
    """Dataclass to create a `Slise`

    A slise is the way of slicing datasets that is convenient to the user and the API

    Note
    ----
        The developers of this API are aware that the correct slpelling for `slice` is with a `c`.
        However, `slice` is a built-in function in python and in the start of the development of this poject was better to use
        `slise`. This is the reason why this class is spelled with `s` (it wouldn't conflict with `slice` right now because
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
    sy: Optional[int] = None
    """Selected year used in methodologies like anom where you can only plot a given year"""

    @classmethod
    def default(cls, month0: int = Month.JAN, monthf: int = Month.DEC, year0: int = 0, yearf: int = 2000,
                sy: Optional[int] = None) -> 'Slise':
        """Alternartive constructor that creates a `Slise` that has the largest spatial region and season possible

        It is useful if you want to create fast a `Slise` where you conly want to modify the initial and final year for example

        Returns
        -------
            Slise(-90, 90, -180, 180, month0, monthf, year0, yearf, sy)"""
        return Slise(-90, 90, -180, 180, month0, monthf, year0, yearf, sy)


# class SliseDict(TypedDict):
#     latitude_min: Union[float, int]
#     latitude_max: Union[float, int]
#     longitude_min: Union[float, int]
#     longitude_max: Union[float, int]
#     initial_month: int
#     final_month: int
#     initial_year: int
#     final_year: int

@enum_tools.documentation.document_enum
class F(IntFlag):
    """Flags"""

    SAVE_DATA = auto()  # doc: Save data. Check documentation for each specific function you pass this flag into
    SAVE_FIG = auto()  # doc: Save fig (calls matplotlib.pyplot.Figure.savefig)
    SILENT_ERRORS = auto()  # doc: If an exception is raise in `Plotter.run`, the program won`t be halted
    SHOW_PLOT = auto()  # doc: Shows the figure (calls matplotlib.pyplot.Figure.show() and matplotlib.pyplot.show if `NOT_HALT` is not set)
    FILTER = auto()  # doc: Perform butterworth filter in preprocesseing. Only for spy4caster.Spy4Caster
    NOT_HALT = auto()  # doc: Not halt the program after showing (don't run matplotlib.pyplot.show)

    def __mul__(self, other: int) -> Union[int, 'F']:
        """Returns normal multiplication unless the value is 1 or 0. In that case it will return the same value or F(0)

        Useful to annulate flags by multiplying them by 0 and setting and unsetting them easily

        Example
        -------
            F.SHOW_PLOT * 0 = F(0)
            F.SHOW_PLOT * 1 = F.SHOW_PLOT
        """
        res = super().__mul__(other)
        if other != 1 and other != 0:
            return res
        return F(res)


ChunkType = Union[int, Tuple[int, ...], Tuple[Tuple[int, ...] ,...], Dict[Union[str, int], int]]
"""Type variable to indicate the types that can be passed into the `chunk` argument in `read_data.ReadData`
"""


@dataclass
class RDArgs():
    """Dataclass to convientely pass arguements for a `ReadData` constructor

    It is mainly used with `spy4cast.Spy4Caster` when you create to ReadData objects
    """
    dataset_dir: Optional[str] = None
    dataset_name: Optional[str] = None
    variable: Optional[str] = None
    chunks: Optional[ChunkType] = None

    def as_dict(self) -> 'RDArgsDict':
        """Method to translate a `RDArgs` into `RDArgsDict`
        """
        return {'dataset_dir': self.dataset_dir,
                'dataset_name': self.dataset_name,
                'variable': self.variable,
                'chunks': self.chunks}


RDArgsDict = TypedDict('RDArgsDict', {'dataset_dir': Optional[str], 'dataset_name': Optional[str],
                                      'variable': Optional[str], 'chunks': Optional[ChunkType]})
"""Typed dict to set as as a type and indicate the arguments that should be passes into `read_data.ReadData`

It is used mainly as an alternative to `RDArgs`
"""



