from typing import Union, TypedDict, Tuple, Dict, Optional
import pandas as pd
import datetime
from enum import auto, IntEnum, IntFlag
from dataclasses import dataclass


__all__ = ['Slise', 'Color', 'T_FORMAT', 'TimeStamp', 'Month', 'F', 'RDArgs', 'RDArgsDict']


Color = Tuple[float, float, float]
T_FORMAT = '%d/%m/%Y %H:%M:%S'
TimeStamp = Union[pd.Timestamp, datetime.datetime]


class Month(IntEnum):
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
class Slise:
    lat0: Union[float, int]
    latf: Union[float, int]
    lon0: Union[float, int]
    lonf: Union[float, int]
    month0: Union[Month, int]
    monthf: Union[Month, int]
    year0: int
    yearf: int
    sy: Optional[int] = None

    @classmethod
    def default(cls, month0: int = Month.JAN, monthf: int = Month.DEC, year0: int = 0, yearf: int = 2000,
                sy: Optional[int] = None) -> 'Slise':
        """Returns: Slise(-90, 90, -180, 180, month0, monthf, year0, yearf, sy)"""
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


class F(IntFlag):
    SAVE_DATA = auto()
    SAVE_FIG = auto()
    SILENT_ERRORS = auto()
    SHOW_PLOT = auto()
    FILTER = auto()  # Perform butterworth filter in preprocesseing
    NOT_HALT = auto()  # Not halt the program after showing (running plt.show)


ChunkType = Union[int, Tuple[int, ...], Tuple[Tuple[int, ...] ,...], Dict[Union[str, int], int]]


@dataclass
class RDArgs():
    dataset_dir: Optional[str] = None
    dataset_name: Optional[str] = None
    variable: Optional[str] = None
    chunks: Optional[ChunkType] = None

    def as_dict(self) -> 'RDArgsDict':
        return {'dataset_dir': self.dataset_dir,
                'dataset_name': self.dataset_name,
                'variable': self.variable,
                'chunks': self.chunks}


RDArgsDict = TypedDict('RDArgsDict', {'dataset_dir': Optional[str], 'dataset_name': Optional[str],
                                      'variable': Optional[str], 'chunks': Optional[ChunkType]})

