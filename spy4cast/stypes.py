from typing import Union, TypedDict, Tuple, Dict, Optional
import pandas as pd
import datetime
from enum import auto, IntEnum, IntFlag
from dataclasses import dataclass


__all__ = ['Slise', 'Color', 'T_FORMAT', 'TimeStamp', 'Month', 'F', 'RDArgs', 'RDArgsDict']


Color = Tuple[float, float, float]
T_FORMAT = '%d/%m/%Y %H:%M:%S'
TimeStamp = Union[pd.Timestamp, datetime.datetime]


@dataclass
class Slise:
    latitude_min: Union[float, int]
    latitude_max: Union[float, int]
    longitude_min: Union[float, int]
    longitude_max: Union[float, int]
    initial_month: int
    final_month: int
    initial_year: int
    final_year: int
    selected_year: Optional[int] = None

    @classmethod
    def default(cls, initial_month: int = 1, final_month: int = 12, initial_year: int = 0, final_year: int = 2000,
                selected_year: Optional[int] = None) -> 'Slise':
        """Returns: Slise(-90, 90, -180, 180, 1, 12, 1870, 2000, None)"""
        return Slise(-90, 90, -180, 180, initial_month, final_month, initial_year, final_year, selected_year)


# class SliseDict(TypedDict):
#     latitude_min: Union[float, int]
#     latitude_max: Union[float, int]
#     longitude_min: Union[float, int]
#     longitude_max: Union[float, int]
#     initial_month: int
#     final_month: int
#     initial_year: int
#     final_year: int


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


class F(IntFlag):
    SAVE_DATA = auto()
    SAVE_FIG = auto()
    TESTING = auto()
    SILENT_ERRORS = auto()
    SHOW_PLOT = auto()

    @staticmethod
    def checkf(f: 'F', other: Union[int, 'F']) -> bool:
        return (other & f) == f


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

