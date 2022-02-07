from typing import Union, Any, TypedDict, Callable, Tuple, Dict, List, Optional
import xarray as xr
import pandas as pd
import datetime
from enum import Enum, auto, IntEnum, IntFlag
from dataclasses import dataclass


__all__ = ['DsType', 'RegionType', 'Slise', 'Color', 'T_FORMAT', 'TimeStamp', 'MeteoFunction', 'TEST', 'REC',
           'ARGS', 'KWARGS', 'Month', 'Methodology', 'PltType', 'STR2PLTTYPE', 'METHODOLOGY2STR', 'STR2METHODOLOGY', 'F',
           'RDArgs', 'RDArgsDict']


class DsType(Enum):
    Preprocessed = auto()
    New = auto()


# REGION_TYPE = tuple[float, float, float, float, int, int]
Color = tuple[float, float, float]
T_FORMAT = '%d/%m/%Y %H:%M:%S'
TimeStamp = Union[pd.Timestamp, datetime.datetime]


class RegionType(TypedDict):
    latitude_min: Union[float, int]
    latitude_max: Union[float, int]
    longitude_min: Union[float, int]
    longitude_max: Union[float, int]


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


class SliseDict(TypedDict):
    latitude_min: Union[float, int]
    latitude_max: Union[float, int]
    longitude_min: Union[float, int]
    longitude_max: Union[float, int]
    initial_month: int
    final_month: int
    initial_year: int
    final_year: int


class MeteoFunction:
    def __call__(self, arr: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
        pass


TEST = Callable[[], Union[bool, Tuple[int, int]]]
REC = Callable[[], Union[bool, Tuple[int, int]]]
ARGS = Union[Tuple[Any, ...], List[Any]]
KWARGS = Dict[str, Any]


class Methodology(Enum):
    ANOM = auto()
    CLIM = auto()
    SPY4CAST = auto()


class PltType(Enum):
    MAP = auto()
    TS = auto()


STR2PLTTYPE: Dict[str, PltType] = {
    'map': PltType.MAP,
    'ts': PltType.TS,
}

assert len(Methodology) == 3, 'Exhaustive handling of Methodology in METHODOLOGY2STR'
METHODOLOGY2STR: Dict[Methodology, str] = {
    Methodology.ANOM: 'anom',
    Methodology.CLIM: 'clim',
    Methodology.SPY4CAST: 'spy4cast',
}
STR2METHODOLOGY: Dict[str, Methodology] = {
    'anom': Methodology.ANOM,
    'clim': Methodology.CLIM,
    'spy4cast': Methodology.SPY4CAST,
}

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

