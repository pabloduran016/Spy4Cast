import os
from typing import Tuple, Optional, TypeVar, Type

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .. import Dataset, Slise, F
from .._functions import slise2str
from .._procedure import _Procedure, _plot_map, _plot_ts, _apply_flags_to_fig
import xarray as xr
import pandas as pd
import numpy as np
import numpy.typing as npt
from . import _PlotType
from ..stypes import Color, Month


__all__ = [
    'Anom',
]


T = TypeVar('T')


class Anom(_Procedure):
    """Procedure to create the anomaly of a `Dataset`

    Parameters
    ----------
        ds : Dataset
            Dataset on to which perform the anomaly

        type : {'map', 'ts'}
            Perform the anomaly and outputting a map (will result in a series of maps)
             or ouputting a timeseries by doing the mean across space.

        st : bool, default=False
            Indicates whether to standarise the anomaly

    See Also
    --------
        Dataset, Clim

    """

    _data: xr.DataArray
    _lat: xr.DataArray
    _lon: xr.DataArray
    _ds: Dataset

    _time_key: str
    _lat_key: str
    _lon_key: str
    _slise: Slise
    _st: bool

    _type: _PlotType

    def __init__(self, ds: Dataset, type: str, st: bool = False):
        self.type = type
        self._ds = ds
        self._slise = ds.slise
        self._st = st

        if self._type == _PlotType.TS:
            self._data = self._ds.data.mean(dim=self._ds._lon_key).mean(dim=self._ds._lat_key)
        elif self._type == _PlotType.MAP:
            self._data = self._ds.data
            self._lat = self._ds.lat
            self._lon = self._ds.lon

            self._lat_key = self._ds._lat_key
            self._lon_key = self._ds._lon_key
        else:
            assert False, 'Unreachable'

        self._data = _anom(self.data, st)
        self._time_key = 'year'
        self._slise.year0 = int(self.time[0])
        self._slise.yearf = int(self.time[-1])

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, val: str):
        if val not in _PlotType.values():
            raise TypeError(
                f'Exected type to be one of '
                f'{{{", ".join(_PlotType.values())}}}, '
                f'but got `{val}`'
            )
        self._type = _PlotType(val)

    @property
    def lat(self) -> xr.DataArray:
        if self._type == _PlotType.TS:
            raise TypeError('Can not acces latitude for type `ts`')
        return self._lat

    @lat.setter
    def lat(self, value: npt.NDArray[np.float32]):
        if self.type == _PlotType.TS:
            raise TypeError('Latitude can not be set on a TS')
        elif self.type == _PlotType.MAP:
            pass
        else:
            assert False, 'Unreachable'
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `lat` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `lat` has to be `float32` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[1]:
            raise ValueError(f'Unmatching shapes for `lat` and `data` variables')

        self._lat = xr.DataArray(value)
        self._data = self.data.assign_coords({self._lat_key: value})

    @property
    def lon(self) -> xr.DataArray:
        if self._type == _PlotType.TS:
            raise TypeError('Can not acces longitude for type `ts`')
        return self._lon

    @lon.setter
    def lon(self, value: npt.NDArray[np.float32]):
        if self.type == _PlotType.TS:
            raise TypeError('Latitude can not be set on a TS')
        elif self.type == _PlotType.MAP:
            pass
        else:
            assert False, 'Unreachable'
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `lon` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `lon` has to be `float32` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[2]:
            raise ValueError(f'Unmatching shapes for `lon` and `data` variables')

        self._lon = xr.DataArray(value)
        self._data = self.data.assign_coords({self._lon_key: value})


    @property
    def time(self) -> xr.DataArray:
        return self._data[self._time_key].astype(np.uint)

    @time.setter
    def time(self, value: npt.NDArray[int]):
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `time` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) != np.dtype('uint'):
            raise ValueError(f'Dtype of `time` has to be `uint` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[0]:
            raise ValueError(f'Unmatching shapes for `time` and `data` variables')

        self._data = self.data.assign_coords({self._time_key: value.astype(np.uint)})

    @property
    def slise(self) -> Slise:
        if hasattr(self, '_slise'):
            return self._slise
        elif self.type == _PlotType.TS:
            # TODO: Replace month0 and monthf with meaninful values
            self._slise = Slise(
                lat0=-90,
                latf=90,
                lon0=-180,
                lonf=180,
                month0=Month.JAN,
                monthf=Month.DEC,
                year0=self.time.values[0],
                yearf=self.time.values[-1],
            )
            return self._slise
        elif self.type == _PlotType.MAP:
            # TODO: Replace month0 and monthf with meaninful values
            self._slise = Slise(
                lat0=self.lat.values[0],
                latf=self.lat.values[-1],
                lon0=self.lon.values[0],
                lonf=self.lon.values[-1],
                month0=Month.JAN,
                monthf=Month.DEC,
                year0=self.time.values[0],
                yearf=self.time.values[-1],
            )
            return self._slise
        else:
            assert False, 'Unreachable'

    @property
    def data(self) -> xr.DataArray:
        return self._data

    @property
    def var(self) -> str:
        if hasattr(self, '_ds'):
            return self._ds.var
        return ''

    @data.setter
    def data(self, value: npt.NDArray[np.float32]):
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `data` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `data` has to be `np.float32` got {np.dtype(value.dtype)}')

        if len(value.shape) == 1:
            self._type = _PlotType.TS
            self._data = xr.DataArray(value, coords={'year': np.arange(len(value))}, dims=['year'])
        elif len(value.shape) == 3:
            self._type = _PlotType.MAP
            self._data = xr.DataArray(value, coords={
                'year': np.arange(value.shape[0]),
                'lat': np.arange(value.shape[1]),
                'lon': np.arange(value.shape[2])
            }, dims=['year', 'lat', 'lon'])
            self._lon_key = 'lon'
            self._lat_key = 'lat'
        else:
            raise ValueError(f'Array to set must be 3-dimensional or 1-dimensional')

        self._time_key = 'year'

    @classmethod
    def from_xrarray(cls, array: xr.DataArray, st: bool = False) -> 'Anom':
        """Function to calculate the anomalies on a xarray DataArray

        The anomaly is the time variable minus the mean across all times of a given point

        Parameters
        ----------
            array : xr.DataArray
                Array to process the anomalies. Must have a dimension called `time`
            st : bool, default=False
                Indicates whether the anomaly should standarized. Divide by the standard deviation

        Raises
        ------
            TypeError
                If array is not an instance of `xr.DataArray`
            ValueError
                If the number of dimension of the array is not either 3 (map) or 1 (time series)

        Returns
        -------
            Anom
                Anom object

        See Also
        --------
        npanom
        """
        obj = Anom.__new__(Anom)
        obj._ds = Dataset.from_xrarray(array)

        obj._data = _anom(array, st)
        obj._time_key = 'year'
        return obj

    def plot(self,
        flags: F = F(0),
        *,
        year: Optional[int] = None,
        cmap: Optional[str] = None,
        color: Optional[Color] = None,
        dir: str = '.',
        name: str = 'anomaly.png'
    ) -> None:
        fig = plt.figure(figsize=(10, 10))
        if self._type == _PlotType.TS:
            if year is not None:
                raise TypeError('year parameter is not valid to plot a time series anomaly')
            if cmap is not None:
                raise TypeError('cmap parameter is not valid to plot a time series anomaly')
            ax = fig.add_subplot()
            _plot_ts(
                time=self.time.values,
                arr=self.data.values,
                ax=ax,
                ylabel=f'{self.var}',
                xlabel='Year',
                color=color,
                xtickslabels=[x for x in self.time.values[::1]],
            )
            fig.suptitle(
                f'Time series of {self.var} ({slise2str(self.slise)})',
                fontweight='bold'
            )
        elif self._type == _PlotType.MAP:
            if color is not None:
                raise TypeError('Color parameter is not valid to plot a map anomaly')
            if year is None:
                raise TypeError(f'Year is a required argument for plotting an anomaly map')
            ax = fig.add_subplot(projection=ccrs.PlateCarree())
            _plot_map(
                arr=self.data.sel({self._time_key: year}).values,
                lat=self.lat,
                lon=self.lon,
                fig=fig,
                ax=ax,
                cmap=(cmap if cmap is not None else 'jet'),
            )
            fig.suptitle(
                f'Map of {self.var} ({slise2str(self.slise)})',
                fontweight='bold'
            )
        else:
            assert False, 'Unreachable'

        path = os.path.join(dir, name)
        _apply_flags_to_fig(
            fig, path, F(flags)
        )

    @classmethod
    def load(cls: Type[T], prefix: str, dir: str = '.', *, type: str = None, **kwargs) -> T:
        if len(kwargs) != 0:
            raise TypeError('Only accepetd kwarg `type` accepted for load method')
        if type is None:
            raise TypeError('`type` is a required kwarg')
        return super().load(prefix, dir, type=type)

    @property
    def var_names(self) -> Tuple[str, ...]:
        if self._type == _PlotType.TS:
            return ('data', 'time')
        elif self._type == _PlotType.MAP:
            return ('data', 'time', 'lat', 'lon')
        else:
            assert False, 'Unreachable'


def _anom(
        array: xr.DataArray, st: bool = False
) -> xr.DataArray:
    # print(f'[INFO] <meteo.Meteo.anom()> called, st: {st}')
    if not isinstance(array, xr.DataArray):
        raise TypeError(f"Invalid type for array: {type(array)}")

    assert 'time' in array.dims, 'Cant\'t recognise time key in array'
    # List of month values
    months_set = set(array.groupby('time.month').groups.keys())
    nm = len(months_set)
    months = array['time.month'][:nm].data
    # Create index to reshape time variab le
    ind = pd.MultiIndex.from_product(
        (array.time[nm - 1::nm]['time.year'].data, months),
        names=('year', 'month')
    )
    if len(array.time) != len(ind):
        raise ValueError('Unkpected time variable. Try slicing the dataset')
    if len(array.shape) == 3:  # 3d array
        # Reshape time variable
        lat_key = 'latitude' if 'latitude' in array.dims else 'lat'
        lon_key = 'longitude' if 'longitude' in array.dims else 'lon'
        assert lat_key in array.dims and lon_key in array.dims,\
            'Can\'t recognise keys'
        arr = array.assign_coords(time=('time', ind))
        # arr must be a DataArray with dims=(months, year, lat, lon)
        a = arr.groupby('year').mean()
        b: xr.DataArray = a - a.mean('year')
        if st:
            # print('[INFO] <meteo.Meteo.anom()> standarzing')
            rv: xr.DataArray = b / b.std()
            return rv
        return b

    elif len(array.shape) == 1:  # time series
        assert 'latitude' not in array.dims and 'longitude' not in array.dims,\
            'Unidimensional arrays time must be the only dimension'
        arr = array.assign_coords(
            time=('time', ind)
        ).unstack('time').transpose('year', 'month')
        a = arr.mean('month')
        b = a - a.mean('year')
        if st:
            # print('[INFO] <meteo.Meteo.anom()> standarzing')
            rv = b / b.std()
            return rv
        return b
    else:
        raise ValueError(
            'Invalid dimensions of array from anom methodology'
        )


def npanom(
    array: npt.NDArray[np.float32],
    axis: int = 0,
    st: bool = False
) -> npt.NDArray[np.float32]:

    """Function to calculate the anomalies on a numpy array

    The anomaly is the time variable minus the mean across all times
    of a given point

    Parameters
    ----------
        array : npt.NDArray[np.float32]
            Array to process the anomalies.
        axis : int, default=0
            Axis where to perform teh anomaly, ussually the time axis
        st : bool, default=False
            Indicates whether the anomaly should standarized.
            Divide by the standard deviation

    See Also
    --------
    anom
    """
    b: npt.NDArray[np.float32] = array - array.mean(axis=axis)
    if st:
        rv: npt.NDArray[np.float32] = b / b.std(axis=axis)
        return rv
    return b
