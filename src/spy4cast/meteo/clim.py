import os
from typing import Type, Tuple, Optional, Union, Any

from matplotlib import pyplot as plt

from . import _PlotType, _get_type
from .. import Slise, Dataset, Month, F
from .._functions import slise2str
from .._procedure import _Procedure, _apply_flags_to_fig, _plot_ts, _plot_map, _calculate_figsize, MAX_WIDTH, MAX_HEIGHT
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
import numpy.typing as npt
import pandas as pd

from ..stypes import Color


__all__ = [
    'Clim',
]


class Clim(_Procedure, object):
    """Procedure to create the climatology of a dataset

    Parameters
    ----------
        ds : spy4cast.dataset.Dataset
            spy4cast.dataset.Dataset onto which perform the anomaly

        type : {'map', 'ts'}
            Perform the climatology and outputting a map by doing the mean across time
             or ouputting a timeseries by doing the mean across space.

    See Also
    --------
        Dataset, Clim

    """

    _ds: Dataset
    _data: xr.DataArray
    _lat: xr.DataArray
    _lon: xr.DataArray

    _time_key: str
    _lat_key: str
    _lon_key: str
    _slise: Slise

    _type: _PlotType

    def __init__(self, ds: Dataset, type: str):
        self.type = _get_type(type)
        self._ds = ds
        self._slise = ds.slise
        self._data = ds.data

        if self._type == _PlotType.TS:
            self._data = self._data.mean(dim=self._ds._lon_key).mean(dim=self._ds._lat_key)
            self._data = _clim(self._data, dim='month')
            self._time_key = 'year'
            self._time = self._data[self._time_key]
        elif self._type == _PlotType.MAP:
            self._data = _clim(self._data)
            self._lat = self._ds.lat
            self._lon = self._ds.lon
        else:
            assert False, 'Unreachable'

    @property
    def type(self) -> _PlotType:
        return self._type

    @type.setter
    def type(self, val: _PlotType) -> None:
        self._type = val

    @property
    def lat(self) -> xr.DataArray:
        if self._type == _PlotType.TS:
            raise TypeError('Can not acces latitude for type `ts`')
        return self._lat

    @lat.setter
    def lat(self, value: npt.NDArray[np.float32]) -> None:
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

        if len(value) != self.data.shape[0]:
            raise ValueError(f'Unmatching shapes for `lat` and `data` variables')

        self._lat = xr.DataArray(value)
        self._data = self.data.assign_coords({self._lat_key: value})

    @property
    def lon(self) -> xr.DataArray:
        if self._type == _PlotType.TS:
            raise TypeError('Can not acces longitude for type `ts`')
        return self._lon

    @lon.setter
    def lon(self, value: npt.NDArray[np.float32]) -> None:
        if self.type == _PlotType.TS:
            raise TypeError('Longitude can not be set on a TS')
        elif self.type == _PlotType.MAP:
            pass
        else:
            assert False, 'Unreachable'
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `lon` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `lon` has to be `float32` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[1]:
            raise ValueError(f'Unmatching shapes for `lon` and `data` variables')

        self._lon = xr.DataArray(value)
        self._data = self.data.assign_coords({self._lon_key: value})

    @property
    def time(self) -> xr.DataArray:
        if self.type == _PlotType.MAP:
            raise TypeError('Time can not be get from a Map')
        elif self.type == _PlotType.TS:
            pass
        else:
            assert False, 'Unreachable'
        return self._data[self._time_key]

    @time.setter
    def time(self, value: npt.NDArray[Union[np.uint, np.datetime64]]) -> None:
        if self.type == _PlotType.MAP:
            raise TypeError('Time can not be set on a Map')
        elif self.type == _PlotType.TS:
            pass
        else:
            assert False, 'Unreachable'
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `time` has to be `np.ndarray` got {type(value)}')
        if (
                np.dtype(value.dtype) != np.dtype('uint') and
                np.dtype(value.dtype) != np.dtype('datetime64[ns]')
        ):
            raise ValueError(f'Dtype of `time` has to be `uint` or `datetime64[ns]` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[0]:
            raise ValueError(f'Unmatching shapes for `time` and `data` variables')

        self._data = self.data.assign_coords({self._time_key: value})

    @property
    def slise(self) -> Slise:
        if hasattr(self, '_slise'):
            return self._slise
        elif hasattr(self, '_ds'):
            return self._ds.slise
        elif self._type == _PlotType.TS:
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
        elif self._type == _PlotType.MAP:
            self._slise = Slise(
                lat0=self.lat.values[0],
                latf=self.lat.values[-1],
                lon0=self.lon.values[0],
                lonf=self.lon.values[-1],
                month0=Month.JAN,
                monthf=Month.DEC,
                year0=0,
                yearf=1000,
            )
            return self._slise
        else:
            assert False, 'Unreachable'

    @property
    def var(self) -> str:
        if hasattr(self, '_ds'):
            return self._ds.var
        return ''

    @property
    def data(self) -> xr.DataArray:
        return self._data

    @data.setter
    def data(self, value: npt.NDArray[np.float32]) -> None:
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `data` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `data` has to be `np.float32` got {np.dtype(value.dtype)}')

        if len(value.shape) == 1:
            self._type = _PlotType.TS
            self._data = xr.DataArray(value, coords={'time': np.arange(len(value))}, dims=['time'])
            self._time_key = 'time'
        elif len(value.shape) == 2:
            self._type = _PlotType.MAP
            self._data = xr.DataArray(value, coords={
                'lat': np.arange(value.shape[0]),
                'lon': np.arange(value.shape[1])
            }, dims=['lat', 'lon'])
            self._lon_key = 'lon'
            self._lat_key = 'lat'
        else:
            raise ValueError(f'Array to set must be 2-dimensional or 1-dimensional')

    def plot(
        self,
        flags: F = F(0),
        *,
        cmap: Optional[str] = None,
        color: Optional[Color] = None,
        dir: str = '.',
        name: str = 'clim.png'
    ) -> None:
        if self._type == _PlotType.TS:
            fig = plt.figure(figsize=_calculate_figsize(None, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
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
            nlat, nlon = len(self.lat), len(self.lon)
            fig = plt.figure(figsize=_calculate_figsize(nlat / nlon, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
            if color is not None:
                raise TypeError('Color parameter is not valid to plot a map anomaly')
            ax = fig.add_subplot(projection=ccrs.PlateCarree())
            _plot_map(
                arr=self.data.values,
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
    def load(cls: Type['Clim'], prefix: str, dir: str = '.', *, type: Optional[str] = None, **attrs: Any) -> 'Clim':
        if len(attrs) != 0:
            raise TypeError('Only accepetd kwarg `type` accepted for load method')
        if type is None:
            raise TypeError('`type` is a required kwarg')
        return super().load(prefix, dir, type=_get_type(type))

    @property
    def var_names(self) -> Tuple[str, ...]:
        if self._type == _PlotType.TS:
            return 'data', 'time'
        elif self._type == _PlotType.MAP:
            return 'data', 'lat', 'lon'
        else:
            assert False, 'Unreachable'


def _clim(array: xr.DataArray, dim: str = 'time') -> xr.DataArray:
    """Function that performs the climatology of a xarray Dataset

    The climatology is the average across a given axis

    Parameters
    ----------
        array : xr.DataArray
            Xarray DataArray where you wish to perform the climatology

        dim : str, default='time'
            Dimension where the climatology is going to be performed on

    See Also
    --------
    plotters.ClimerTS, plotters.ClimerMap

    Raises
    ------
        TypeError
            If array is not an instance of `xr.DataArray`
        ValueError
            If dim is not `month`, `time` or `year`
    """
    if not isinstance(array, xr.DataArray):
        raise TypeError(f"Expected type xarray.DataArray, got {type(array)}")
    if dim == 'year' or dim == 'month':
        months = list(array.groupby('time.month').groups.keys())  # List of month values
        nm = len(months)
        # Create index to reshape time variable
        ind = pd.MultiIndex.from_product(
            (months, array.time[nm - 1::nm].data),
            names=('month', 'year')
        )
        # Reshape time variable
        assert len(array.shape) == 2 or len(array.shape) == 1,\
            f'Clim implemented only for 1 and 2 dimensional arrays, for now'
        arr = array.assign_coords(
            time=('time', ind)
        ).unstack('time').transpose('year', 'month')
        rv: xr.DataArray = arr.mean(dim=dim)
    elif dim == 'time':  # Apply across year and month
        assert 'time' in array.dims
        rv = array.mean(dim=dim)
    else:
        raise ValueError(f'Invalid dim {dim}')
    return rv
