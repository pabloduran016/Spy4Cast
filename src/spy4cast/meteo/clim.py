import os
from typing import Type, Tuple, Optional, Union, Any, Sequence, Literal

from matplotlib import pyplot as plt

from . import PlotType, _get_type
from .. import Region, Dataset, Month
from .._functions import region2str
from .._procedure import _Procedure, _apply_flags_to_fig, _plot_ts, _plot_map, _calculate_figsize, MAX_WIDTH, MAX_HEIGHT
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
import numpy.typing as npt
import cartopy.util

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
            spy4cast.dataset.Dataset onto which perform the climatology

        type : 'map' or 'ts'
            Perform the climatology and outputing a map by doing the mean across time
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
    _region: Region

    _type: PlotType

    def __init__(self, ds: Dataset, type: Literal["map", "ts"]):
        self.type = _get_type(type)
        self._ds = ds
        self._region = ds.region

        if self._type == PlotType.TS:
            self._data = _clim(ds.data.mean(ds._lon_key).mean(ds._lat_key), dim='year')
            self._time_key = 'year'
            self._time = self._data[self._time_key]
        elif self._type == PlotType.MAP:
            self._data = _clim(ds.data)
            self._lon_key = self._ds._lon_key
            self._lat_key = self._ds._lat_key
            self._lat = self._ds.lat
            self._lon = self._ds.lon
        else:
            assert False, 'Unreachable'

    @property
    def ds(self) -> Dataset:
        """Dataset introduced"""
        return self._ds

    @property
    def type(self) -> PlotType:
        """Type of climatology passed in initialization

        Returns
        -------
            PlotType
        """
        return self._type

    @type.setter
    def type(self, val: PlotType) -> None:
        self._type = val

    @property
    def lat(self) -> xr.DataArray:
        """Array of latitude values of the dataset passes in initialization if type is `map`

        Returns
        -------
            xarray.DataArray
        """
        if self._type == PlotType.TS:
            raise TypeError('Can not acces latitude for type `ts`')
        return self._lat

    @lat.setter
    def lat(self, value: npt.NDArray[np.float32]) -> None:
        if self.type == PlotType.TS:
            raise TypeError('Latitude can not be set on a TS')
        elif self.type == PlotType.MAP:
            pass
        else:
            assert False, 'Unreachable'
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `lat` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) == np.dtype('float64'):
            value = value.astype(np.float32)
        elif np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `lat` has to be `float32` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[0]:
            raise ValueError(f'Unmatching shapes for `lat` and `data` variables')

        self._lat = xr.DataArray(value)
        self._data = self.data.assign_coords({self._lat_key: value})

    @property
    def lon(self) -> xr.DataArray:
        """Array of longitude values of the dataset passes in initialization if type is `map`

        Returns
        -------
            xarray.DataArray
        """
        if self._type == PlotType.TS:
            raise TypeError('Can not acces longitude for type `ts`')
        return self._lon

    @lon.setter
    def lon(self, value: npt.NDArray[np.float32]) -> None:
        if self.type == PlotType.TS:
            raise TypeError('Longitude can not be set on a TS')
        elif self.type == PlotType.MAP:
            pass
        else:
            assert False, 'Unreachable'
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `lon` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) == np.dtype('float64'):
            value = value.astype(np.float32)
        elif np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `lon` has to be `float32` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[1]:
            raise ValueError(f'Unmatching shapes for `lon` and `data` variables')

        self._lon = xr.DataArray(value)
        self._data = self.data.assign_coords({self._lon_key: value})

    @property
    def time(self) -> xr.DataArray:
        """Array of years of the time variable

        Returns
        -------
            xarray.DataArray
        """
        if self.type == PlotType.MAP:
            raise TypeError('Con not get time from a map')
        elif self.type == PlotType.TS:
            pass
        else:
            assert False, 'Unreachable'
        return self._data[self._time_key].astype(np.uint)

    @time.setter
    def time(self, value: npt.NDArray[Union[np.uint, np.datetime64]]) -> None:
        if self.type == PlotType.MAP:
            raise TypeError('Time can not be set on a Map')
        elif self.type == PlotType.TS:
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
    def region(self) -> Region:
        """Region applied to the matrix.

        Returns
        -------
            spy4cast.stypes.Region

        Note
        ----
            If type is `ts` and initilization from ds was not run then a default time and region region is returned

        Note
        ----
            If type is `map` and initilization from ds was not run then a default time region is returned
        """
        if hasattr(self, '_region'):
            return self._region
        elif hasattr(self, '_ds'):
            return self._ds.region
        elif self._type == PlotType.TS:
            self._region = Region(
                lat0=-90,
                latf=90,
                lon0=-180,
                lonf=180,
                month0=Month.JAN,
                monthf=Month.DEC,
                year0=self.time.values[0],
                yearf=self.time.values[-1],
            )
            return self._region
        elif self._type == PlotType.MAP:
            self._region = Region(
                lat0=self.lat.values[0],
                latf=self.lat.values[-1],
                lon0=self.lon.values[0],
                lonf=self.lon.values[-1],
                month0=Month.JAN,
                monthf=Month.DEC,
                year0=0,
                yearf=1000,
            )
            return self._region
        else:
            assert False, 'Unreachable'

    @property
    def var(self) -> str:
        """Variable name

        Returns
        -------
            str
        """
        if hasattr(self, '_ds'):
            return self._ds.var
        return ''

    @property
    def data(self) -> xr.DataArray:
        """Data Matrix

        Returns
        -------
            xarray.DataArray
        """
        return self._data

    @data.setter
    def data(self, value: npt.NDArray[np.float32]) -> None:
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `data` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `data` has to be `np.float32` got {np.dtype(value.dtype)}')

        if self.type == PlotType.TS:
            if len(value.shape) != 1:
                raise ValueError(f'Expected data to be one-dimensional for time series. Got shape {value.shape}')
            self._data = xr.DataArray(value, coords={'time': np.arange(len(value))}, dims=['time'])
            self._time_key = 'time'
        elif self.type == PlotType.MAP:
            if len(value.shape) != 2:
                raise ValueError(f'Expected data to be two-dimensional for map. Got shape {value.shape}')
            self._data = xr.DataArray(value, coords={
                'lat': np.arange(value.shape[0]),
                'lon': np.arange(value.shape[1])
            }, dims=['lat', 'lon'])
            self._lon_key = 'lon'
            self._lat_key = 'lat'
        else:
            assert False, 'Unreachable'

    def plot(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        cmap: Optional[str] = None,
        color: Optional[Color] = None,
        dir: str = '.',
        name: str = 'clim.png',
        levels: Optional[list] = None,
        ticks: Optional[list] = None,
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot the climatology map or time series

        Parameters
        ----------
        save_fig
            Saves the fig in with `dir` / `name` parameters
        show_plot
            Shows the plot
        halt_program
            Only used if `show_plot` is `True`. If `True` shows the plot if plt.show
            and stops execution. Else uses fig.show and does not halt program
        cmap
            Colormap of the `map` types
        color
            Color of the line for `ts` types
        dir
            Directory to save fig if `save_fig` is `True`
        name
            Name of the fig saved if `save_fig` is `True`

        Returns
        -------
        plt.Figure
            Figure object from matplotlib

        Sequence[plt.Axes]
            Tuple of axes in figure
        """
        if self._type == PlotType.TS:
            fig = plt.figure(figsize=_calculate_figsize(None, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
            if cmap is not None:
                raise TypeError('cmap parameter is not valid to plot a time series climatology')
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
                f'Time series of {self.var} ({region2str(self.region)})',
                fontweight='bold'
            )
        elif self._type == PlotType.MAP:
            nlat, nlon = len(self.lat), len(self.lon)
            fig = plt.figure(figsize=_calculate_figsize(nlat / nlon, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
            if color is not None:
                raise TypeError('Color parameter is not valid to plot a map climatology')
            ax = fig.add_subplot(projection=ccrs.PlateCarree(0 if self.region.lon0 < self.region.lonf else 180))
            _plot_map(
                arr=self.data.values,
                lat=self.lat,
                lon=self.lon,
                fig=fig,
                ax=ax,
                cmap=(cmap if cmap is not None else 'jet'),
                levels=levels,
                ticks=ticks
            )
            fig.suptitle(
                f'Map of {self.var} ({region2str(self.region)})',
                fontweight='bold'
            )
        else:
            assert False, 'Unreachable'

        path = os.path.join(dir, name)
        _apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program,
        )
        return fig, [ax]

    @classmethod
    def load(cls: Type['Clim'], prefix: str, dir: str = '.', *, type: Optional[Literal["map", "ts"]] = None, **attrs: Any) -> 'Clim':
        """Load an clim object from matrices and type

        Parameters
        ----------
        prefix : str
            Prefix of the files containing the information for the object
        dir : str
            Directory of the files
        type : 'map' or 'ts'
            Type of climatology

        Returns
        -------
            Clim
        """
        if len(attrs) != 0:
            raise TypeError('Only kwarg `type` accepted for load method')
        if type is None:
            raise TypeError('`type` is a required kwarg')
        return super().load(prefix, dir, type=_get_type(type))

    @property
    def var_names(self) -> Tuple[str, ...]:
        """Returns the variables contained in the object (data, time, lat, lon, ...)"""
        if self._type == PlotType.TS:
            return 'data', 'time'
        elif self._type == PlotType.MAP:
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
    if dim == 'year':
        season_size = len(set(array['time.month'].values))
        if len(array['time']) % season_size:
            raise ValueError(f'Can not group time array of length {len(array["time"])} with season size of {season_size}')
        years = array['time.year'][season_size-1::season_size].data
        nyears = len(years)
        rv: xr.DataArray = array.groupby_bins('time', nyears, labels=years).mean()
        rv = rv.rename({'time_bins': 'year'})
    elif dim == 'month':
        rv = array.groupby('time.month').mean()
    elif dim == 'time':  # Apply across year and month
        assert 'time' in array.dims
        rv = array.mean(dim=dim)
    else:
        raise ValueError(f'Invalid dim {dim}')
    return rv
