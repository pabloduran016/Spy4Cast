import os
from typing import Tuple, Optional, Type, Any, cast, Sequence, Literal, Union

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .. import Dataset, Region
from .._functions import region2str
from .._procedure import _Procedure, _plot_map, _plot_ts, _apply_flags_to_fig, _calculate_figsize, MAX_HEIGHT, MAX_WIDTH
import xarray as xr
import pandas as pd
import numpy as np
import numpy.typing as npt
from . import PlotType, _get_type
from ..stypes import Color, Month


__all__ = [
    'Anom',
]


class Anom(_Procedure):
    """Procedure to create the anomaly of a `spy4cast.dataset.Dataset`

    Parameters
    ----------
        ds : spy4cast.dataset.Dataset
            spy4cast.dataset.Dataset on to which perform the anomaly

        type : 'map' or 'ts'
            Perform the anomaly and outputing a map (will result in a series of maps)
             or ouputting a timeseries by doing the mean across space.

        st : bool, default=False
            Indicates whether to standarise the anomaly

    See Also
    --------
        spy4cast.dataset.Dataset, Clim

    """

    _data: xr.DataArray
    _lat: xr.DataArray
    _lon: xr.DataArray
    _ds: Dataset

    _time_key: str
    _lat_key: str
    _lon_key: str
    _region: Region
    _st: bool

    _type: PlotType

    def __init__(self, ds: Dataset, type: Literal["map", "ts"], st: bool = False):
        self.type = _get_type(type)
        self._ds = ds
        self._region = ds.region
        self._st = st

        if self._type == PlotType.TS:
            self._data = self._ds.data.mean(dim=self._ds._lon_key).mean(dim=self._ds._lat_key)
        elif self._type == PlotType.MAP:
            self._data = self._ds.data
            self._lat = self._ds.lat
            self._lon = self._ds.lon

            self._lat_key = self._ds._lat_key
            self._lon_key = self._ds._lon_key
        else:
            assert False, 'Unreachable'

        self._data = _anom(self.data, st)
        self._time_key = 'year'
        self._region.year0 = int(self.time[0])
        self._region.yearf = int(self.time[-1])

    @property
    def ds(self) -> Dataset:
        """Dataset introduced"""
        return self._ds

    @property
    def type(self) -> PlotType:
        """Type of anomaly passed in initialization

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
        if np.dtype(value.dtype) != np.dtype('float32'):
            raise ValueError(f'Dtype of `lat` has to be `float32` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[1]:
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
            raise TypeError('Latitude can not be set on a TS')
        elif self.type == PlotType.MAP:
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
        """Array of years of the time variable

        Returns
        -------
            xarray.DataArray
        """
        return self._data[self._time_key].astype(np.uint)

    @time.setter
    def time(self, value: npt.NDArray[np.uint]) -> None:
        if type(value) != np.ndarray:
            raise ValueError(f'Type of `time` has to be `np.ndarray` got {type(value)}')
        if np.dtype(value.dtype) != np.dtype('uint'):
            raise ValueError(f'Dtype of `time` has to be `uint` got {np.dtype(value.dtype)}')

        if len(value) != self.data.shape[0]:
            raise ValueError(f'Unmatching shapes for `time` and `data` variables')

        self._data = self.data.assign_coords({self._time_key: value.astype(np.uint)})

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
        elif self.type == PlotType.TS:
            # TODO: Replace month0 and monthf with meaninful values
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
        elif self.type == PlotType.MAP:
            # TODO: Replace month0 and monthf with meaninful values
            self._region = Region(
                lat0=self.lat.values[0],
                latf=self.lat.values[-1],
                lon0=self.lon.values[0],
                lonf=self.lon.values[-1],
                month0=Month.JAN,
                monthf=Month.DEC,
                year0=self.time.values[0],
                yearf=self.time.values[-1],
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
            self._data = xr.DataArray(value, coords={'year': np.arange(len(value))}, dims=['year'])
        elif self.type == PlotType.MAP:
            if len(value.shape) != 3:
                raise ValueError(f'Expected data to be three-dimensional for map. Got shape {value.shape}')
            self._data = xr.DataArray(value, coords={
                'year': np.arange(value.shape[0]),
                'lat': np.arange(value.shape[1]),
                'lon': np.arange(value.shape[2])
            }, dims=['year', 'lat', 'lon'])
            self._lon_key = 'lon'
            self._lat_key = 'lat'
        else:
            assert False, 'Unreachable'

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
        if len(array.shape) != 3 and len(array.shape) != 1:
            raise TypeError('Dimensions for array must be either 3 (MAP) or 1 (TS)')
        if 'year' not in array.coords:
            raise ValueError('Missing coord "year" for dataset anom')
        obj = Anom.__new__(Anom)
        obj._ds = Dataset.from_xrarray(array)
        obj.type = PlotType.MAP if len(array.shape) == 3 else PlotType.TS
        obj.data = _anom(array, st).values
        if obj.type == PlotType.MAP:
            obj.lat = obj._ds.lat.values
            obj.lon = obj._ds.lon.values
        return obj

    def plot(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        year: Optional[int] = None,
        cmap: Optional[str] = None,
        color: Optional[Color] = None,
        folder: str = '.',
        name: str = 'anomaly.png',
        levels: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: Optional[Literal["contour", "pcolor"]] = None,
    ) -> Tuple[Tuple[plt.Figure], Tuple[plt.Axes]]:
        """Plot the anomaly map or time series

        Parameters
        ----------
        save_fig
            Saves the fig in with `folder` / `name` parameters
        show_plot
            Shows the plot
        halt_program
            Only used if `show_plot` is `True`. If `True` shows the plot if plt.show
            and stops execution. Else uses fig.show and does not halt program
        year
            Requiered for plotting map anomalies
        cmap
            Colormap for the `map` types
        color
            Color of the line for `ts` types
        folder
            Directory to save fig if `save_fig` is `True`
        name
            Name of the fig saved if `save_fig` is `True`
        levels
            Levels for the anomaly map
        ticks
            Ticks for the anomaly map
        figsize
            Set figure size. See `plt.figure`
        plot_type : {"contour", "pcolor"}, defaut = "pcolor"
            Plot type for map. If `contour` it will use function `ax.contourf`, 
            if `pcolor` `ax.pcolormesh`.

        Returns
        -------
        figures : Tuple[plt.Figure]
            Figures objects from matplotlib. In this case just one figure

        ax : Tuple[plt.Axes]
            Tuple of axes in figure. In this case just one axes
        """
        if self._type == PlotType.TS:
            figsize = _calculate_figsize(None, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
            fig = plt.figure(figsize=figsize)
            if year is not None:
                raise TypeError('`year` parameter is not valid to plot a time series anomaly')
            if cmap is not None:
                raise TypeError('`cmap` parameter is not valid to plot a time series anomaly')
            if levels is not None:
                raise TypeError('`levels` parameter is not valid to plot a time series anomaly')
            if ticks is not None:
                raise TypeError('`ticks` parameter is not valid to plot a time series anomaly')
            if plot_type is not None:
                raise TypeError('`plot_type` parameter is not valid to plot a time series climatology')
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
                f'Anomaly time series of {self.var} ({region2str(self.region)})',
                fontweight='bold'
            )
        elif self._type == PlotType.MAP:
            if plot_type not in ("contour", "pcolor"):
                raise ValueError(f"Expected `contour` or `pcolor` for argument `plot_type`, but got {plot_type}")
            nlat, nlon = len(self.lat), len(self.lon)
            figsize = _calculate_figsize(nlat / nlon, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
            fig = plt.figure(figsize=figsize)
            if color is not None:
                raise TypeError('`color` parameter is not valid to plot a map anomaly')
            if year is None:
                raise TypeError(f'`Must provide argument `year` to plot anom')
            ax = fig.add_subplot(projection=ccrs.PlateCarree(0 if self.region.lon0 < self.region.lonf else 180))
            if self.region.lon0 < self.region.lonf:
                xlim = sorted((self.lon.values[0], self.lon.values[-1]))
            else:
                xlim = [self.region.lon0 - 180, self.region.lonf + 180]
            _plot_map(
                arr=self.data.sel({self._time_key: year}).values,
                lat=self.lat,
                lon=self.lon,
                fig=fig,
                ax=ax,
                cmap=(cmap if cmap is not None else 'jet'),
                levels=levels,
                ticks=ticks,
                xlim=xlim,
                add_cyclic_point=self.region.lon0 >= self.region.lonf,
                plot_type=plot_type,
            )
            fig.suptitle(
                f'Anomaly map of {self.var} ({region2str(self.region)})',
                fontweight='bold'
            )
        else:
            assert False, 'Unreachable'

        path = os.path.join(folder, name)
        _apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program
        )
        return (fig, ), (ax, )

    @classmethod
    def load(cls: Type['Anom'], prefix: str, folder: str = '.', *, type: Optional[Literal["map", "ts"]] = None, **attrs: Any) -> 'Anom':
        """Load an anom object from matrices and type

        Parameters
        ----------
        prefix : str
            Prefix of the files containing the information for the object
        folder : str
            Directory of the files
        type : 'map' or 'ts'
            Type of anomaly

        Returns
        -------
            Clim
        """
        if len(attrs) != 0:
            raise TypeError('Only accepetd kwarg `type` accepted for load method')
        if type is None:
            raise TypeError('`type` is a required keyword argument')
        return super().load(prefix, folder, type=_get_type(type))

    @property
    def var_names(self) -> Tuple[str, ...]:
        """Returns the variables contained in the object (data, time, lat, lon, ...)"""
        if self._type == PlotType.TS:
            return 'data', 'time'
        elif self._type == PlotType.MAP:
            return 'data', 'time', 'lat', 'lon'
        else:
            assert False, f'Unreachable: {self._type}'


def _anom(
        array: xr.DataArray, st: bool = False
) -> xr.DataArray:
    if not isinstance(array, xr.DataArray):
        raise TypeError(f"Invalid type for array: {type(array)}")

    assert 'year' in array.coords, 'Cant\'t recognise year or time key in array'
    if len(array.shape) == 3:  # 3d array
        # Reshape time variable
        lat_key = 'latitude' if 'latitude' in array.dims else 'lat'
        lon_key = 'longitude' if 'longitude' in array.dims else 'lon'
        assert lat_key in array.dims and lon_key in array.dims,\
            'Can\'t recognise keys'
        a = array.groupby('year').mean()
        b: xr.DataArray = a - a.mean('year')
        if st:
            rv: xr.DataArray = b / b.std()
            return rv
        return b

    elif len(array.shape) == 1:  # time series
        assert 'latitude' not in array.dims and 'longitude' not in array.dims,\
            'Unidimensional arrays time must be the only dimension'
        a = array.groupby('year').mean()
        b = a - a.mean('year')
        if st:
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
