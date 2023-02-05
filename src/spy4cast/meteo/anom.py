import os
from typing import Tuple, Optional, Type, Any, cast, Sequence, Literal

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .. import Dataset, Slise
from .._functions import slise2str
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
    _slise: Slise
    _st: bool

    _type: PlotType

    def __init__(self, ds: Dataset, type: Literal["map", "ts"], st: bool = False):
        self.type = _get_type(type)
        self._ds = ds
        self._slise = ds.slise
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
        self._slise.year0 = int(self.time[0])
        self._slise.yearf = int(self.time[-1])

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
    def slise(self) -> Slise:
        """Slise applied to the matrix.

        Returns
        -------
            spy4cast.stypes.Slise

        Note
        ----
            If type is `ts` and initilization from ds was not run then a default time and region slise is returned

        Note
        ----
            If type is `map` and initilization from ds was not run then a default time slise is returned
        """
        if hasattr(self, '_slise'):
            return self._slise
        elif self.type == PlotType.TS:
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
        elif self.type == PlotType.MAP:
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
        dir: str = '.',
        name: str = 'anomaly.png'
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot the anomaly map or time series

        Parameters
        ----------
        save_fig
            Saves the fig in with `dir` / `name` parameters
        show_plot
            Shows the plot
        halt_program
            Only used if `show_plot` is `True`. If `True` shows the plot if plt.show
            and stops execution. Else uses fig.show and does not halt program
        year
            Requiered for plotting map anomalies
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
            if year is not None:
                raise TypeError('`year` parameter is not valid to plot a time series anomaly')
            if cmap is not None:
                raise TypeError('`cmap` parameter is not valid to plot a time series anomaly')
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
        elif self._type == PlotType.MAP:
            nlat, nlon = len(self.lat), len(self.lon)
            fig = plt.figure(figsize=_calculate_figsize(nlat / nlon, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
            if color is not None:
                raise TypeError('`color` parameter is not valid to plot a map anomaly')
            if year is None:
                raise TypeError(f'`year` is a required argument for plotting an anomaly map')
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
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program
        )
        return fig, [ax]

    @classmethod
    def load(cls: Type['Anom'], prefix: str, dir: str = '.', *, type: Optional[Literal["map", "ts"]] = None, **attrs: Any) -> 'Anom':
        """Load an anom object from matrices and type

        Parameters
        ----------
        prefix : str
            Prefix of the files containing the information for the object
        dir : str
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
        return super().load(prefix, dir, type=_get_type(type))

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
        raise ValueError('Strange time variable. Try slicing the dataset')
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
