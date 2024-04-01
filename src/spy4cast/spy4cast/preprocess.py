import os
from typing import Optional, Tuple, Any, cast, Sequence, Literal
from cartopy.util import add_cyclic_point

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy.typing as npt
from scipy import signal
import xarray as xr
import cartopy.crs as ccrs

from .. import Region, Month
from .._functions import time_from_here, time_to_here, region2str, _debuginfo, debugprint
from ..dataset import Dataset
from .._procedure import _Procedure, _get_index_from_sy, _plot_map, _apply_flags_to_fig, _calculate_figsize, MAX_WIDTH, \
    MAX_HEIGHT, _add_cyclic_point
from ..land_array import LandArray
from ..meteo import Anom


__all__ = [
    'Preprocess',
]


class Preprocess(_Procedure):
    """Preprocess variables for MCA and Crossvalidation: anomaly and reshaping

    Parameters
    ----------
        ds : Dataset
            Dataset to preprocess
        order : optional, int
            If specified as well as period, a butterworth filter with those parameters will be applied
        period : optional, float
            If specified as well as period, a butterworth filter with those parameters will be applied
        freq : {'high', 'low'}, default = 'high'
            If specified as well as period, a butterworth filter with those parameters will be applied
        detrend : bool, default = False
            Apply `scipy.signal.detrend` on the time axis.

    Examples
    --------

    Preprocess a dataset with one line

    >>> from spy4cast import Dataset, Region, Month
    >>> from spy4cast.spy4cast import Preprocess
    >>> ds = Dataset("dataset.nc").open("sst").slice(
    ...     Region(-40, 40, -20, 20, Month.JAN, Month.MAR, 1940, 2000))
    >>> y = Preprocess(ds)

    Add a butterworth filter

    >>> y = Preprocess(ds, period=12, order=4)

    Detrend

    >>> y = Preprocess(ds, detrend=True)

    Acces all the :doc:`/variables/preprocess` easily

    >>> data = y.data.reshape((len(y.lat), len(y.lon), len(y.time)))
    >>> # Plot with any plotting library
    >>> import matplotlib.pyplot as plt
    >>> import cartopy.crs as ccrs
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection=ccrs.PlateCarree())
    >>> ax.contourf(y.lon, y.lat, data[:, :, 0])
    >>> ax.coastlines()
    >>> plt.show()

    Save the preprocess in a file to use later

    >>> y.save("sst_preprocessed_", folder="saved_data")

    Avoid loading the dataset and work with the preprocessed data directly

    >>> y = Preprocess.load("sst_preprocessed_", folder="saved_data")

    Plot a map with just one line to visualize the anomaly

    >>> y.plot(1990, show_plot=True, halt_program=True)

    """
    _time: xr.DataArray
    _lat: xr.DataArray
    _lon: xr.DataArray

    _region: Region
    _var: str

    @property
    def var_names(self) -> Tuple[str, ...]:
        """Returns the variables contained in the object (data, time, lat, lon, ...)"""
        return (
            'time', 'lat',  'lon', 'data', 'meta'
        )

    def __init__(
        self,
        ds: Dataset,
        order: Optional[int] = None,
        period: Optional[float] = None,
        freq: Literal["high", "low"] = "high",
        detrend: bool = False,
    ):
        _debuginfo(f'Preprocessing data for variable {ds.var}', end='')
        time_from_here()
        assert len(ds.data.dims) == 3
        anomaly = Anom(ds, 'map').data
        self._ds: Dataset = ds
        
        anomaly = anomaly.transpose(
            'year', ds._lat_key,  ds._lon_key
        )

        nt, nlat, nlon = anomaly.shape

        data = anomaly.values.reshape((nt, nlat * nlon)).transpose()  # space x time

        if order is not None and period is not None:
            b, a = cast(Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], signal.butter(order, 2 / period, btype=freq, analog=False, output='ba', fs=None))
            data = np.apply_along_axis(lambda ts: cast(Tuple[npt.NDArray[np.float_]], signal.filtfilt(b, a, ts)), 1, data)
        elif order is not None or period is not None:
            if order is None:
                raise TypeError('Missing keyword argument `order`')
            elif period is None:
                raise TypeError('Missing keyword argument `period`')
            else:
                assert False, 'Unreachable'

        self._land_data = LandArray(data)

        if detrend:
            self._land_data.values[~self._land_data.land_mask] = signal.detrend(self._land_data.not_land_values)  # detrend in time

        self._time = anomaly['year']
        self._lat = anomaly[ds._lat_key]
        self._lon = anomaly[ds._lon_key]

        debugprint(f' took: {time_to_here():.03f} seconds')

    @property
    def ds(self) -> Dataset:
        """Dataset that has been preprocessed. On loaded preprocess this raises an error"""
        return self._ds

    @property
    def meta(self) -> npt.NDArray[Any]:
        """Returns a np.ndarray containg information about the preprocessed
        dataset. It includes the region and the variable

        First 9 values is region as numpy, then variable as str
        """
        return np.concatenate((self.region.as_numpy(), np.array([self.var])))

    @meta.setter
    def meta(self, arr: npt.NDArray[Any]) -> None:
        self._region = Region.from_numpy(arr[:9].astype(np.float32))
        self.var = str(arr[9])

    @property
    def time(self) -> xr.DataArray:
        """Time coordinate of the data in years."""
        return self._time

    @time.setter
    def time(self, arr: npt.NDArray[np.int32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `time`, '
                            f'got `{type(arr)}`')
        if arr.dtype not in [np.int8, np.int16, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64]:  # type: ignore
            raise TypeError(f'Expected dtype `int` for `np.ndarray` for variable `time`, '
                            f'got `{np.dtype(arr.dtype)}`')
        self._time = xr.DataArray(arr, dims=['year'])

    @property
    def lat(self) -> xr.DataArray:
        """Latitude coordinate of the variable in degrees ranging from -90 to 90"""
        return self._lat

    @lat.setter
    def lat(self, arr: npt.NDArray[np.float32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `lat`, '
                            f'got `{type(arr)}`')
        if (
            np.dtype(arr.dtype) != np.dtype('float32') and
            np.dtype(arr.dtype) != np.dtype('float64') and
            arr.dtype not in [np.int8, np.int16, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64]  # type: ignore
        ):
            raise TypeError(f'Expected dtype `float` or `int` for  `np.ndarray` for variable `lat`, '
                            f'got `{np.dtype(arr.dtype)}`')
        self._lat = xr.DataArray(arr, dims=['lat'])

    @property
    def lon(self) -> xr.DataArray:
        """Longitude coordinate of the data in degrees ranging from -180 to 180"""
        return self._lon

    @lon.setter
    def lon(self, arr: npt.NDArray[np.float32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `lon`, '
                            f'got `{type(arr)}`')
        if (
            np.dtype(arr.dtype) != np.dtype('float32') and
            np.dtype(arr.dtype) != np.dtype('float64') and
            arr.dtype not in [np.int8, np.int16, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64]  # type: ignore
        ):
            raise TypeError(f'Expected dtype `float` or `int` for `np.ndarray` for variable `lon`, '
                            f'got `{np.dtype(arr.dtype)}`')
        self._lon = xr.DataArray(arr, dims=['lon'])

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the data as a tuple of space x time"""
        return self._land_data.shape

    @property
    def land_data(self) -> LandArray:
        """Data but organised in a land array. This includes a mask that indicates where the land
        is in the dataset by masking the `nan` values. This is useful when handling variables like
        sea surface temperature"""
        return self._land_data

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """Raw data in the object with `nan` as in the original dataset. It has dimensions of
        space x time. Should be reshaped like: data.reshape((nlat, nlon, ntime))"""
        return self._land_data.values

    @data.setter
    def data(self, arr: npt.NDArray[np.float32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `data`, '
                            f'got `{type(arr)}`')
        if (
            np.dtype(arr.dtype) != np.dtype('float32') and
            np.dtype(arr.dtype) != np.dtype('float64') and
            arr.dtype not in [np.int8, np.int16, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64]  # type: ignore
        ):
            raise TypeError(f'Expected dtype `float` or `int` for `np.ndarray` for variable `data`, '
                            f'got {np.dtype(arr.dtype)}')
        if len(arr.shape) != 2:
            raise TypeError(f'Expected 2-dimensional for `np.ndarray` for variable `data`, '
                            f'got shape `{arr.shape}`')
        ns, nt = arr.shape
        nlon = len(self.lon)
        nlat = len(self.lat)
        ntime = len(self.time)
        if ntime != nt:
            raise TypeError('Expected first dimension of `np.ndarray` for variable `data` to have '
                            'the same dimensions as time, '
                            f'got shape `{arr.shape}` and time dimension is {ntime}')
        if nlon * nlat != ns:
            raise TypeError('Expected second dimension of `np.ndarray` for variable `data` to have '
                            'the same dimensions as lon * lat, '
                            f'got shape `{arr.shape}` and lon dimesion is {nlon} and lat {nlat} ({nlat * nlon = }')
        self._land_data = LandArray(arr)

    @property
    def var(self) -> str:
        """Name of the variable of the dataset that was preprocessed."""
        return self._var if hasattr(self, '_var') else self._ds.var if hasattr(self, '_ds') else ''

    @var.setter
    def var(self, value: str) -> None:
        if hasattr(self, '_ds'):
            raise TypeError('Can not set var in Preprocess')
        if type(value) != str:
            raise TypeError(f'Expected type string for var, got {type(value)}')
        self._var = value

    @property
    def region(self) -> Region:
        """Region used to slice the original dataset"""
        if hasattr(self, '_region'):
            return self._region
        elif hasattr(self, '_ds'):
            self._region = self._ds.region
            return self._ds.region
        else:
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

    def plot(
        self,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        selected_year: Optional[int] = None,
        cmap: str = 'bwr',
        folder: Optional[str] = None,
        name: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: Literal["contour", "pcolor"] = "contour",
    ) -> Tuple[Tuple[plt.Figure], Tuple[plt.Axes]]:
        """Plot the preprocessed data for spy4cast methodologes

        Parameters
        ----------
        save_fig
            Saves the fig using `folder` and `name` parameters
        show_plot
            Shows the plot
        halt_program
            Only used if `show_plot` is `True`. If `True` shows the plot if plt.show
            and stops execution. Else uses fig.show and does not halt program
        cmap
            Colormap for the map
        folder
            Directory to save fig if `save_fig` is `True`
        name
            Name of the fig saved if `save_fig` is `True`
        figsize
            Set figure size. See `plt.figure`
        plot_type : {"contour", "pcolor"}, defaut = "pcolor"
            Plot type. If `contour` it will use function `ax.contourf`, 
            if `pcolor` `ax.pcolormesh`.

        Examples
        --------

        Plot the anomaly on any year of the dataset

        >>> y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 2010)))
        >>> # Plot 1990, 1991, 1992 and save 1990
        >>> y.plot(1990, show_plot=True, save_fig=True, name='y_1990.png')
        >>> y.plot(1991, show_plot=True, cmap='viridis')  # Change the default color map
        >>> y.plot(1992, show_plot=True, halt_program=True)  # halt_program lets you show multiple figures at the same time

        Returns
        -------
        figures : Tuple[plt.Figure]
            Figures objects from matplotlib. In this case just one figure with one axes

        ax : Tuple[plt.Axes]
            Tuple of axes in figure. In this case just one axes
        """
        if plot_type not in ("contour", "pcolor"):
            raise ValueError(f"Expected `contour` or `pcolor` for argument `plot_type`, but got {plot_type}")

        nt, nlat, nlon = len(self.time), len(self.lat), len(self.lon)

        plotable = self.land_data.values.transpose().reshape((nt, nlat, nlon))

        index = 0 if selected_year is None \
            else _get_index_from_sy(self.time, selected_year)

        figsize = _calculate_figsize(nlat / nlon, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(0 if self.region.lon0 < self.region.lonf else 180))

        if self.region.lon0 < self.region.lonf:
            xlim = sorted((self.lon.values[0], self.lon.values[-1]))
        else:
            xlim = [self.region.lon0 - 180, self.region.lonf + 180]
        _plot_map(
            plotable[index], self.lat, self.lon, fig, ax,
            f'Year {self.time[index].values}',
            cmap=cmap, xlim=xlim, cax=fig.add_subplot(gs[1]),
            add_cyclic_point=self.region.lon0 >= self.region.lonf,
            plot_type=plot_type,
        )
        fig.suptitle(f'{self.var}: {region2str(self.region)}', fontweight='bold')

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'preprocessed-plot_{self.var}.png')
        else:
            path = os.path.join(folder, name)

        _apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program,
        )
        return (fig, ), (ax, )
