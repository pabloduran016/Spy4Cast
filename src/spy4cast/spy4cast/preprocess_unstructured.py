import os
from typing import Optional, Tuple, Any, Union, cast, Sequence, Literal

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
from .._procedure import _Procedure, _get_index_from_sy, plot_map, _apply_flags_to_fig, _calculate_figsize, MAX_WIDTH, \
    MAX_HEIGHT, add_cyclic_point_to_data, get_xlim_from_region, get_central_longitude_from_region
from ..land_array import LandArray


__all__ = [
    'PreprocessUnstructured',
]


class PreprocessUnstructured(_Procedure):
    """
    Stores unstructured data (data points not distributed in a grid).
    Preprocess variables for MCA and Crossvalidation: anomaly and reshaping

    Parameters
    ----------
        data_matrix : array
            Data matrix with dimensions (space x time) where each row is a node in the unstructured data 
            and each column corresponds to the time dimension
        time : array
            Labels each time as integers. For example years
        coords : array
            Matrix with dimensions (space x 2) where each row is a node from the structured data, the 
            first column corresponds to the latitude and the second to longitude
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
    """
    var: str

    _data_matrix: xr.DataArray
    _time: xr.DataArray
    _coords: xr.DataArray
    _region: Region

    VAR_NAMES = ('time', 'coords', "data", "var", 'data_matrix')

    @property
    def var_names(self) -> Tuple[str, ...]:
        """Returns the variables contained in the object (data, time, lat, lon, ...)"""
        return self.VAR_NAMES

    def __init__(
        self,
        data_matrix: npt.NDArray[np.float32],
        time: npt.NDArray[np.int32],
        coords: npt.NDArray[np.float32],
        var: Optional[str] = None,
        order: Optional[int] = None,
        period: Optional[float] = None,
        freq: Literal["high", "low"] = "high",
        detrend: bool = False,
    ):
        _debuginfo(f'Preprocessing Unstructured data', end='')
        time_from_here()

        self.time = time
        self.coords = coords
        self.data_matrix = data_matrix
        self.var = var if var is not None else "Z"
 
        data = data_matrix - np.mean(data_matrix)

        ns, nt = data.shape

        if order is not None and period is not None:
            b, a = cast(Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]], signal.butter(order, 2 / period, btype=freq, analog=False, output='ba', fs=None))
            data = np.apply_along_axis(lambda ts: cast(Tuple[npt.NDArray[np.float_]], 
                                                       signal.filtfilt(b, a, ts)), 1, data)
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

        debugprint(f' took: {time_to_here():.03f} seconds')

    @property
    def region(self) -> Region:
        """Region of the data"""
        if hasattr(self, '_region'):
            return self._region
        else:
            lon0, lonf = self.coords.values[:, 1].min(), self.coords.values[:, 1].max(),
            pad_lon = abs(lonf - lon0)*0.1
            lat0, latf = self.coords.values[:, 0].min(), self.coords.values[:, 0].max(),
            pad_lat = abs(latf - lat0)*0.1
            lon0 -= pad_lon
            lonf += pad_lon
            lat0 -= pad_lat
            latf += pad_lat
            self._region = Region(
                lat0=lat0, latf=latf,
                lon0=lon0, lonf=lonf,
                month0=Month.JAN, monthf=Month.DEC,
                year0=self.time.values[0], yearf=self.time.values[-1],
            )
            return self._region

    @property
    def data_matrix(self) -> xr.DataArray:
        """Data martrix introduced as input"""
        return self._data_matrix

    @data_matrix.setter
    def data_matrix(self, arr: npt.NDArray[np.float32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `data_matrix`, '
                            f'got `{type(arr)}`')
        if (
            np.dtype(arr.dtype) != np.dtype('float32') and
            np.dtype(arr.dtype) != np.dtype('float64') and
            arr.dtype not in [np.int8, np.int16, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64]  # type: ignore
        ):
            raise TypeError(f'Expected dtype `float` or `int` for `np.ndarray` for variable `data_matrix`, '
                            f'got {np.dtype(arr.dtype)}')
        if len(arr.shape) != 2:
            raise TypeError(f'Expected 2-dimensional for `np.ndarray` for variable `data_matrix`, '
                            f'got shape `{arr.shape}`')
        ns, nt = arr.shape
        nlatlon = self.coords.shape[0]
        ntime = self.time.shape[0]
        if ntime != nt:
            raise TypeError('Expected first dimension of `np.ndarray` for variable `data_matrix` to have '
                            'the same dimensions as time, '
                            f'got shape `{arr.shape}` and time dimension is {ntime}')
        if nlatlon != ns:
            raise TypeError('Expected second dimension of `np.ndarray` for variable `data_matrix` to have '
                            'the same dimensions as lon * lat, '
                            f'got shape `{arr.shape}` and space dimesion is {nlatlon}')
        self._data_matrix = xr.DataArray(arr, dims=["space", "time"])

    @property
    def time(self) -> xr.DataArray:
        """Time coordinate of the data"""
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
        """Latitude coordinate of the variable"""
        return self._coords[:, 0]

    @property
    def lon(self) -> xr.DataArray:
        """Longitude coordinate of the variable"""
        return self._coords[:, 1]

    @property
    def coords(self) -> xr.DataArray:
        """Latitude and longitude coordinate of the variable"""
        return self._coords

    @coords.setter
    def coords(self, arr: npt.NDArray[np.float32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `coords`, '
                            f'got `{type(arr)}`')
        if arr.shape[1] != 2:
            raise TypeError(f'Expected dimensions to be (space x 2), got {arr.shape}')
        if (
            np.dtype(arr.dtype) != np.dtype('float32') and
            np.dtype(arr.dtype) != np.dtype('float64') and
            arr.dtype not in [np.int8, np.int16, np.int64, np.int32, np.uint8, np.uint16, np.uint32, np.uint64]  # type: ignore
        ):
            raise TypeError(f'Expected dtype `float` or `int` for  `np.ndarray` for variable `coords`, '
                            f'got `{np.dtype(arr.dtype)}`')
        self._coords = xr.DataArray(arr, dims=['space', 'latlon'])

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
        space x time."""
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
        nlatlon = self.coords.shape[0]
        ntime = self.time.shape[0]
        if ntime != nt:
            raise TypeError('Expected first dimension of `np.ndarray` for variable `data` to have '
                            'the same dimensions as time, '
                            f'got shape `{arr.shape}` and time dimension is {ntime}')
        if nlatlon != ns:
            raise TypeError('Expected second dimension of `np.ndarray` for variable `data` to have '
                            'the same dimensions as lon * lat, '
                            f'got shape `{arr.shape}` and space dimesion is {nlatlon}')
        self._land_data = LandArray(arr)

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
        plot_type: Literal["tricontour", "scatter"] = "scatter",
        levels: Optional[
            Union[npt.NDArray[np.float32], Sequence[float], bool]
        ] = None,
    ) -> Tuple[Tuple[plt.Figure], Tuple[plt.Axes]]:
        """Plot the unstructured preprocessed data for spy4cast methodologes

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
        plot_type : {"tricontour", "scatter"}, defaut = "scatter"
            Plot type. If `tricontour` it will use function `ax.tricontourf`, 
            if `scatter` `ax.scatter`.
        levels
            Levels for the map

        Examples
        --------

        Returns
        -------
        figures : Tuple[plt.Figure]
            Figures objects from matplotlib. In this case just one figure with one axes

        ax : Tuple[plt.Axes]
            Tuple of axes in figure. In this case just one axes
        """
        if plot_type not in ("tricontour", "scatter"):
            raise ValueError(f"Expected `tricontour` for argument `plot_type`, but got {plot_type}")

        index = 0 if selected_year is None \
            else _get_index_from_sy(self.time, selected_year)

        z = self.land_data.values[:, index]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])

        # central longitude
        lat = self.coords[:, 0]
        lon = self.coords[:, 1]
        lon0, lonf = lon.min(), lon.max()
        central_longitude = get_central_longitude_from_region(lon0, lonf)
        xlim = get_xlim_from_region(lon0, lonf, central_longitude)

        ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree(central_longitude))


        if plot_type == "tricontour":
            levels = np.linspace(z.min(), z.max(), 14)
            ax.plot(
                lon, lat, 'o', 
                transform=ccrs.PlateCarree(),
                markersize=2, markerfacecolor="none",
                color='black'
            )
        elif plot_type == "scatter":
            levels = None
        else:
            assert False, "Unreachable"
        plot_map(
            z, lat, lon, fig, ax,
            f'Year {self.time[index].values}',
            levels=levels,
            cmap=cmap, xlim=xlim, cax=fig.add_subplot(gs[1]),
            # add_cyclic_point=self.region.lon0 >= self.region.lonf,
            plot_type=plot_type,
        )

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'preprocessed-plot.png')
        else:
            path = os.path.join(folder, name)

        _apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program,
        )
        return (fig, ), (ax, )
