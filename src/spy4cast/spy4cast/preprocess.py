import os
import sys
from typing import Optional, Any, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
import numpy.typing as npt
from scipy import signal
import xarray as xr
import cartopy.crs as ccrs

from .. import Slise, F, Month
from .._functions import debugprint, time_from_here, time_to_here, slise2str
from ..dataset import Dataset
from .._procedure import _Procedure, _get_index_from_sy, _plot_map, _apply_flags_to_fig
from ..meteo import anom




class Preprocess(_Procedure):
    _data: npt.NDArray[np.float32]
    _time: xr.DataArray
    _lat: xr.DataArray
    _lon: xr.DataArray

    _slise: Slise

    @property
    def var_names(self) -> Tuple[str, ...]:
        return (
            'time', 'lat',  'lon', 'data'
        )

    def __init__(
        self,
        ds: Dataset,
        order: Optional[int] = None,
        period: Optional[int] = None
    ):
        debugprint(f'[INFO] Preprocessing data for variable {ds.var}', end='')
        time_from_here()
        anomaly = anom(ds.data)
        self._ds: Dataset = ds
        self._time_key: str = 'year'
        self._lon_key: str = ds._lon_key
        self._lat_key: str = ds._lat_key

        if order is not None and period is not None:
            b, a = signal.butter(order, 1 / period, btype='high', analog=False, output='ba', fs=None)
            anomaly = xr.apply_ufunc(
                lambda ts: signal.filtfilt(b, a, ts),
                anomaly,
                dask='allowed',
                input_core_dims=[[self._time_key]],
                output_core_dims=[[self._time_key]]
            )
        elif order is not None or period is not None:
            if order is None:
                raise TypeError('Missing keyword argument `order`')
            elif period is None:
                raise TypeError('Missing keyword argument `period`')
            else:
                assert False, 'Unreachable'

        nt, nlat, nlon = anomaly.shape

        self._data = anomaly.transpose(
            self._time_key, self._lat_key,  self._lon_key
        ).fillna(0).values.reshape(
            (nt, nlat * nlon)
        ).transpose()

        self._time = anomaly[self._time_key]
        self._lat = anomaly.lat
        self._lon = anomaly.lon

        debugprint(f' took: {time_to_here():.03f} seconds')

    @property
    def time(self) -> xr.DataArray:
        return self._time

    @time.setter
    def time(self, arr: npt.NDArray[np.int32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `time`, '
                            f'got `{type(arr)}`')
        if arr.dtype != int:
            raise TypeError(f'Expected dtype `int` for `np.ndarray` for variable `time`, '
                            f'got `{np.dtype(arr.dtype)}`')
        self._time = xr.DataArray(arr, dims=['year'])

    @property
    def lat(self) -> xr.DataArray:
        return self._lat

    @lat.setter
    def lat(self, arr: npt.NDArray[np.float32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `lat`, '
                            f'got `{type(arr)}`')
        if (
            np.dtype(arr.dtype) != np.dtype('float32') and
            np.dtype(arr.dtype) != np.dtype('float64') and
            arr.dtype != int
        ):
            raise TypeError(f'Expected dtype `float` or `int` for  `np.ndarray` for variable `lat`, '
                            f'got `{np.dtype(arr.dtype)}`')
        self._lat = xr.DataArray(arr, dims=['lat'])

    @property
    def lon(self) -> xr.DataArray:
        return self._lon

    @lon.setter
    def lon(self, arr: npt.NDArray[np.float32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `lon`, '
                            f'got `{type(arr)}`')
        if (
            np.dtype(arr.dtype) != np.dtype('float32') and
            np.dtype(arr.dtype) != np.dtype('float64') and
            arr.dtype != int
        ):
            raise TypeError(f'Expected dtype `float` or `int` for `np.ndarray` for variable `lon`, '
                            f'got `{np.dtype(arr.dtype)}`')
        self._lon = xr.DataArray(arr, dims=['lon'])

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def data(self) -> npt.NDArray[np.float32]:
        return self._data

    @data.setter
    def data(self, arr: npt.NDArray[np.float32]) -> None:
        if type(arr) != np.ndarray:
            raise TypeError(f'Expected type `np.ndarray` for variable `data`, '
                            f'got `{type(arr)}`')
        if (
            np.dtype(arr.dtype) != np.dtype('float32') and
            np.dtype(arr.dtype) != np.dtype('float64') and
            arr.dtype != int
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
        self._data = arr

    @property
    def var(self) -> str:
        return self._ds.var if hasattr(self, '_ds') else ''

    @property
    def slise(self) -> Slise:
        if hasattr(self, '_slise'):
            return self._slise
        elif hasattr(self, '_ds'):
            self._slise = self._ds.slise
            return self._ds.slise
        else:
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

    def plot(
        self,
        flags: F = (0),
        selected_year: Optional[int] = None,
        cmap: str = 'bwr',
        dir: Optional[str] = None,
        name: Optional[str] = None
    ) -> None:
        nt, nlat, nlon = len(self.time), len(self.lat), len(self.lon)

        plotable = self.data.transpose().reshape((nt, nlat, nlon))

        index = 0 if selected_year is None \
            else _get_index_from_sy(self.time, selected_year)

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())

        _plot_map(
            plotable[index], self.lat, self.lon, fig, ax,
            f'Y on year {self.time[index]}',
            cmap=cmap,
        )
        ax.set_title(f'{self.var}: {slise2str(self.slise)}', fontweight='bold')

        if dir is None:
            dir = '.'
        if name is None:
            path = os.path.join(dir, f'preprocessed-plot_{self.var}.png')
        else:
            path = os.path.join(dir, name)

        _apply_flags_to_fig(
            fig, path, F(flags)
        )
