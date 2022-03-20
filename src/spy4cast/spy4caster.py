"""
Module containing the class to perfor the Spy4Cast methodology
"""

import sys
import os
import traceback
from typing import Union, Dict, Optional, Any, Tuple, Sequence, List

import xarray as xr
from scipy import signal
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from ._functions import \
    debugprint, time_from_here, time_to_here, slise2str
from .stypes import Slise, F, RDArgs, RDArgsDict
from .errors import \
    DataSavingError, Spy4CastError, PlotCreationError
from .read_data import ReadData, NAN_VAL
from .meteo import \
    MCAOut, CrossvalidationOut, anom, \
    crossvalidation, crossvalidation_mp, mca

__all__ = ['Spy4Caster']


class Spy4Caster:
    def __init__(
        self,
        yargs: Union[RDArgs, RDArgsDict],
        zargs: Union[RDArgs, RDArgsDict],
        plot_dir: str = '',
        mats_plot_name: str = 'mats_plot.png',
        mca_plot_name: str = 'mca_plot.png',
        cross_plot_name: str = 'cross_plot.png',
        zhat_plot_name: str = 'zhat_plot.png',
        plot_data_dir: str = '',
        plot_data_sufix: str = ''
    ):
        if type(yargs) == RDArgs:
            yargs = yargs.as_dict()
        if type(zargs) == RDArgs:
            zargs = zargs.as_dict()
        assert type(yargs) == dict
        assert type(zargs) == dict
        self._rdy = ReadData(**yargs)  # Predictor
        self._rdz = ReadData(**zargs)  # Predictand
        self._mca_out: Optional[MCAOut] = None
        self._crossvalidation_out: Optional[CrossvalidationOut] = None
        self._plot_dir = plot_dir
        self._mats_plot_name = mats_plot_name
        self._mca_plot_name = mca_plot_name
        self._cross_plot_name = cross_plot_name
        self._zhat_plot_name = zhat_plot_name
        self._plot_data_dir = plot_data_dir
        self._plot_data_sufix = plot_data_sufix
        self._y: Optional[npt.NDArray[np.float32]] = None
        self._ylat: Optional[npt.NDArray[np.float32]] = None
        self._ylon: Optional[npt.NDArray[np.float32]] = None
        self._ytime: Optional[npt.NDArray[np.int64]] = None
        self._z: Optional[npt.NDArray[np.float32]] = None
        self._zlat: Optional[npt.NDArray[np.float32]] = None
        self._zlon: Optional[npt.NDArray[np.float32]] = None
        self._ztime: Optional[npt.NDArray[np.int64]] = None
        self._opened_figs: List[plt.Figure] = []

    def _set_var(self, name: str, val: Union[npt.NDArray[np.float32],
                 npt.NDArray[np.int64], xr.DataArray]) -> None:
        if name == 'z':
            assert val.dtype == np.float32, \
                f'Expacted dtype `{np.float32}` for `z`, got `{val.dtype}`'
            self._z = val.astype(np.float32)
        elif name == 'zlat':
            assert val.dtype == np.float32, \
                f'Expacted dtype `{np.float32}` for `zlat`, got `{val.dtype}`'
            self._zlat = val.astype(np.float32)
        elif name == 'zlon':
            assert val.dtype == np.float32, \
                f'Expacted dtype `{np.float32}` for `zlon`, got `{val.dtype}`'
            self._zlon = val.astype(np.float32)
        elif name == 'ztime':
            assert val.dtype == np.int64, \
                f'Expacted dtype `{np.int64}` for `ztime`, got `{val.dtype}`'
            self._ztime = val.astype(np.int64)
        elif name == 'y':
            assert val.dtype == np.float32, \
                f'Expacted dtype `{np.float32}` for `y`, got `{val.dtype}`'
            self._y = val.astype(np.float32)
        elif name == 'ylat':
            assert val.dtype == np.float32, \
                f'Expacted dtype `{np.float32}` for `ylat`, got `{val.dtype}`'
            self._ylat = val.astype(np.float32)
        elif name == 'ylon':
            assert val.dtype == np.float32, \
                f'Expacted dtype `{np.float32}` for `ylon`, got `{val.dtype}`'
            self._ylon = val.astype(np.float32)
        elif name == 'ytime':
            assert val.dtype == np.int64,\
                f'Expacted dtype `{np.int64}` for `ztime`, got `{val.dtype}`'
            self._ytime = val.astype(np.int64)
        else:
            raise ValueError(f'Unknown variable name {name}')

    def _close_figures(self) -> None:
        for fig in self._opened_figs:
            plt.close(fig)
        self._opened_figs = []

    def _new_figure(
            self, figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        fig = plt.figure(figsize=figsize)
        self._opened_figs.append(fig)
        return fig

    def open_datasets(self) -> 'Spy4Caster':
        debugprint('[INFO] Opening datasets', end='')
        time_from_here()
        self._rdy.open_dataset()
        self._rdz.open_dataset()
        self._rdy._data = self._rdy._data.where(
            lambda a: abs(a) < NAN_VAL
        ).sortby(self._rdy._lon_key)
        self._rdz._data = self._rdz._data.where(
            lambda a: abs(a) < NAN_VAL
        ).sortby(self._rdz._lon_key)
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def load_datasets(self) -> 'Spy4Caster':
        debugprint('[INFO] Loading datasets', end='')
        time_from_here()
        self._rdy.load_dataset()
        self._rdz.load_dataset()
        self._rdy._data = self._rdy._data.where(
            lambda a: abs(a) < NAN_VAL
        ).sortby(self._rdy._lon_key)
        self._rdz._data = self._rdz._data.where(
            lambda a: abs(a) < NAN_VAL
        ).sortby(self._rdy._lon_key)
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def slice_datasets(self, yslise: Slise, zslise: Slise,
                       yskip: int = 0, zskip: int = 0) -> 'Spy4Caster':
        debugprint('[INFO] Slicing datasets', end='')
        time_from_here()
        self._rdy.slice_dataset(yslise, yskip)
        self._rdz.slice_dataset(zslise, zskip)
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def preprocess(self, flags: int = 0, order: Optional[int] = None,
                   period: Optional[float] = None) -> 'Spy4Caster':
        debugprint('[INFO] Preprocessing data', end='')
        flags = F(flags) if type(flags) == int else flags
        assert type(flags) == F
        time_from_here()
        self._rdy._data = anom(self._rdy._data)
        self._rdy._time_key = 'year'
        self._rdz._data = anom(self._rdz._data)
        self._rdz._time_key = 'year'

        if len(self._rdz.time) != len(self._rdy.time):
            raise ValueError(
                f'The number of years of the predictand must be the '
                f'same as the number of years of the predictor: '
                f'got {len(self._rdz.time)} and '
                f'{len(self._rdy.time)}'
            )

        _z = self._rdz._data
        if F.FILTER in flags:
            if order is None:
                raise TypeError('Missing keyword argument `order`')
            if period is None:
                raise TypeError('Missing keyword argument `period`')

            b, a = signal.butter(order, 1 / period, btype='high',
                                 analog=False, output='ba', fs=None)
            _z = xr.apply_ufunc(
                lambda ts: signal.filtfilt(b, a, ts),
                _z,
                dask='allowed',
                input_core_dims=[[self._rdz._time_key]],
                output_core_dims=[[self._rdz._time_key]]
            )

        nyt, nylat, nylon = self._rdy._data.shape
        nzt, nzlat, nzlon = self._rdz._data.shape

        self._z = _z.transpose(
            self._rdz._time_key, self._rdz._lat_key,
            self._rdz._lon_key
        ).fillna(0).values.reshape(
            (nzt, nzlat * nzlon)
        ).transpose()

        del _z
        self._zlat = self._rdz.lat.values
        self._zlon = self._rdz.lon.values
        self._ztime = self._rdz.time.values

        self._y = self._rdy._data.fillna(0).values.reshape(
            (nyt, nylat * nylon)
        ).transpose()

        self._ylat = self._rdy.lat.values
        self._ylon = self._rdy.lon.values
        self._ytime = self._rdy.time.values

        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def load_preprocessed(self, path0: str, prefix: str = '',
                          ext: str = '.npy') -> 'Spy4Caster':
        debugprint(
            f'[INFO] Loading Preprocessed data from '
            f'`{path0}/{prefix}*{ext}`',
            end=''
        )
        time_from_here()

        for field in ('y', 'z'):
            for var in ('', 'lat', 'lon', 'time'):
                path = os.path.join(path0, f'{prefix}{field}{var}{ext}')
                try:
                    self._set_var(field+var, np.load(path))
                except FileNotFoundError:
                    print(
                        f'\n[ERROR] Could not find file `{path}` for '
                        f'variable `{field}{var}`'
                    )

        debugprint(f' took {time_to_here():.03f} seconds')
        return self

    def load_mca(self, path0: str, prefix: str = '',
                 ext: str = '.npy') -> 'Spy4Caster':
        debugprint(
            f'[INFO] Loading MCA data from `{path0}/{prefix}*{ext}`',
            end=''
        )
        time_from_here()
        out: Dict[str, Any] = {
            'RUY': None,
            'RUY_sig': None,
            'SUY': None,
            'SUY_sig': None,
            'RUZ': None,
            'RUZ_sig': None,
            'SUZ': None,
            'SUZ_sig': None,
            'Us': None,
            'Vs': None,
            'scf': None,
            'alpha': None,
        }
        for key in out.keys():
            path = os.path.join(path0, f'{prefix}{key}{ext}')
            try:
                out[key] = np.load(path)
            except FileNotFoundError:
                print(
                    f'\n[ERROR] Could not find file `{path}` for '
                    f'variable `{key}`'
                )

        self._mca_out = MCAOut(
            RUY=out['RUY'],
            RUY_sig=out['RUY_sig'],
            SUY=out['SUY'],
            SUY_sig=out['SUY_sig'],
            RUZ=out['RUZ'],
            RUZ_sig=out['RUZ_sig'],
            SUZ=out['SUZ'],
            SUZ_sig=out['SUZ_sig'],
            Us=out['Us'],
            Vs=out['Vs'],
            scf=out['scf'],
            alpha=out['alpha'],
        )
        debugprint(f' took: {time_to_here():.03f}')
        return self

    def load_crossvalidation(self, path0: str, prefix: str = '',
                             ext: str = '.npy') -> 'Spy4Caster':
        debugprint(
            f'[INFO] Loading Crossvalidation data from '
            f'`{path0}/{prefix}*{ext}`',
            end=''
        )
        time_from_here()
        out: Dict[str, Any] = {
            'zhat': None,
            'scf': None,
            'r_z_zhat_t': None,
            'p_z_zhat_t': None,
            'r_z_zhat_s': None,
            'p_z_zhat_s': None,
            'r_uv': None,
            'p_uv': None,
            'us': None,
            'alpha': None,
        }
        for key in out.keys():
            path = os.path.join(path0, f'{prefix}{key}{ext}')
            try:
                out[key] = np.load(path)
            except FileNotFoundError:
                print(
                    f'\n[ERROR] Could not find file `{path}` for '
                    f'variable `{key}`'
                )

        self._crossvalidation_out = CrossvalidationOut(
            zhat=out['zhat'],
            scf=out['scf'],
            r_z_zhat_t=out['r_z_zhat_t'],
            p_z_zhat_t=out['p_z_zhat_t'],
            r_z_zhat_s=out['r_z_zhat_s'],
            p_z_zhat_s=out['p_z_zhat_s'],
            r_uv=out['r_uv'],
            p_uv=out['p_uv'],
            us=out['us'],
            alpha=out['alpha'],
        )
        debugprint(f' took {time_to_here():.03f} seconds')
        return self

    def mca(self, nm: int, alpha: float) -> 'Spy4Caster':
        debugprint('[INFO] Applying MCA', end='')
        time_from_here()
        if any([
            x is None for x in (
                self._y, self._ylat, self._ylon, self._ytime,
                self._z, self._zlat, self._zlon, self._ztime
            )
        ]):
            raise TypeError(
                'Must prprocess data before applying MCA'
            )
        assert self._z is not None
        assert self._y is not None
        self._mca_out = mca(self._z, self._y, nm, alpha)
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def crossvalidation(self, nm: int, alpha: float,
                        multiprocessing: bool = False) -> 'Spy4Caster':
        debugprint(
            f'[INFO] Applying crossvalidation '
            f'{"(mp) " if multiprocessing else ""}',
            end=''
        )
        time_from_here()
        if any([
            x is None for x in (
                self._y, self._ylat, self._ylon, self._ytime,
                self._z, self._zlat, self._zlon, self._ztime
            )
        ]):
            raise TypeError(
                'Must preprocess data before applying Crossvalidation'
            )
        assert self._z is not None
        assert self._y is not None
        if multiprocessing:
            self._crossvalidation_out = crossvalidation_mp(
                self._y, self._z, nm, alpha
            )
        else:
            self._crossvalidation_out = crossvalidation(
                self._y, self._z, nm, alpha
            )
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def _get_index_from_sy(self, arr: npt.NDArray[np.int64], sy: int) -> int:
        index = 0
        while index < len(arr) and arr[index] != sy:
            index += 1

        if index > len(arr) - 1 or arr[index] > sy:
            raise ValueError(
                f'Selected Year {sy} is not valid\nNOTE: Valid years {arr}'
            )
        return index

    def _plot_title(self, fig: plt.Figure, alpha: Optional[float] = None,
                    yslise: Optional[Slise] = None, zslise: Optional[Slise] = None) -> None:
        fig.suptitle(
            f'{(f"Y: {slise2str(yslise)}" if yslise is not None else "")}'
            f'{(", " if yslise is not None and zslise is not None else "")}'
            f'{(f"Z: {slise2str(zslise)}" if zslise is not None else "")}'
            f'{(". " if yslise is not None or zslise is not None and alpha is not None else "")}'
            f'{(f"Alpha: {alpha}" if alpha is not None else "")}'
        )

    def _plot_map(
        self,
        arr: npt.NDArray[np.float32],
        lat: npt.NDArray[np.float32],
        lon: npt.NDArray[np.float32],
        fig: plt.Figure,
        ax: plt.Axes,
        title: Optional[str] = None,
        levels: Optional[npt.NDArray[np.float32]] = None,
        xlim: Optional[Sequence[int]] = None,
        ylim: Optional[Sequence[int]] = None,
        cmap: Optional[str] = None
    ) -> None:

        if levels is None:
            n = 30
            _std = np.nanstd(arr)
            _m = np.nanmean(arr)
            levels = np.linspace(_m - _std, _m + _std, n)
        else:
            n = len(levels)

        cmap = 'bwr' if cmap is None else cmap
        xlim = sorted((lon[0], lon[-1])) if xlim is None else xlim
        ylim = sorted((lat[-1], lat[0])) if ylim is None else ylim

        im = ax.contourf(
            lon, lat, arr, cmap=cmap, levels=levels,
            extend='both', transform=ccrs.PlateCarree()
        )
        fig.colorbar(
            im, ax=ax, orientation='horizontal', pad=0.02,
            ticks=np.concatenate(
                (levels[::n // 4], levels[-1:len(levels)])
            )
        )
        ax.coastlines()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        # # axs.margins(0)
        if title is not None:
            ax.set_title(title)

    def _apply_flags_to_fig(self, fig: plt.Figure, path: str,
                            flags: int) -> None:
        if type(flags) == int:
            flags = F(flags)
        assert type(flags) == F
        if F.SAVE_FIG in flags:
            for _ in range(2):
                try:
                    fig.savefig(path)
                    break
                except FileNotFoundError:
                    os.mkdir(path.split('/')[0])
        if F.SHOW_PLOT in flags:
            fig.show()
        if F.NOT_HALT not in flags:
            plt.show()

    def plot_preprocessed(self, flags: int = 0, fig: plt.Figure = None,
                          selected_year: Optional[int] = None,
                          cmap: Optional[str] = None,
                          yslise: Optional[Slise] = None,
                          zslise: Optional[Slise] = None) -> None:
        if any([
            x is None for x in (
                self._y, self._ylat, self._ylon, self._ytime,
                self._z, self._zlat, self._zlon, self._ztime
            )
        ]):
            nones = [n for n, v in (
                ("y", self._y), ("ylat", self._ylat),
                ("ylon", self._ylon), ("ytime", self._ytime),
                ("z", self._z), ("zlat", self._zlat),
                ("zlon", self._zlon), ("ztime", self._ztime)
            ) if v is None]

            raise TypeError(
                f'Must preprocess data before plotting\n\t'
                f'{", ".join(nones)} {("does" if len(nones) == 1 else "do")} '
                f'not exist'
            )
        assert (
            self._y is not None and
            self._ylat is not None and
            self._ylon is not None and
            self._ytime is not None and
            self._z is not None and
            self._zlat is not None and
            self._zlon is not None and
            self._ztime is not None
        )

        nyt, nylat, nylon = len(self._ytime), len(self._ylat), len(self._ylon)
        nzt, nzlat, nzlon = len(self._ztime), len(self._zlat), len(self._zlon)

        y = self._y.transpose().reshape((nyt, nylat, nylon))
        z = self._z.transpose().reshape((nzt, nzlat, nzlon))

        yindex = 0 if selected_year is None \
            else self._get_index_from_sy(self._ytime, selected_year)
        zindex = 0 if selected_year is None \
            else self._get_index_from_sy(self._ztime, selected_year)

        fig = self._new_figure(figsize=(15, 10)) if fig is None else fig
        axs = fig.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})

        self._plot_map(
            y[yindex], self._ylat, self._ylon, fig, axs[0],
            f'Y on year {self._ytime[yindex]}'
        )
        self._plot_map(
            z[zindex], self._zlat, self._zlon, fig, axs[1],
            f'Z on year {self._ztime[zindex]}',
            cmap=cmap
        )

        self._plot_title(fig, yslise=yslise, zslise=zslise)

        self._apply_flags_to_fig(
            fig, os.path.join(self._plot_dir, self._mats_plot_name), flags
        )

    def plot_mca(self, flags: int = 0, fig: Optional[plt.Figure] = None,
                 cmap: Optional[str] = None,
                 yslise: Optional[Slise] = None,
                 zslise: Optional[Slise] = None) -> 'Spy4Caster':
        if any([
            x is None for x in (
                self._y, self._ylat, self._ylon, self._ytime,
                self._z, self._zlat, self._zlon, self._ztime
            )]
        ):
            print(
                '[ERROR] Can not plot mca. '
                'No preprocessing or data loading',
                file=sys.stderr
            )
            return self

        assert (
            self._y is not None and
            self._ylat is not None and
            self._ylon is not None and
            self._ytime is not None and
            self._z is not None and
            self._zlat is not None and
            self._zlon is not None and
            self._ztime is not None
        )
        cmap = cmap if cmap is not None else 'bwr'
        if self._mca_out is None:
            print(
                '[ERROR] Can not plot mca. '
                'Methodology was not applied or loaded yet',
                file=sys.stderr
            )
            return self

        fig = fig if fig is not None else self._new_figure(figsize=(15, 10))

        ylats = self._ylat
        yts = self._ytime
        zts = self._ztime
        ylons = self._ylon
        zlats = self._zlat
        zlons = self._zlon

        us = self._mca_out.Us
        vs = self._mca_out.Vs
        scf = self._mca_out.scf

        suy = self._mca_out.SUY
        suz = self._mca_out.SUZ

        ruz_sig = self._mca_out.RUZ_sig
        ruy_sig = self._mca_out.RUY_sig

        nrows = 3
        ncols = 3
        axs = [
            plt.subplot(nrows * 100 + ncols * 10 + i,
                        projection=(ccrs.PlateCarree() if i > 3 else None))
            for i in range(1, ncols * nrows + 1)
        ]

        # Plot timeseries
        for i, ax in enumerate(axs[:3]):
            # # ax.margins(0)
            ax.plot(yts, us[i, :], color='green', label='Us')
            ax.plot(zts, vs[i, :], color='blue', label='Vs')
            ax.grid(True)
            ax.set_title(f'Us Vs mode {i + 1}')
        axs[0].legend(loc='upper left')

        # suy = SUY
        # suy[suy == 0.0] = np.nan

        n = 20
        for i, (name, su, ru, lats, lons, cm) in enumerate((
                ('SUY', suy, ruy_sig, ylats, ylons, 'bwr'),
                ('SUZ', suz, ruz_sig, zlats, zlons, cmap)
        )):
            _std = np.nanstd(su)
            _m = np.nanmean(su)
            levels = np.linspace(_m - _std, _m + _std, n)
            xlim = sorted((lons[0], lons[-1]))
            ylim = sorted((lats[-1], lats[0]))

            for j, ax in enumerate(axs[3 * (i + 1):3 * (i + 1) + 3]):
                title = f'{name} mode {j + 1}. ' \
                        f'SCF={scf[j]*100:.01f}'
                t = su[:, j].transpose().reshape((len(lats), len(lons)))
                th = ru[:, j].transpose().reshape((len(lats), len(lons)))
                th[t < 0] = np.nan

                self._plot_map(
                    t, lats, lons, fig, ax, title,
                    levels=levels, xlim=xlim, ylim=ylim, cmap=cm
                )
                ax.contourf(
                    lons, lats, th, colors='none', hatches='..', extend='both',
                    transform=ccrs.PlateCarree()
                )

        self._plot_title(fig, self._mca_out.alpha, yslise=yslise, zslise=zslise)

        self._apply_flags_to_fig(
            fig, os.path.join(self._plot_dir, self._mca_plot_name), flags
        )

        return self

    def plot_zhat(self, flags: int = 0, fig: Optional[plt.Figure] = None,
                  sy: Optional[int] = None,
                  cmap: Optional[str] = None,
                  yslise: Optional[Slise] = None,
                  zslise: Optional[Slise] = None) -> 'Spy4Caster':
        """
        Paramaters:
          - sy: Predicted year to show
        Plot:
          - zhat: Use `sy` to plot zhat on that year
          - z: Use `sy` to plot z on that year
        """
        cmap = 'bwr' if cmap is None else cmap
        if sy is None:
            print(
                '[ERROR] Could not create zhat plot, '
                'the selected_year was not provided',
                file=sys.stderr
            )
            return self

        if self._crossvalidation_out is None:
            print(
                '[ERROR] Could not create zhat plot, '
                'the methodology has not been applied or loaded yet',
                file=sys.stderr
            )
            return self

        if any([
            x is None for x in (
                self._y, self._ylat, self._ylon, self._ytime,
                self._z, self._zlat, self._zlon, self._ztime
            )]
        ):
            print(
                '[ERROR] Could not create zhat plot, the methodology '
                'has not been applied or loaded yet',
                file=sys.stderr
            )
            return self

        assert (
            self._y is not None and
            self._ylat is not None and
            self._ylon is not None and
            self._ytime is not None and
            self._z is not None and
            self._zlat is not None and
            self._zlon is not None and
            self._ztime is not None
        )

        fig = self._new_figure(figsize=(15, 10)) if fig is None else fig
        ax0 = plt.subplot(311, projection=ccrs.PlateCarree())
        ax1 = plt.subplot(312, projection=ccrs.PlateCarree())
        ax2 = plt.subplot(313, projection=ccrs.PlateCarree())

        zlats = self._zlat
        ylats = self._ylat
        yts = self._ytime
        zts = self._ztime
        zlons = self._zlon
        ylons = self._ylon
        zhat = self._crossvalidation_out.zhat
        alpha = self._crossvalidation_out.alpha

        zindex = self._get_index_from_sy(zts, sy)
        y_year = self._ytime[zindex]
        yindex = zindex

        nts, nylats, nylons = len(yts), len(ylats), len(ylons)
        d0 = self._y.transpose().reshape((nts, nylats, nylons))

        self._plot_map(d0[yindex], ylats, ylons, fig, ax0, f'Y on year {y_year}')

        nts, nzlats, nzlons = len(zts), len(zlats), len(zlons)
        d1 = zhat.transpose().reshape((nts, nzlats, nzlons))
        d2 = self._z.transpose().reshape((nts, nzlats, nzlons))

        n = 30
        _std = np.nanstd(d2[zindex])
        _m = np.nanmean(d2[zindex])
        levels = np.linspace(_m - _std, _m + _std, n)

        self._plot_map(
            d1[zindex], zlats, zlons, fig, ax1, f'Zhat on year {sy}',
            cmap=cmap, levels=levels
        )
        self._plot_map(
            d2[zindex], zlats, zlons, fig, ax2, f'Z on year {sy}',
            cmap=cmap, levels=levels
        )

        self._plot_title(fig, alpha, yslise=yslise, zslise=zslise)

        self._apply_flags_to_fig(
            fig, os.path.join(self._plot_dir, self._zhat_plot_name), flags
        )

        return self

    def plot_crossvalidation(self, flags: int = 0,
                             fig: Optional[plt.Figure] = None,
                             cmap: Optional[str] = None,
                             yslise: Optional[Slise] = None,
                             zslise: Optional[Slise] = None) -> 'Spy4Caster':
        """
        Plots:
          - r_z_zhat_s and p_z_zhat_s: Cartopy map of r
            and then hatches when p is <= alpha
          - r_z_zhat_t and p_z_zhat_t: Bar plot of r
            and then points when p is <= alpha
          - scf: Draw scf for all times for mode i.
            For the time being all in one plot
          - US
        """

        # Layout:
        #    r_z_zhat_s    r_z_zhat_t
        #       scf           r_uv_1
        #     r_uv_1          r_uv_2

        if any([
            x is None for x in (
                self._y, self._ylat, self._ylon, self._ytime,
                self._z, self._zlat, self._zlon, self._ztime
            )
        ]):
            print(
                '[ERROR] Could not create crossvalidation plot, '
                'must apply preprocessing first',
                file=sys.stderr
            )
            return self
        if self._crossvalidation_out is None:
            print(
                '[ERROR] Could not create crossvalidation plot, the '
                'methodology has not been applied or loaded yet',
                file=sys.stderr
            )
            return self
        # if self._mca_out is None:
        #     print(
        #         '[ERROR] Could not create crossvalidation plot, MCA has not '
        #         'been applied or loaded yet',
        #         file=sys.stderr
        #     )
        #     return self

        assert (
            self._y is not None and
            self._ylat is not None and
            self._ylon is not None and
            self._ytime is not None and
            self._z is not None and
            self._zlat is not None and
            self._zlon is not None and
            self._ztime is not None
        )

        cmap = 'bwr' if cmap is None else cmap

        fig = self._new_figure(figsize=(15, 10)) if fig is None else fig
        nrows = 3
        ncols = 2
        axs = [
            plt.subplot(nrows * 100 + ncols * 10 + i,
                        projection=(ccrs.PlateCarree() if i == 1 else None))
            for i in range(1, ncols * nrows + 1)
        ]

        r_z_zhat_s = self._crossvalidation_out.r_z_zhat_s
        p_z_zhat_s = self._crossvalidation_out.p_z_zhat_s
        r_z_zhat_t = self._crossvalidation_out.r_z_zhat_t
        p_z_zhat_t = self._crossvalidation_out.p_z_zhat_t
        scf = self._crossvalidation_out.scf
        # r_uv = self._crossvalidation_out.r_uv
        # p_uv = self._crossvalidation_out.p_uv

        alpha = self._crossvalidation_out.alpha
        # nyt, nylat, nylon = self._y_data.shape
        zlats = self._zlat
        zts = self._ztime
        yts = self._ytime
        zlons = self._zlon

        nzlat = len(zlats)
        nzlon = len(zlons)
        # nztime = len(ts)

        # ------ r_z_zhat_s and p_z_zhat_s ------ #
        d = r_z_zhat_s.transpose().reshape((nzlat, nzlon))
        self._plot_map(
            d, zlats, zlons, fig, axs[0],
            'Correlation in space between z and zhat',
            cmap=cmap
        )
        hatches = d.copy()
        hatches[((p_z_zhat_s > alpha) & (r_z_zhat_s > 0)).transpose().reshape((nzlat, nzlon))] = np.nan

        axs[0].contourf(
            zlons, zlats, hatches,
            colors='none', hatches='..', extend='both',
            transform=ccrs.PlateCarree()
        )
        # ^^^^^^ r_z_zhat_s and p_z_zhat_s ^^^^^^ #

        # ------ r_z_zhat_t and p_z_zhat_t ------ #
        axs[1].bar(zts, r_z_zhat_t)
        axs[1].scatter(
            zts[p_z_zhat_t <= alpha], r_z_zhat_t[p_z_zhat_t <= alpha]
        )
        axs[1].set_title('Correlation in space between z and zhat')
        # ^^^^^^ r_z_zhat_t and p_z_zhat_t ^^^^^^ #

        # ------ scf ------ #
        for mode in range(scf.shape[0]):
            axs[2].plot(yts, scf[mode], label=f'Mode {mode + 1}')
        axs[2].legend()
        axs[2].set_title('Squared convariance fraction')
        # ^^^^^^ scf ^^^^^^ #

        # ^^^^^^ Us ^^^^^^ #
        mean = self._crossvalidation_out.us.mean(2)
        _std = np.std(self._crossvalidation_out.us, axis=2)
        mx = mean + _std
        mn = mean - _std
        for mode in range(mean.shape[0]):
            axs[3 + mode].grid(True)
            axs[3 + mode].bar(
                yts, mx[mode] - mn[mode], bottom=mn[mode], color='purple',
                width=.2
            )
            axs[3 + mode].plot(
                yts, mean[mode], label='Mean', color='orange', linewidth=3
            )
            axs[3 + mode].plot(
                yts, mx[mode], label='Mean', color='r', linewidth=.5, alpha=.5
            )
            axs[3 + mode].plot(
                yts, mn[mode], label='Mean', color='r', linewidth=.5, alpha=.5
            )
            axs[3 + mode].set_title(
                f'Us for mode {mode + 1}'
            )
            axs[3 + mode].legend()

        self._plot_title(fig, alpha, yslise=yslise, zslise=zslise)

        self._apply_flags_to_fig(
            fig, os.path.join(self._plot_dir, self._cross_plot_name), flags
        )

        return self

    def create_plot(
            self, fig: Any = None, **kw: Union[str, int]
    ) -> 'Spy4Caster':

        sy = kw.pop('sy')
        if sy is not None and type(sy) != int:
            raise TypeError(f'Expected type int for `cmap`, got {type(int)}')

        cmap = kw.pop('cmap')
        if cmap is not None and type(cmap) != str:
            raise TypeError(
                f'Expected type str for `cmap`, got {type(cmap)}'
            )

        if len(kw) != 0:
            raise TypeError(
                f'`create_plot` only accepts the following keyword '
                f'arguments: `sy` and `cmap`. '
                f'Got: {", ".join(kw.keys())}'
            )
        self.plot_mca(
            fig=(fig[0] if fig is not None else None), cmap=cmap
        )
        self.plot_crossvalidation(
            fig=(fig[1] if fig is not None else None), cmap=cmap
        )
        self.plot_zhat(
            fig=(fig[2] if fig is not None else None), cmap=cmap, sy=sy
        )
        self._close_figures()
        return self

    @staticmethod
    def save_output(
        name: str,
        variables: Union[
            Dict[
                str,
                Union[
                     npt.NDArray[np.float32], npt.NDArray[np.int64]
                ]
            ], MCAOut, CrossvalidationOut
        ]
    ) -> None:
        if type(variables) != dict:
            d = variables.__dict__
        else:
            d = variables

        for k, v in d.items():
            if type(v) == np.ma.MaskedArray:
                v = v.data
            for _ in range(2):
                try:
                    np.save(name + '_' + k, v)
                    break
                except FileNotFoundError:
                    os.mkdir(name.split('/')[0])
                except Exception:
                    traceback.print_exc()

    def save_fig_data(self) -> 'Spy4Caster':
        if any([
            x is None for x in (
                self._y, self._ylat,
                self._ylon, self._ytime,
                self._z, self._zlat,
                self._zlon, self._ztime
            )]
        ):
            print('[ERROR] No preprocessed data to save', file=sys.stderr)
        else:
            path = os.path.join(
                self._plot_data_dir,
                f"save_preprocessed{self._plot_data_sufix}"
            )
            debugprint(f'[INFO] Saving Preprocessed data in `{path}*.npy`')
            assert (
                self._y is not None and
                self._ylat is not None and
                self._ylon is not None and
                self._ytime is not None and
                self._z is not None and
                self._zlat is not None and
                self._zlon is not None and
                self._ztime is not None
            )
            self.save_output(
                path,
                {
                    'y': self._y, 'ylat': self._ylat,
                    'ylon': self._ylon, 'ytime': self._ytime,
                    'z': self._z, 'zlat': self._zlat,
                    'zlon': self._zlon, 'ztime': self._ztime,
                }
            )

        if self._mca_out is None:
            print('[ERROR] No MCA data to save', file=sys.stderr)
        else:
            path = os.path.join(
                self._plot_data_dir, f"save_mca{self._plot_data_sufix}"
            )
            debugprint(f'[INFO] Saving MCA data in `{path}*.npy`')
            self.save_output(path, self._mca_out)

        if self._crossvalidation_out is None:
            print('[ERROR] No Crossvalidation data to save', file=sys.stderr)
        else:
            path = os.path.join(
                self._plot_data_dir, f"save_cross{self._plot_data_sufix}"
            )
            debugprint(
                f'[INFO] Saving crossvalidation data in '
                f'`{path}*.npy`')
            self.save_output(path, self._crossvalidation_out)

        return self

    def run(self, flags: int = 0, **kwargs: Any) -> 'Spy4Caster':
        sy = kwargs.pop('sy', None)
        cmap = kwargs.pop('cmap', None)
        yslise = kwargs.pop('yslise', None)
        zslise = kwargs.pop('zslise', None)
        flags = F(flags) if type(flags) == int else flags
        assert type(flags) == F
        if len(kwargs) != 0:
            raise TypeError(
                f'`run` takes only the following keyword '
                f'arguments: `sy` and `cmap`. '
                f'Got {", ".join(kwargs.keys())}'
            )

        # Save the data if needed
        if F.SAVE_DATA in flags:
            try:
                self.save_fig_data()
            except Exception as e:
                traceback.print_exc()
                if F.SILENT_ERRORS not in flags:
                    raise DataSavingError(str(e)) from e

        # Create the plot
        if F.SHOW_PLOT in flags or F.SAVE_FIG in flags:
            try:
                self.plot_mca(
                    flags & ~F.SHOW_PLOT | F.NOT_HALT, cmap=cmap, yslise=yslise, zslise=zslise
                )
                self.plot_crossvalidation(
                    flags & ~F.SHOW_PLOT | F.NOT_HALT, cmap=cmap, yslise=yslise, zslise=zslise
                )
                self.plot_zhat(
                    flags & ~F.SHOW_PLOT | F.NOT_HALT, sy=sy, cmap=cmap, yslise=yslise, zslise=zslise
                )
                if flags & F.SHOW_PLOT:
                    plt.show()
                self._close_figures()
            except Spy4CastError:
                raise
            except Exception as e:
                traceback.print_exc()
                if F.SILENT_ERRORS not in flags:
                    raise PlotCreationError(str(e)) from e
        return self
