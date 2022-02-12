import sys
import os
import traceback
from typing import Union, Dict, Optional, Any

import xarray as xr
from scipy import signal
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .functions import debugprint, time_from_here, time_to_here
from .stypes import Slise, F, RDArgs, RDArgsDict
from .errors import DataSavingError, Spy4CastError, PlotCreationError
from .read_data import ReadData, NAN_VAL
from .meteo import Meteo, MCAOut, CrossvalidationOut

__all__ = ['Spy4Caster']


class Spy4Caster:
    def __init__(self, yargs: Union[RDArgs, RDArgsDict], zargs: Union[RDArgs, RDArgsDict],
                 plot_dir: str = '', mca_plot_name: str = 'mca_plot.png', cross_plot_name: str = 'cross_plot.png',
                 zhat_plot_name: str = 'zhat_plot.png', plot_data_dir: str = '', force_name: bool = False):
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
        self._mca_plot_name = mca_plot_name
        self._cross_plot_name = cross_plot_name
        self._zhat_plot_name = zhat_plot_name
        self._plot_data_dir = plot_data_dir
        self._force_name = force_name
        self._y: Optional[npt.NDArray[np.float64]] = None
        self._z: Optional[npt.NDArray[np.float64]] = None

    def load_datasets(self) -> 'Spy4Caster':
        debugprint(f'[INFO] Loading datasets', end='')
        time_from_here()
        self._rdy.load_dataset()
        self._rdz.load_dataset()
        self._rdy._data = self._rdy._data.where(lambda a: abs(a) < NAN_VAL).sortby(self._rdy._lon_key)
        self._rdz._data = self._rdz._data.where(lambda a: abs(a) < NAN_VAL).sortby(self._rdy._lon_key)
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def slice_datasets(self, yslise: Slise, zslise: Slise) -> 'Spy4Caster':
        debugprint(f'[INFO] Slicing datasets', end='')
        time_from_here()
        self._rdy.slice_dataset(yslise)
        self._rdz.slice_dataset(zslise)
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def preprocess(self, order: int, period: float) -> 'Spy4Caster':
        debugprint(f'[INFO] Preprocessing data', end='')
        time_from_here()
        self._rdy._data = Meteo.anom(self._rdy._data)
        self._rdy._time_key = 'year'
        self._rdz._data = Meteo.anom(self._rdz._data)
        self._rdz._time_key = 'year'

        if len(self._rdz.time) != len(self._rdy.time):
            raise ValueError(f'The number of years of the predictand must be the same as the number '
                             f'of years of the predictor: got {len(self._rdz.time)} and '
                             f'{len(self._rdy.time)}')

        b, a = signal.butter(order, 1 / period, btype='high', analog=False, output='ba', fs=None)

        nyt, nylat, nylon = self._rdy._data.shape
        nzt, nzlat, nzlon = self._rdz._data.shape
        self._z = xr.apply_ufunc(
            lambda ts: signal.filtfilt(b, a, ts),
            self._rdz._data,
            input_core_dims=[[self._rdz._time_key]], output_core_dims=[[self._rdz._time_key]]
        ).transpose(self._rdz._time_key, self._rdz._lat_key, self._rdz._lon_key).fillna(0).values.reshape(
            (nzt, nzlat * nzlon)).transpose()
        self._y = self._rdy._data.fillna(0).values.reshape((nyt, nylat * nylon)).transpose()

        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    # def preprocess_old(self, order: int, period: float) -> 'Spy4Caster':
    #     debugprint('[INFO] Preprocessing data', end='')
    #     time_from_here()
    #     self._rdy._data = Meteo.anom(self._rdy._data)
    #     self._rdy._time_key = 'year'
    #     self._rdz._data = Meteo.anom(self._rdz._data)
    #     self._rdz._time_key = 'year'
    #
    #     y0 = max(self._rdy._slise.year0, self._rdz._slise.year0)
    #     yf = min(self._rdy._slise.yearf, self._rdz._slise.yearf)
    #
    #     self._rdy.slice_dataset(Slise.default(year0=y0, yearf=yf))
    #     self._rdz.slice_dataset(Slise.default(year0=y0, yearf=yf))
    #
    #     z = self._rdz._data.values
    #     # zlon = self._rdz._data.lon.values
    #     # zlat = self._rdz._data.lat.values
    #     ztrans = np.reshape(z, (z.shape[0], z.shape[1] * z.shape[2])).transpose()
    #
    #     y = self._rdy._data.values
    #     # ylon = self._rdy._data.longitude.values
    #     # ylat = self._rdy._data.latitude.values
    #     ytrans = np.reshape(y, (y.shape[0], y.shape[1] * y.shape[2])).transpose()
    #
    #     b, a = signal.butter(order, 1/period, btype='high', analog=False, output='ba', fs=None)
    #
    #     # Filtro la señal ampliada y me quedo con la parte central:
    #     zmask = np.ma.empty(ztrans.shape)
    #     for index in range(ztrans.shape[0]):
    #         zmask[index, :] = signal.filtfilt(b, a, ztrans[index, :])
    #
    #     # zmask, zlon, zlat; ytrans, ylon, ylat
    #     self._y = np.nan_to_num(ytrans)  # fill nan with 0
    #     self._z = np.nan_to_num(zmask)  # fill nan with 0
    #     debugprint(f' took: {time_to_here():.03f} seconds')
    #     return self

    def mca(self, nm: int, alpha: float) -> 'Spy4Caster':
        debugprint(f'[INFO] Applying MCA', end='')
        time_from_here()
        if self._y is None or self._z is None:
            raise TypeError('Must prprocess data before applying MCA')
        self._mca_out = Meteo.mca(self._z, self._y, 1, nm, alpha)
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def load_crossvalidation(self, path0: str, prefix: str = '', ext: str = '.npy') -> 'Spy4Caster':
        debugprint(f'[INFO] Loading Crossvalidation data from `{path0}/{prefix}*{ext}`', end='')
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
            'alpha': None,
        }
        for key in out.keys():
            path = os.path.join(path0, f'{prefix}{key}{ext}')
            try:
                out[key] = np.load(path)
            except FileNotFoundError:
                print('\n[ERROR] Could not find file {path} for variable {key}')

        self._crossvalidation_out = CrossvalidationOut(
            zhat=out['zhat'],
            scf=out['scf'],
            r_z_zhat_t=out['r_z_zhat_t'],
            p_z_zhat_t=out['p_z_zhat_t'],
            r_z_zhat_s=out['r_z_zhat_s'],
            p_z_zhat_s=out['p_z_zhat_s'],
            r_uv=out['r_uv'],
            p_uv=out['p_uv'],
            alpha=out['alpha'],
        )
        print(f' took {time_to_here():.03f} seconds')
        return self

    def load_mca(self, path0: str, prefix: str = '', ext: str = '.npy') -> 'Spy4Caster':
        debugprint(f'[INFO] Loading MCA data from `{path0}/{prefix}*{ext}`', end='')
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
        }
        for key in out.keys():
            path = os.path.join(path0, f'{prefix}{key}{ext}')
            try:
                out[key] = np.load(path)
            except FileNotFoundError:
                print(f'[ERROR] Could not find file {path} for variable {key}')

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
        )
        print(f' took: {time_to_here():.03f}')
        return self

    def crossvalidation(self, nm: int, alpha: float, multiprocessing: bool) -> 'Spy4Caster':
        debugprint(f'[INFO] Applying crossvalidation {"(mp) " if multiprocessing else ""}', end='')
        time_from_here()
        if self._y is None or self._z is None:
            raise TypeError('Must prprocess data before applying Crossvalidation')
        if multiprocessing:
            self._crossvalidation_out = Meteo.crossvalidation_mp(self._y, self._z, 1, nm, alpha)
        else:
            self._crossvalidation_out = Meteo.crossvalidation(self._y, self._z, 1, nm, alpha)
        debugprint(f' took: {time_to_here():.03f} seconds')
        return self

    def plot_matrices(self):
        # sst = self._rdy._data.values
        # sst_lon = self._rdy.lon
        # sst_lat = self._rdy.lat
        # sst_time = self._rdy.time
        # slp = self._rdz._data.values
        # slp_lat = self._rdz.lat
        # slp_lon = self._rdz.lon
        # slp_time = self._rdz.time
        #
        # def plot_matrices(y, ytime, ylats, ylons, z, ztime, zlats, zlons):
        #     fig = plt.figure()
        #     ax0, ax1 = fig.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
        #     t0 = z[0, :, :]
        #     im0 = ax0.contourf(zlons, zlats, t0, cmap='viridis', transform=ccrs.PlateCarree())
        #     fig.colorbar(im0, ax=ax0, orientation='horizontal', pad=0.02)
        #     ax0.coastlines()
        #     t1 = y[10, :, :]
        #     im1 = ax1.contourf(ylons, ylats, t1, cmap='viridis', transform=ccrs.PlateCarree())
        #     fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.02)
        #     ax1.coastlines()
        #     plt.show()
        #
        # plot_matrices(sst, sst_time, sst_lat, sst_lon, slp, slp_time, slp_lat, slp_lon)

        fig = plt.figure()
        ax0, ax1 = fig.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
        # print(self._rdy._data.values)
        t0 = self._rdz._data.values[0, :, :]
        im0 = ax0.contourf(self._rdz.lon, self._rdz.lat,
                           t0, cmap='viridis', transform=ccrs.PlateCarree())
        fig.colorbar(im0, ax=ax0, orientation='horizontal', pad=0.02)
        ax0.coastlines()

        t1 = self._rdy._data.values[0, :, :]
        im1 = ax1.contourf(self._rdy.lon, self._rdy.lat,
                           t1, cmap='viridis', transform=ccrs.PlateCarree())
        fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.02)
        ax1.coastlines()
        plt.show()

        return self

    def plot_mca(self, flags: int = 0, fig: plt.Figure = None) -> 'Spy4Caster':
        if self._mca_out is None:
            print('[WARNING] Can not plot mca. Methodology was not applied yet', file=sys.stderr)
            return self

        fig = fig if fig is not None else plt.figure()

        ylats = self._rdy.lat
        ts = self._rdy.time
        ylons = self._rdy.lon
        zlats = self._rdz.lat
        zlons = self._rdz.lon

        nrows = 3
        ncols = 3
        axs = [plt.subplot(nrows * 100 + ncols * 10 + i, projection=(ccrs.PlateCarree() if i > 3 else None))
               for i in range(1, ncols * nrows + 1)]

        # Plot timeseries
        for i, ax in enumerate(axs[:3]):
            ax.margins(0)
            ax.plot(ts, self._mca_out.Us[i, :], color='green', label=f'Us')
            ax.plot(ts, self._mca_out.Vs[i, :], color='blue', label=f'Vs')
            ax.grid(True)
            ax.set_title(f'Us Vs mode {i}')
        axs[0].legend(loc='upper left')

        n = 20
        for i, (name, su, ru, lats, lons) in enumerate((
                ('SUY', self._mca_out.SUY, self._mca_out.RUY_sig, ylats, ylons),
                ('SUZ', self._mca_out.SUZ, self._mca_out.RUZ_sig, zlats, zlons)
        )):
            if i == 0:
                su[su == 0.0] = np.nan
            _std = np.nanstd(su)
            levels = np.linspace(-_std, _std, n)
            xlim = lons[0], lons[-1]
            ylim = lats[-1], lats[0]
            for j, ax in enumerate(axs[3 * (i + 1):3 * (i + 1) + 3]):
                t = su[:, j].transpose().reshape((len(lats), len(lons)))
                th = ru[:, j].transpose().reshape((len(lats), len(lons)))
                im = ax.contourf(lons, lats, t, cmap='bwr', levels=levels,
                                 extend='both', transform=ccrs.PlateCarree())
                _imh = ax.contourf(lons, lats, th, colors='none', hatches='..', extend='both',
                                   transform=ccrs.PlateCarree())
                _cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                                   ticks=np.concatenate((levels[::n // 4], levels[-1:len(levels)])))
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.margins(0)
                ax.coastlines()
                ax.set_title(f'{name} mode {j}')

        plt.tight_layout()

        if F.checkf(F.SAVE_FIG, flags):
            fig.savefig(self._mca_plot_name)
        if F.checkf(F.SHOW_PLOT, flags):
            fig.show()

        return self

    def plot_zhat(self, flags: int = 0, fig: plt.Figure = None, sy: int = None) -> 'Spy4Caster':
        """
        Paramaters:
          - sy: Predicted year to show
        Plot: zhat: Use `sy` to plot zhat on that year
        """
        if sy is None:
            raise TypeError('`sy` argument must be provided')

        if self._crossvalidation_out is None:
            print(f'[ERROR] Could not create zhat plot, the methodology has not been applied yet',
                  file=sys.stderr)
            return self

        fig = plt.figure() if fig is None else fig

        lats = self._rdz.lat
        ts = self._rdy.time
        lons = self._rdz.lon
        zhat = self._crossvalidation_out.zhat

        nts, nlats, nlons = len(ts), len(lats), len(lons)
        n = 20
        _std = np.nanstd(zhat)
        levels = np.linspace(-_std, _std, n)
        xlim = lons[0], lons[-1]
        ylim = lats[-1], lats[0]
        index = 0
        while index < len(ts) and ts.values[index] != sy:  index += 1
        if ts.values[index] != sy: raise ValueError(f'Selected Year {sy} is not valid')

        ax = plt.subplot(111, projection=ccrs.PlateCarree())
        d = zhat.transpose().reshape((nts, nlats, nlons))
        im = ax.contourf(lons, lats, d[index],
                         transform=ccrs.PlateCarree(), cmap='bwr', levels=levels, extend='both')
        _cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                           ticks=np.concatenate((levels[::n // 4], levels[-1:len(levels)])))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.margins(0)
        ax.coastlines()
        ax.set_title(f'Zhat on year {sy}')

        if F.checkf(F.SAVE_FIG, flags):
            fig.savefig(self._zhat_plot_name)
        if F.checkf(F.SHOW_PLOT, flags):
            fig.show()
        return self

    def plot_crossvalidation(self, flags: int = 0, fig: plt.Figure = None) -> 'Spy4Caster':
        """
        Plots:
          - r_z_zhat_s and p_z_zhat_s: Cartopy map of r and then hatches when p is <= alpha
          - r_z_zhat_t and p_z_zhat_t: Bar plot of r and then points when p is <= alpha
          - scf: Draw scf for all times for mode i. For the time being all in one plot
          - r_uv and p_uv: Same as scf and points when p <= alpha
        Layout:
           r_z_zhat_s    r_z_zhat_t
              scf           r_uv_1
            r_uv_1          r_uv_2
        """
        fig = plt.figure() if fig is None else fig
        if self._crossvalidation_out is None:
            print(f'[ERROR] Could not create crossvalidation plot, the methodology has not been applied yet',
                  file=sys.stderr)
            return self

        r_z_zhat_s = self._crossvalidation_out.r_z_zhat_s
        p_z_zhat_s = self._crossvalidation_out.p_z_zhat_s
        r_z_zhat_t = self._crossvalidation_out.r_z_zhat_t
        p_z_zhat_t = self._crossvalidation_out.p_z_zhat_t
        scf = self._crossvalidation_out.scf
        r_uv = self._crossvalidation_out.r_uv
        p_uv = self._crossvalidation_out.p_uv

        alpha = self._crossvalidation_out.alpha
        # nyt, nylat, nylon = self._rdy._data.shape
        nzt, nzlat, nzlon = self._rdz._data.shape
        zlats = self._rdz.lat
        ts = self._rdz.time
        zlons = self._rdz.lon

        nrows = 3
        ncols = 2
        axs = [plt.subplot(nrows * 100 + ncols * 10 + i, projection=(ccrs.PlateCarree() if i == 1 else None))
               for i in range(1, ncols * nrows + 1)]

        n = 20

        # ------ r_z_zhat_s and p_z_zhat_s ------ #
        _std = np.nanstd(r_z_zhat_s)
        levels = np.linspace(-_std, _std, n)
        xlim = zlons[0], zlons[-1]
        ylim = zlats[-1], zlats[0]

        d = r_z_zhat_s.transpose().reshape((nzlat, nzlon))
        hatches = d.copy()
        hatches[(p_z_zhat_s > alpha).transpose().reshape((nzlat, nzlon))] = np.nan
        im = axs[0].contourf(zlons, zlats, d,
                             cmap='bwr', levels=levels, extend='both', transform=ccrs.PlateCarree())
        _imh = axs[0].contourf(zlons, zlats, hatches,
                               colors='none', hatches='..', extend='both', transform=ccrs.PlateCarree())
        _cb = fig.colorbar(im, ax=axs[0], orientation='horizontal',
                           ticks=np.concatenate((levels[::n // 4], levels[-1:len(levels)])))
        axs[0].set_xlim(xlim)
        axs[0].set_ylim(ylim)
        axs[0].margins(0)
        axs[0].coastlines()
        axs[0].set_title(f'Correlation in space between z and zhat')
        # ^^^^^^ r_z_zhat_s and p_z_zhat_s ^^^^^^ #

        # TODO: Add titles and legends
        # ------ r_z_zhat_t and p_z_zhat_t ------ #
        axs[1].bar(ts, r_z_zhat_t)
        axs[1].scatter(ts[p_z_zhat_t <= alpha], r_z_zhat_t[p_z_zhat_t <= alpha])
        axs[1].set_title(f'Correlation in space between z and zhat')
        # ^^^^^^ r_z_zhat_t and p_z_zhat_t ^^^^^^ #

        # ------ scf ------ #
        for mode in range(scf.shape[0]):
            axs[2].plot(ts, scf[mode], label=f'Mode {mode + 1}')
        axs[2].legend()
        axs[2].set_title(f'Squared convariance fraction')
        # ^^^^^^ scf ^^^^^^ #

        # TODO: Plot seperately
        # ------ r_uv and p_uv ------ #
        for mode in range(scf.shape[0]):
            axs[3 + mode].plot(ts, r_uv[mode])
            axs[3 + mode].scatter(ts[p_uv[mode] <= alpha], r_uv[mode][p_uv[mode] <= alpha])
            axs[3 + mode].set_title(f'RUV and PUV for mode {mode + 1}')
        # ^^^^^^ r_uv and p_uv ^^^^^^ #

        if F.checkf(F.SAVE_FIG, flags):
            fig.savefig(self._cross_plot_name)
        if F.checkf(F.SHOW_PLOT, flags):
            fig.show()
        return self

    def create_plot(self, fig: Any = None, **kw: int) -> 'Spy4Caster':
        sy = kw.pop('sy')
        if len(kw) != 0:
            raise TypeError('`create_plot` only accepts one argument (sy)')
        self.plot_mca(fig=(fig[0] if fig is not None else None))
        self.plot_crossvalidation(fig=(fig[1] if fig is not None else None))
        self.plot_zhat(fig=(fig[2] if fig is not None else None), sy=sy)
        return self

    @staticmethod
    def save_output(name: str, variables: Union[MCAOut, CrossvalidationOut]):
        for k, v in variables.__dict__.items():
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
        if self._mca_out is not None:
            print(f'[INFO] Saving MCA data in `saved/save_mca*.npy`')
            self.save_output('saved/save_mca', self._mca_out)
        else:
            print('[WARNING] No MCA data to save', file=sys.stderr)

        if self._crossvalidation_out is not None:
            print(f'[INFO] Saving crossvalidation data in `saved/save_cross*.npy`')
            self.save_output('saved/save_cross', self._crossvalidation_out)
        else:
            print('[WARNING] No Crossvalidation data to save', file=sys.stderr)

        return self

    def run(self, flags: int = 0, **kwargs: Any) -> 'Spy4Caster':
        # Save the data if needed
        if F.checkf(F.SAVE_DATA, flags):
            try:
                self.save_fig_data()
            except Exception as e:
                traceback.print_exc()
                if F.checkf(F.SILENT_ERRORS, flags):
                    raise DataSavingError(str(e)) from e

        # Create the plot
        if F.checkf(F.SHOW_PLOT, flags) or F.checkf(F.SAVE_FIG, flags):
            try:
                self.plot_mca(flags & ~F.SHOW_PLOT)
                self.plot_crossvalidation(flags & ~F.SHOW_PLOT)
                self.plot_zhat(flags & ~F.SHOW_PLOT, sy=kwargs.get('sy'))
                if flags & F.SHOW_PLOT:
                    plt.show()
            except Spy4CastError:
                raise
            except Exception as e:
                traceback.print_exc()
                if F.checkf(F.SILENT_ERRORS, flags):
                    raise PlotCreationError(str(e)) from e
        return self
