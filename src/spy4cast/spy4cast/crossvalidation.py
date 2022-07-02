import os
from math import floor
from typing import Tuple, Optional, Any, Union, Sequence, cast

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
from scipy import stats

from . import MCA
from .. import Slise, F
from .._functions import debugprint, slise2str, time_from_here, time_to_here, _debuginfo
from .._procedure import _Procedure, _apply_flags_to_fig, _plot_map, _get_index_from_sy, _calculate_figsize, MAX_WIDTH, \
    MAX_HEIGHT, _plot_ts
from .preprocess import Preprocess
import xarray as xr


__all__ = [
    'Crossvalidation',
]


class Crossvalidation(_Procedure):
    """Perform crossvalidation methodology

    Parameters
    ----------
        dsy : Preprocess
            Predictor field
        dsz : Preprocess
            Predictand field
        nm : int
            Number of modes
        alpha : float
            Significance coeficient
        multiprocessed : bool
            Use multiprocessing for the methodology

    Attributes
    ----------
        zhat : npt.NDArray[float32]
            Hindcast of field to predict using crosvalidation
        scf : npt.NDArray[float32]
            Squared covariance fraction of the mca for each mode
        r_z_zhat_t : npt.NDArray[float32]
            Correlation between zhat and Z for each time (time series)
        p_z_zhat_t : npt.NDArray[float32]
            P values of rt
        r_z_zhat_s : npt.NDArray[float32]
            Correlation between time series (for each point) of zhat and z (map)
        p_z_zhat_s : npt.NDArray[float32]
            P values of rr
        r_uv : npt.NDArray[float32]
            Correlation score betweeen u and v for each mode
        p_uv : npt.NDArray[float32]
            P value of ruv
        us : npt.NDArray[float32]
            crosvalidated year on axis 2
        alpha : float
            Correlation factor

    See Also
    --------
        MCA
    """
    zhat: npt.NDArray[np.float32]
    scf: npt.NDArray[np.float32]
    r_z_zhat_t: npt.NDArray[np.float32]
    p_z_zhat_t: npt.NDArray[np.float32]
    r_z_zhat_s: npt.NDArray[np.float32]
    p_z_zhat_s: npt.NDArray[np.float32]
    r_uv: npt.NDArray[np.float32]
    p_uv: npt.NDArray[np.float32]
    us: npt.NDArray[np.float32]
    alpha: float

    @property
    def var_names(self) -> Tuple[str, ...]:
        return (
            'zhat',
            'scf',
            'r_z_zhat_t',
            'p_z_zhat_t',
            'r_z_zhat_s',
            'p_z_zhat_s',
            'r_uv',
            'p_uv',
            'us',
            'alpha',
        )

    def __init__(
        self,
        dsy: Preprocess,
        dsz: Preprocess,
        nm: int,
        alpha: float,
        multiprocessed: bool = False
    ):
        self._dsz = dsz
        self._dsy = dsy

        nz, ntz = self.zdata.shape
        ny, nty = self.ydata.shape

        _debuginfo(f"""Applying Crossvalidation 
    Shapes: Z{dsz.shape} 
            Y{dsy.shape} 
    Slises: Z {slise2str(self.zslise)} 
            Y {slise2str(self.yslise)}""", )
        time_from_here()

        if len(dsz.time) != len(dsy.time):
            raise ValueError(
                f'The number of years of the predictand must be the '
                f'same as the number of years of the predictor: '
                f'got {len(dsz.time)} and '
                f'{len(dsy.time)}'
            )

        nt = ntz

        self.zhat = np.zeros_like(self.zdata, dtype=np.float32)
        self.scf = np.zeros([nm, nt], dtype=np.float32)
        self.r_uv = np.zeros([nm, nt], dtype=np.float32)
        self.p_uv = np.zeros([nm, nt], dtype=np.float32)
        # crosvalidated year on axis 2
        self.us = np.zeros([nm, nt, nt], dtype=np.float32)
        # estimación de self.zhat para cada año
        yrs = np.arange(nt)

        if multiprocessed:
            import multiprocessing as mp
            # Step 1: Init multiprocessing.Pool()
            count = mp.cpu_count()
            with mp.Pool(count) as pool:
                # print(f'Starting pool with {count=}')
                processes = []
                for i in yrs:
                    # print(f'applying async on process {i=}')
                    p = pool.apply_async(self._crossvalidate_year, kwds={
                        'year': i, 'z': self.zdata, 'y': self.ydata, 'nt': nt, 'yrs': yrs,
                        'nm': nm, 'alpha': alpha
                    })
                    processes.append(p)

                for i in yrs:
                    values = processes[i].get()
                    self.scf[:, i], self.zhat[:, i], self.r_uv[:, i], self.p_uv[:, i], \
                    self.us[:, [x for x in range(nt) if x != i], i] \
                        = values
        else:
            for i in yrs:
                out = self._crossvalidate_year(
                    year=i, z=self.zdata, y=self.ydata, nt=nt, yrs=yrs,
                    nm=nm, alpha=alpha
                )
                self.scf[:, i], self.zhat[:, i], self.r_uv[:, i], self.p_uv[:, i], \
                self.us[:, [x for x in range(nt) if x != i], i] = out

        self.r_z_zhat_t = np.zeros(nt, dtype=np.float32)
        self.p_z_zhat_t = np.zeros(nt, dtype=np.float32)
        for j in range(nt):
            rtt = stats.pearsonr(self.zhat[:, j], self.zdata[:, j])  # serie de skill
            self.r_z_zhat_t[j] = rtt[0]
            self.p_z_zhat_t[j] = rtt[1]

        self.r_z_zhat_s = np.zeros([nz], dtype=np.float32)
        self.p_z_zhat_s = np.zeros([nz], dtype=np.float32)
        for i in range(nz):
            self.r_z_zhat_s[i], self.p_z_zhat_s[i] = stats.pearsonr(self.zhat[i, :], self.zdata[i, :])
        self.alpha = alpha
        debugprint(f'\n\tTook: {time_to_here():.03f} seconds')

    def _crossvalidate_year(
        self,
        year: int,
        z: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        nt: int,
        yrs: npt.NDArray[np.int32],
        nm: int,
        alpha: float
    ) -> Tuple[npt.NDArray[np.float32], ...]:
        """Function of internal use that processes a single year for
        crossvalidation"""

        debugprint('\tyear:', year, 'of', nt)
        z2 = z[:, yrs != year]
        y2 = y[:, yrs != year]
        mca_out = MCA.from_nparrays(z2, y2, nm, alpha)
        zhat = np.dot(np.transpose(y[:, year]), mca_out.psi)

        r_uv = np.zeros(nm, dtype=np.float32)
        p_uv = np.zeros(nm, dtype=np.float32)
        for m in range(nm):
            r_uv[m], p_uv[m] = stats.pearsonr(mca_out.Us[m, :], mca_out.Vs[m, :])

        scf = mca_out.scf
        return scf, zhat, r_uv, p_uv, mca_out.Us

    @property
    def ydata(self) -> npt.NDArray[np.float32]:
        return self._dsy.data

    @property
    def yvar(self) -> str:
        return self._dsy.var

    @property
    def yslise(self) -> Slise:
        return self._dsy.slise

    @property
    def ytime(self) -> xr.DataArray:
        return self._dsy.time

    @property
    def ylat(self) -> xr.DataArray:
        return self._dsy.lat

    @property
    def ylon(self) -> xr.DataArray:
        return self._dsy.lon

    @property
    def zdata(self) -> npt.NDArray[np.float32]:
        return self._dsz.data

    @property
    def zvar(self) -> str:
        return self._dsz.var

    @property
    def zslise(self) -> Slise:
        return self._dsz.slise

    @property
    def ztime(self) -> xr.DataArray:
        return self._dsz.time

    @property
    def zlat(self) -> xr.DataArray:
        return self._dsz.lat

    @property
    def zlon(self) -> xr.DataArray:
        return self._dsz.lon

    def plot(
        self,
        flags: int = 0,
        dir: Optional[str] = None,
        name: Optional[str] = None,
        cmap: str = 'bwr',
        map_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None
    ) -> None:
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

        fig = plt.figure(figsize=_calculate_figsize(None, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
        nrows = 3
        ncols = 2
        axs = [
            plt.subplot(nrows * 100 + ncols * 10 + i,
                        projection=(ccrs.PlateCarree() if i == 1 else None))
            for i in range(1, ncols * nrows + 1)
        ]

        nzlat = len(self.zlat)
        nzlon = len(self.zlon)
        # nztime = len(ts)

        # ------ r_z_zhat_s and p_z_zhat_s ------ #
        # Correlation map
        d = self.r_z_zhat_s.transpose().reshape((nzlat, nzlon))
        _mean = np.nanmean(d)
        _std = np.nanstd(d)
        mx = _mean + _std
        mn = _mean - _std
        _plot_map(
            d, self.zlat, self.zlon, fig, axs[0],
            'Correlation in space between z and zhat',
            cmap=cmap, ticks=(np.arange(round(mn * 10) / 10, floor(mx * 10) / 10 + .05, .1) if map_ticks is None else map_ticks)
        )
        hatches = d.copy()
        hatches[((self.p_z_zhat_s > self.alpha) | (self.r_z_zhat_s < 0)).transpose().reshape((nzlat, nzlon))] = np.nan

        axs[0].contourf(
            self.zlon, self.zlat, hatches,
            colors='none', hatches='..', extend='both',
            transform=ccrs.PlateCarree()
        )
        # ^^^^^^ r_z_zhat_s and p_z_zhat_s ^^^^^^ #

        # ------ r_z_zhat_t and p_z_zhat_t ------ #
        axs[1].bar(self.ztime.values, self.r_z_zhat_t)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        axs[1].scatter(
            self.ztime[self.p_z_zhat_t <= self.alpha], self.r_z_zhat_t[self.p_z_zhat_t <= self.alpha]
        )
        axs[1].set_title('Correlation in space between z and zhat')
        axs[1].grid(True)
        # ^^^^^^ r_z_zhat_t and p_z_zhat_t ^^^^^^ #

        # ------ scf ------ #
        for mode in range(self.scf.shape[0]):
            _plot_ts(
                time=self.ytime.values,
                arr=self.scf[mode],
                ax=axs[2],
                label=f'Mode {mode + 1}',
                title='Squared convariance fraction'
            )
        axs[2].legend()
        axs[2].grid(True)
        # ^^^^^^ scf ^^^^^^ #

        # ^^^^^^ Us ^^^^^^ #
        mean = self.us.mean(2)
        std = np.std(self.us, axis=2)
        for mode in range(mean.shape[0]):
            axs[3 + mode].grid(True)
            axs[3 + mode].errorbar(
                self.ytime, mean[mode], yerr=np.abs(std[mode]), label='std', color='orange', linewidth=3, ecolor='purple'
            )
            axs[3 + mode].set_title(
                f'Us for mode {mode + 1}'
            )
            axs[3 + mode].legend()

        fig.suptitle(
            f'Z({self.zvar}): {slise2str(self.zslise)}, '
            f'Y({self.yvar}): {slise2str(self.yslise)}. '
            f'Alpha: {self.alpha}',
            fontweight='bold'
        )

        fig.subplots_adjust(hspace=.4)

        if dir is None:
            dir = '.'
        if name is None:
            path = os.path.join(dir, f'crossvalidation-plot_z-{self.zvar}_y-{self.yvar}.png')
        else:
            path = os.path.join(dir, name)

        _apply_flags_to_fig(
            fig, path, F(flags)
        )

    def plot_zhat(
        self,
        year: int,
        flags: F = F(0),
        dir: Optional[str] = None,
        name: Optional[str] = None,
        cmap: str = 'bwr',
        yticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        zticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
    ) -> None:
        nts, nylat, nylon = len(self.ytime), len(self.ylat), len(self.ylon)
        nts, nzlat, nzlon = len(self.ztime), len(self.zlat), len(self.zlon)

        height = nylat + nzlat + nzlat
        width = max(nzlon, nylon)

        fig = plt.figure(figsize=_calculate_figsize(height / width, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
        ax0 = plt.subplot(311, projection=ccrs.PlateCarree())
        ax1 = plt.subplot(312, projection=ccrs.PlateCarree())
        ax2 = plt.subplot(313, projection=ccrs.PlateCarree())

        zindex = _get_index_from_sy(self.ztime, year)
        yindex = zindex
        y_year = self.ytime.values[yindex]

        d0 = self.ydata.transpose().reshape((nts, nylat, nylon))

        _plot_map(d0[yindex], self.ylat, self.ylon, fig, ax0, f'Y on year {y_year}', ticks=yticks)

        d1 = self.zhat.transpose().reshape((nts, nzlat, nzlon))
        d2 = self.zdata.transpose().reshape((nts, nzlat, nzlon))

        n = 30
        _std = np.nanstd(d2[zindex])
        _m = np.nanmean(d2[zindex])
        bound = max(abs(_m - _std), abs(_m + _std))
        levels = np.linspace(-bound, bound, n)

        _plot_map(
            d1[zindex], self.zlat, self.zlon, fig, ax1, f'Zhat on year {year}',
            cmap=cmap, levels=levels, ticks=zticks
        )
        _plot_map(
            d2[zindex], self.zlat, self.zlon, fig, ax2, f'Z on year {year}',
            cmap=cmap, levels=levels, ticks=zticks
        )

        fig.suptitle(
            f'Z({self.zvar}): {slise2str(self.zslise)}, '
            f'Y({self.yvar}): {slise2str(self.yslise)}. '
            f'Alpha: {self.alpha}',
            fontweight='bold'
        )

        fig.subplots_adjust(hspace=.4)

        if dir is None:
            dir = '.'
        if name is None:
            path = os.path.join(dir, f'crossvalidation-plot_z-{self.zvar}_y-{self.yvar}.png')
        else:
            path = os.path.join(dir, name)

        _apply_flags_to_fig(
            fig, path, flags
        )


    @classmethod
    def load(
        cls,
        prefix: str,
        dir: str = '.',
        *,
        dsz: Optional[Preprocess] = None,
        dsy: Optional[Preprocess] = None,
        **attrs: Any
    ) -> 'Crossvalidation':
        if len(attrs) != 0:
            raise TypeError('Load only takes two keyword arguments: dsz and dsy')
        if dsz is None or dsy is None:
            raise TypeError('To load an Crossvalidation object you must provide `dsz` and `dsy` keyword arguments')
        if type(dsz) != Preprocess or type(dsy) != Preprocess:
            raise TypeError(f'Unexpected types ({type(dsz)} and {type(dsy)}) for `dsz` and `dsy`. Expected type `Preprocess`')

        self: Crossvalidation = cast(Crossvalidation, super().load(prefix, dir))
        self._dsz = dsz
        self._dsy = dsy
        return self

