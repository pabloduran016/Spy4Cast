import os
from math import floor
from typing import Tuple, Optional, Any, Union, Sequence, cast, Literal, List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
from scipy import stats

from . import MCA
from .mca import index_regression
from .. import Region
from .._functions import debugprint, region2str, time_from_here, time_to_here, _debuginfo
from .._procedure import _Procedure, _apply_flags_to_fig, _plot_map, _get_index_from_sy, _calculate_figsize, MAX_WIDTH, \
    MAX_HEIGHT, _plot_ts
from ..land_array import LandArray
from .preprocess import Preprocess
import xarray as xr


__all__ = [
    'Crossvalidation',
    'calculate_psi',
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
        sig : {'monte-carlo', 'test-t'}
            Signification technique: monte-carlo or test-t

    Attributes
    ----------
        zhat_separated_modes : npt.NDArray[float32]
            Hindcast of field to predict using crosvalidation for each mode separated
        zhat_accumulated_modes : npt.NDArray[float32]
            Hindcast of field to predict using crosvalidation accumulating modes (1, 1->2, ..., 1->nm)
        scf : npt.NDArray[float32]
            Squared covariance fraction of the mca for each mode
        r_z_zhat_t_separated_modes : npt.NDArray[float32]
            Correlation between zhat and Z for each time (time series) for each mode separated
        r_z_zhat_t_accumulated_modes : npt.NDArray[float32]
            Correlation between zhat and Z for each time (time series) accumulating modes (1, 1->2, ..., 1->nm)
        p_z_zhat_t_separated_modes : npt.NDArray[float32]
            P values of rt for each mode separated
        p_z_zhat_t_accumulated_modes : npt.NDArray[float32]
            P values of rt accumulating modes (1, 1->2, ..., 1->nm)
        r_z_zhat_s_separated_modes : npt.NDArray[float32]
            Correlation between time series (for each point) of zhat and z (map) for each mode separated
        r_z_zhat_s_accumulated_modes : npt.NDArray[float32]
            Correlation between time series (for each point) of zhat and z (map) accumulating modes (1, 1->2, ..., 1->nm)
        p_z_zhat_s_separated_modes : npt.NDArray[float32]
            P values of rr for each mode separated
        p_z_zhat_s_accumulated_modes : npt.NDArray[float32]
            P values of rr accumulating modes (1, 1->2, ..., 1->nm)
        r_uv : npt.NDArray[float32]
            Correlation score betweeen u and v for each mode
        r_uv_sig : npt.NDArray[float32]
            Correlation score betweeen u and v for each mode where significative
        p_uv : npt.NDArray[float32]
            P value of ruv
        psi_separated_modes : npt.NDArray[np.float32]
            Skill for each mode separated
        psi_accumulated_modes : npt.NDArray[np.float32]
            Skill accumulating modes (1, 1->2, ..., 1->nm)
        suy : npt.NDArray[np.float32]
            Correlation in space of the predictor with the singular vector. Dimension: y_space x time x nm
        suz : npt.NDArray[np.float32]
            Correlation in space of the predictand with the singular vector. Dimension: z_space x time x nm
        suy_sig : npt.NDArray[np.float32]
            Correlation in space of the predictor with the singular vector where pvalue is smaller than alpha. Dimension: y_space x time x nm
        suz_sig : npt.NDArray[np.float32]
            Correlation in space of the predictand with the singular vector where pvalue is smaller than alpha. Dimension: z_space x time x nm
        us : npt.NDArray[float32]
            Singular vectors of the predictor field. Dimension: nm x time x time
        vs : npt.NDArray[float32]
            Singular vectors of the predictand field. Dimension: nm x time x time
        alpha : float
            Correlation factor

    See Also
    --------
        MCA
    """
    scf: npt.NDArray[np.float32]
    r_uv: npt.NDArray[np.float32]
    r_uv_sig: npt.NDArray[np.float32]
    p_uv: npt.NDArray[np.float32]
    us: npt.NDArray[np.float32]
    vs: npt.NDArray[np.float32]
    zhat_separated_modes: npt.NDArray[np.float32]
    zhat_accumulated_modes: npt.NDArray[np.float32]
    psi_separated_modes: npt.NDArray[np.float32]
    psi_accumulated_modes: npt.NDArray[np.float32]
    r_z_zhat_t_accumulated_modes: npt.NDArray[np.float32]
    p_z_zhat_t_accumulated_modes: npt.NDArray[np.float32]
    r_z_zhat_t_separated_modes: npt.NDArray[np.float32]
    p_z_zhat_t_separated_modes: npt.NDArray[np.float32]
    r_z_zhat_s_accumulated_modes: npt.NDArray[np.float32]
    p_z_zhat_s_accumulated_modes: npt.NDArray[np.float32]
    r_z_zhat_s_separated_modes: npt.NDArray[np.float32]
    p_z_zhat_s_separated_modes: npt.NDArray[np.float32]
    suy: npt.NDArray[np.float32]
    suz: npt.NDArray[np.float32]
    suy_sig: npt.NDArray[np.float32]
    suz_sig: npt.NDArray[np.float32]
    alpha: float

    @property
    def var_names(self) -> Tuple[str, ...]:
        """Returns the variables contained in the object (scf, us, vs, r_uv, ...)"""
        return (
            'scf',
            'r_uv',
            'r_uv_sig',
            'p_uv',
            'us',
            'vs',

            'zhat_accumulated_modes',
            'zhat_separated_modes',

            'r_z_zhat_t_accumulated_modes',
            'p_z_zhat_t_accumulated_modes',
            'r_z_zhat_t_separated_modes',
            'p_z_zhat_t_separated_modes',

            'r_z_zhat_s_accumulated_modes',
            'p_z_zhat_s_accumulated_modes',
            'r_z_zhat_s_separated_modes',
            'p_z_zhat_s_separated_modes',

            'suy',
            'suz',
            'suy_sig',
            'suz_sig',

            'psi_accumulated_modes',
            'psi_separated_modes',
            'alpha',
        )

    def __init__(
        self,
        dsy: Preprocess,
        dsz: Preprocess,
        nm: int,
        alpha: float,
        multiprocessed: bool = False,
        sig: str = 'test-t',
        montecarlo_iterations: Optional[int] = None,
    ):
        self._dsy = dsy
        self._dsz = dsz

        nz, ntz = self._dsz.land_data.shape
        ny, nty = self._dsy.land_data.shape

        _debuginfo(f"""Applying Crossvalidation 
    Shapes: Z{dsz.shape} 
            Y{dsy.shape} 
    Regions: Z {region2str(self._dsz.region)} 
            Y {region2str(self._dsy.region)}""", )
        time_from_here()

        if len(dsz.time) != len(dsy.time):
            raise ValueError(
                f'The number of years of the predictand must be the '
                f'same as the number of years of the predictor: '
                f'got {len(dsz.time)} and '
                f'{len(dsy.time)}'
            )

        nt = ntz

        self.scf = np.zeros([nm, nt], dtype=np.float32)
        self.r_uv = np.zeros([nm, nt], dtype=np.float32)
        self.r_uv_sig = np.zeros([nm, nt], dtype=np.float32)
        self.p_uv = np.zeros([nm, nt], dtype=np.float32)

        self.zhat_separated_modes = np.zeros([nm, nz, nt], dtype=np.float32)
        self.zhat_accumulated_modes = np.zeros([nm, nz, nt], dtype=np.float32)
        self.zhat_separated_modes[:, self._dsz.land_data.land_mask, :] = np.nan
        self.zhat_accumulated_modes[:, self._dsz.land_data.land_mask, :] = np.nan

        self.psi_separated_modes = np.zeros([nm, nt, ny, nz], dtype=np.float32)
        self.psi_accumulated_modes = np.zeros([nm, nt, ny, nz], dtype=np.float32)
        self.psi_separated_modes[:, :, self._dsy.land_data.land_mask, :][:, :, :, self._dsz.land_data.land_mask] = np.nan
        self.psi_accumulated_modes[:, :, self._dsy.land_data.land_mask, :][:, :, :, self._dsz.land_data.land_mask] = np.nan

        self.suy = np.zeros([ny, nt, nm], dtype=np.float32)
        self.suy_sig = np.zeros([ny, nt, nm], dtype=np.float32)
        self.suz = np.zeros([nz, nt, nm], dtype=np.float32)
        self.suz_sig = np.zeros([nz, nt, nm], dtype=np.float32)

        self.suy[self._dsy.land_data.land_mask, :, :] = np.nan
        self.suy_sig[self._dsy.land_data.land_mask, :, :] = np.nan
        self.suz[self._dsz.land_data.land_mask, :, :] = np.nan
        self.suz_sig[self._dsz.land_data.land_mask, :, :] = np.nan

        # crosvalidated year on axis 2
        self.us = np.zeros([nm, nt, nt], dtype=np.float32)
        self.vs = np.zeros([nm, nt, nt], dtype=np.float32)
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
                        'year': i, 'z': self._dsz.land_data, 'y': self._dsy.land_data, 'nt': nt, 'yrs': yrs,
                        'nm': nm, 'alpha': alpha, 'sig': sig, 'montecarlo_iterations': montecarlo_iterations,
                    })
                    processes.append(p)

                for i in yrs:
                    values = processes[i].get()
                    self.scf[:, i], self.r_uv[:, i], self.r_uv_sig[:, i], self.p_uv[:, i], \
                        self.us[:, [x for x in range(nt) if x != i], i], \
                        self.vs[:, [x for x in range(nt) if x != i], i], \
                        self.psi_separated_modes[:, i, :, :], self.psi_accumulated_modes[:, i, :, :], \
                        self.zhat_separated_modes[:, :, i], self.zhat_accumulated_modes[:, :, i], \
                        self.suy[:, i, :], self.suy_sig[:, i, :], self.suz[:, i, :], self.suz_sig[:, i, :] = values
        else:
            for i in yrs:
                out = self._crossvalidate_year(
                    year=i, z=self._dsz.land_data, y=self._dsy.land_data, nt=nt, yrs=yrs,
                    nm=nm, alpha=alpha, sig=sig, montecarlo_iterations=montecarlo_iterations
                )
                self.scf[:, i], self.r_uv[:, i], self.r_uv_sig[:, i], self.p_uv[:, i], \
                    self.us[:, [x for x in range(nt) if x != i], i], \
                    self.vs[:, [x for x in range(nt) if x != i], i], \
                    self.psi_separated_modes[:, i, :, :], self.psi_accumulated_modes[:, i, :, :], \
                    self.zhat_separated_modes[:, :, i], self.zhat_accumulated_modes[:, :, i], \
                    self.suy[:, i, :], self.suy_sig[:, i, :], self.suz[:, i, :], self.suz_sig[:, i, :] = out

        self.r_z_zhat_t_accumulated_modes, self.p_z_zhat_t_accumulated_modes, \
            self.r_z_zhat_t_separated_modes, self.p_z_zhat_t_separated_modes \
            = calculate_time_correlation(self._dsz.land_data, self.zhat_accumulated_modes, self.zhat_accumulated_modes)

        self.r_z_zhat_s_accumulated_modes, self.p_z_zhat_s_accumulated_modes, \
            self.r_z_zhat_s_separated_modes, self.p_z_zhat_s_separated_modes \
            = calculate_space_correlation(self._dsz.land_data, self.zhat_accumulated_modes, self.zhat_separated_modes)

        self.alpha = alpha
        debugprint(f'\n\tTook: {time_to_here():.03f} seconds')

    def _crossvalidate_year(
        self,
        year: int,
        z: LandArray,
        y: LandArray,
        nt: int,
        yrs: npt.NDArray[np.int32],
        nm: int,
        alpha: float,
        sig: str,
        montecarlo_iterations: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float32], ...]:
        """Function of internal use that processes a single year for crossvalidation"""
        debugprint('\tyear:', year, 'of', nt)
        z2 = LandArray(z.values[:, yrs != year])
        y2 = LandArray(y.values[:, yrs != year])
        mca_out = MCA.from_land_arrays(y2, z2, nm, alpha)
        ny, _ = y2.shape
        nz, _ = z2.shape

        psi_separated_modes = np.zeros([nm, ny, nz], dtype=np.float32)
        psi_accumulated_modes = np.zeros([nm, ny, nz], dtype=np.float32)
        zhat_separated_modes = np.zeros([nm, nz], dtype=np.float32)
        zhat_accumulated_modes = np.zeros([nm, nz], dtype=np.float32)

        psi_separated_modes[:, y2.land_mask, :] = np.nan
        psi_separated_modes[:, :, z2.land_mask] = np.nan
        psi_accumulated_modes[:, y2.land_mask, :] = np.nan
        psi_accumulated_modes[:, :, z2.land_mask] = np.nan

        zhat_separated_modes[:, z2.land_mask] = np.nan
        zhat_accumulated_modes[:, z2.land_mask] = np.nan

        for mode in range(nm):
            psi_separated_modes[mode, ~np.isnan(psi_separated_modes[mode])] = calculate_psi(mca_out.SUY[~y2.land_mask, mode:mode + 1], mca_out.Us[mode:mode + 1, :], z2.not_land_values, nt, 1, ny).reshape((~y2.land_mask).sum() * (~z2.land_mask).sum())
            psi_accumulated_modes[mode, ~np.isnan(psi_separated_modes[mode])] = calculate_psi(mca_out.SUY[~y2.land_mask, :mode + 1], mca_out.Us[:mode + 1, :], z2.not_land_values, nt, 1, ny).reshape((~y2.land_mask).sum() * (~z2.land_mask).sum())

            zhat_separated_modes[mode, ~z2.land_mask] = np.dot(np.transpose(y.not_land_values[:, year]), psi_separated_modes[mode, ~y2.land_mask, :][:, ~z2.land_mask])
            zhat_accumulated_modes[mode, ~z2.land_mask] = np.dot(np.transpose(y.not_land_values[:, year]), psi_accumulated_modes[mode, ~y2.land_mask, :][:, ~z2.land_mask])

        r_uv = np.zeros(nm, dtype=np.float32)
        r_uv_sig = np.zeros(nm, dtype=np.float32)
        p_uv = np.zeros(nm, dtype=np.float32)
        for m in range(nm):
            r_uv[m], p_uv[m], r_uv_sig[m], _, _ = index_regression(LandArray(mca_out.Us[m:m+1, :]), mca_out.Vs[m:m+1, :].T, alpha, sig, montecarlo_iterations)

        scf = mca_out.scf
        return (
            scf, r_uv, r_uv_sig, p_uv, mca_out.Us, mca_out.Vs,
            psi_separated_modes, psi_accumulated_modes,
            zhat_separated_modes, zhat_accumulated_modes,
            mca_out.SUY, mca_out.SUY_sig,
            mca_out.SUZ, mca_out.SUZ_sig
        )

    def plot(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        dir: Optional[str] = None,
        name: Optional[str] = None,
        cmap: Optional[str] = None,
        map_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        version: Literal["default", "elena"] = "default",
        mca: Optional[MCA] = None
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot the Crossvalidation results

        Parameters
        ----------
        save_fig
            Saves the fig in with `dir` / `name` parameters
        show_plot
            Shows the plot
        halt_program
            Only used if `show_plot` is `True`. If `True` shows the plot if plt.show
            and stops execution. Else uses fig.show and does not halt program
        dir
            Directory to save fig if `save_fig` is `True`
        name
            Name of the fig saved if `save_fig` is `True`
        cmap
            Colormap for the predicting maps
        map_ticks
            Ticks for the z map in version default
        version
            Select version from: `default` and `elena`
        mca
            MCA results for version `elena`

        Returns
        -------
        plt.Figure
            Figure object from matplotlib

        Sequence[plt.Axes]
            Tuple of axes in figure
        """
        fig: plt.Figure
        axs: Sequence[plt.Axes]
        if version == "default":
            if mca is not None:
                raise TypeError("Unexpected argument `mca` for version `default`")
            if cmap is None:
                cmap = 'bwr'
            fig, axs = _plot_crossvalidation_default(self, cmap, map_ticks)
        elif version == "elena":
            if mca is None:
                raise TypeError("Expected argument `mca` for version `elena`")
            if map_ticks is not None:
                raise TypeError("Unexpected argument `map_ticks` for version `elena`")
            if cmap is not None:
                raise TypeError("Unexpected argument `cmap` for version `elena`")
            fig, axs = _plot_crossvalidation_elena(self, mca)
        else:
            raise ValueError(f"Version can only be one of: `elena`, `default`")

        if dir is None:
            dir = '.'
        if name is None:
            path = os.path.join(dir, f'crossvalidation-plot_z-{self._dsz.var}_y-{self._dsy.var}.png')
        else:
            path = os.path.join(dir, name)

        _apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program
        )

        return fig, axs

    def plot_zhat(
        self,
        year: int,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        dir: Optional[str] = None,
        name: Optional[str] = None,
        cmap: str = 'bwr',
        yticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        zticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
        """Plots the map of Zhat

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
            Year to plot
        dir
            Directory to save fig if `save_fig` is `True`
        name
            Name of the fig saved if `save_fig` is `True`
        cmap
            Colormap for the predicting map
        yticks
            Ticks for the y map
        zticks
            Ticks for the z map

        Returns
        -------
        plt.Figure
            Figure object from matplotlib

        Sequence[plt.Axes]
            Tuple of axes in figure
        """
        nts, nylat, nylon = len(self._dsy.time), len(self._dsy.lat), len(self._dsy.lon)
        nts, nzlat, nzlon = len(self._dsz.time), len(self._dsz.lat), len(self._dsz.lon)

        height = nylat + nzlat + nzlat
        width = max(nzlon, nylon)

        fig = plt.figure(figsize=_calculate_figsize(height / width, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
        ax0 = plt.subplot(311, projection=ccrs.PlateCarree())
        ax1 = plt.subplot(312, projection=ccrs.PlateCarree())
        ax2 = plt.subplot(313, projection=ccrs.PlateCarree())

        zindex = _get_index_from_sy(self._dsz.time, year)
        yindex = zindex
        y_year = self._dsy.time.values[yindex]

        d0 = self._dsy.data.transpose().reshape((nts, nylat, nylon))

        _plot_map(d0[yindex], self._dsy.lat, self._dsy.lon, fig, ax0, f'Y on year {y_year}', ticks=yticks)

        d1 = self.zhat_accumulated_modes[-1, :].transpose().reshape((nts, nzlat, nzlon))
        d2 = self._dsz.data.transpose().reshape((nts, nzlat, nzlon))

        n = 30
        _std = np.nanstd(d2[zindex])
        _m = np.nanmean(d2[zindex])
        bound = max(abs(_m - _std), abs(_m + _std))
        levels = np.linspace(-bound, bound, n)

        _plot_map(
            d1[zindex], self._dsz.lat, self._dsz.lon, fig, ax1, f'Zhat on year {year}',
            cmap=cmap, levels=levels, ticks=zticks
        )
        _plot_map(
            d2[zindex], self._dsz.lat, self._dsz.lon, fig, ax2, f'Z on year {year}',
            cmap=cmap, levels=levels, ticks=zticks
        )

        fig.suptitle(
            f'Z({self._dsz.var}): {region2str(self._dsz.region)}, '
            f'Y({self._dsy.var}): {region2str(self._dsy.region)}. '
            f'Alpha: {self.alpha}',
            fontweight='bold'
        )

        fig.subplots_adjust(hspace=.4)

        if dir is None:
            dir = '.'
        if name is None:
            path = os.path.join(dir, f'crossvalidation-plot_z-{self._dsz.var}_y-{self._dsy.var}.png')
        else:
            path = os.path.join(dir, name)

        _apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program
        )

        return fig, (ax0, ax1, ax2)

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

        self: Crossvalidation = super().load(prefix, dir)
        self._dsz = dsz
        self._dsy = dsy
        return self


def calculate_psi(
    suy: npt.NDArray[np.float32],
    us: npt.NDArray[np.float32],
    z: npt.NDArray[np.float32],
    nt: int,
    nm: int,
    ny: int,
) -> npt.NDArray[np.float32]:
    # (((SUY * inv(Us * Us')) * Us) * Z') * nt * nm / ny
    return cast(
        npt.NDArray[np.float32],
        np.dot(np.dot(np.dot(suy, np.linalg.inv(np.dot(us, np.transpose(us)))), us), np.transpose(z)) * nt * nm / ny)


def _plot_crossvalidation_elena(
    cross: Crossvalidation,
    mca: MCA,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    fig = plt.figure(figsize=(24, 8.5))
    # fig = plt.figure(figsize=(28, 24))

    nm = cross.us.shape[0]
    nt = cross.us.shape[1]
    nlat = cross._dsz.lat.shape[0]
    nlon = cross._dsz.lon.shape[0]

    spec = gridspec.GridSpec(
        ncols=3, nrows=4, figure=fig, wspace=0.4, hspace=0.6, width_ratios=[1, 1, 1],
        height_ratios=[1.3, 1.3, 2, 2]
    )

    axs: List[plt.Axes] = []

    # fig.suptitle('Crossvalidation',fontsize=20,weight='bold')
    for n_mode in range(nm):
        us_2 = cross.us[n_mode, :, :].copy()
        vs_2 = cross.vs[n_mode, :, :].copy()
        for year in range(nt):
            u_r, _u_pvalue = stats.pearsonr(mca.Us[n_mode, :], cross.us[n_mode, :, year])
            if u_r < 0:
                us_2[:, year] *= -1
                # cross.us[n_mode, :, year] *= -1
                # cross.suy[:, year, n_mode] *= -1
                # cross.suy_sig[:, year, n_mode] *= -1
                # cross.suz[:, year, n_mode] *= -1
                # cross.suz_sig[:, year, n_mode] *= -1
            v_r, _v_pvalue = stats.pearsonr(mca.Vs[n_mode, :], cross.vs[n_mode, :, year])
            if v_r < 0:
                # cross.vs[n_mode, :, year] *= -1
                vs_2[:, year] *= -1
            # us_2[:, year] = cross.us[n_mode, :, year]
            # vs_2[:, year] = cross.vs[n_mode, :, year]

        error_u = np.nanstd(us_2[:, :], 1)
        error_v = np.nanstd(vs_2[:, :], 1)

        # Coeficientes expansión
        ax0 = fig.add_subplot(spec[0, n_mode])
        axs.append(ax0)

        # ax0.errorbar(cross._dsz.time, mca.Us[n_mode, :], yerr=error_u, capsize=3, linewidth=2, color='blue', ecolor='k', label='Us')
        # ax0.errorbar(cross._dsz.time, mca.Vs[n_mode, :], yerr=error_v, capsize=3, linewidth=2, color='green', ecolor='k', label='Vs')
        ax0.errorbar(cross._dsz.time, mca.Us[n_mode, :], yerr=error_u, color='blue', ecolor='k', label='Us')
        ax0.errorbar(cross._dsz.time, mca.Vs[n_mode, :], yerr=error_v, color='green', ecolor='k', label='Vs')
        # ax0.plot(cross._dsz.time,index_an/np.std(index_an),linewidth=1.5,linestyle='--',color='green',label='index')
        ax0.grid()
        ax0.set_xlabel('Years', weight='bold', fontsize=8)
        # ax0.set_xlabel('Years', fontsize=20, weight='bold')

        if n_mode == 0:
            ax0.legend(ncol=3)
            # ax0.legend(fontsize=20, ncol=3)
            ax0.set_ylabel('nº std', weight='bold', fontsize=8)
            # ax0.set_ylabel('nº std', fontsize=20, weight='bold')

        r_uv, p_value, _ruv_sig, _reg, _reg_sig = index_regression(
            LandArray(mca.Us[n_mode:n_mode+1, :]), mca.Vs[n_mode:n_mode+1, :].T, cross.alpha, 'test-t', 1000, )

        # plt.xticks()
        # plt.yticks()
        ax0.set_ylim(-5.5, 5.5)
        ax0.set_title(f'Mode {n_mode + 1}, |ruv|={abs(r_uv[0]):.2f} p={1 - p_value[0]:.3f}', fontweight='bold', fontsize=8)
        # ax0.set_title(
        #     f'Mode {n_mode + 1}, |ruv|={abs(r_uv[0][0]):.2f} p={1 - p_value[0]:.3f}', fontsize=20,
        #     weight='bold', y=1.02)

        # Evolución scf y ruv
        ax1 = fig.add_subplot(spec[1, n_mode])
        axs.append(ax1)
        ax1.plot(cross._dsz.time, abs(cross.r_uv[n_mode, :]), color='cornflowerblue', label='ruv')
        ax1.scatter(cross._dsz.time, abs(cross.r_uv_sig[n_mode, :]), color='cornflowerblue')
        ax1.plot(cross._dsz.time, [abs(r_uv[0])] * len(cross._dsz.time), color='cornflowerblue')
        ax1.scatter([], [], color='purple', marker='o', label='scf')
        # ax1.scatter([], [], linewidth=2, color='purple', marker='o', label='scf')
        ax1.grid()
        ax1.set_ylim(0, round(np.max(cross.r_uv[n_mode]), 1) + 0.1)
        ax1.set_xlabel('Years', weight='bold', fontsize=8)
        # ax1.set_xlabel('Years', fontsize=20, weight='bold')
        # if i == 2:
        #     ax0.legend(fontsize=20)
        ax01 = ax1.twinx()
        ax01.scatter(cross._dsz.time, cross.scf[n_mode, :] * 100, color='purple', marker='o', label='scf')
        # ax01.scatter(cross._dsz.time, cross.scf[n_mode, :] * 100, linewidth=2, color='purple', marker='o', label='scf')
        ax01.plot(cross._dsz.time, [cross.scf[n_mode, :] * 100] * len(cross._dsz.time), color='purple')
        ax01.set_ylim([np.min(cross.scf[n_mode]) * 100 + 2, np.max(cross.scf[n_mode, :]) * 100 + 2])

        if n_mode == 0:
            ax1.set_ylabel('|ruv|', weight='bold', fontsize=8)
            # ax1.set_ylabel('|ruv|', fontsize=20, weight='bold')
            ax1.legend(loc='lower right')
            # ax1.legend(loc='lower right', fontsize=20)

        if n_mode == 2:
            ax01.set_ylabel('%', weight='bold', fontsize=8)
            # ax01.set_ylabel('%', fontsize=20, weight='bold')

        # ax01.set_ylabel('scf',fontsize=20,weight='bold')
        ax1.set_title(
            f'Mode {n_mode + 1}, |ruv|={abs(r_uv[0]):.2f} , scf={mca.scf[n_mode] * 100:.2f}%',
            weight='bold', fontsize=8)
        # ax1.set_title(
        #     f'Mode {n_mode + 1}, |ruv|={abs(r_uv[0]):.2f} , scf={mca.scf[n_mode] * 100:.2f}%',
        #     fontsize=20, weight='bold', y=1.02)

        # Skill
        levels = np.arange(-1, 1.1, 0.1)

        # o, s = ('horizontal', 0.6) if region == 'Ecuador' or region == 'Ecuador2' else ('vertical', 1)

        # lon_grid, lat_grid = np.meshgrid(cross._dsz.lon, cross._dsz.lat)
        # is_on_land = globe.is_land(lat_grid, lon_grid)

        sk = cross.r_z_zhat_s_accumulated_modes[n_mode, :].reshape(nlat, nlon)
        r_sig = cross.r_z_zhat_s_accumulated_modes[n_mode, :].copy()
        r_sig[cross.p_z_zhat_s_accumulated_modes[n_mode] > cross.alpha] = np.nan
        sk_sig = r_sig.reshape(nlat, nlon)

        ax2 = fig.add_subplot(spec[2, n_mode], projection=ccrs.PlateCarree())
        axs.append(ax2)
        im = ax2.contourf(cross._dsz.lon, cross._dsz.lat, sk, levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree())
        ax2.contourf(cross._dsz.lon, cross._dsz.lat, sk_sig, levels=levels, cmap='coolwarm', hatches='.', transform=ccrs.PlateCarree())
        ax2.coastlines()
        if n_mode == 3:
            _cbar = fig.colorbar(im, ax2)
        gl = ax2.gridlines(draw_labels=True)
        gl.ylabels_right = False
        gl.xlabels_top = False
        ax2.set_extent([min(cross._dsz.lon), max(cross._dsz.lon), min(cross._dsz.lat), max(cross._dsz.lat)])
        ax2.set_title(f'ACC {n_mode + 1} modes', weight='bold', fontsize=8)
        # ax2.set_title(f'ACC {n_mode + 1} modes', fontsize=20, weight='bold', y=1.02)

        if n_mode == 0:
            ax3 = fig.add_subplot(spec[3, n_mode])
            width = 0.3
            # Evolución r y rmse
            r_sig = cross.r_z_zhat_t_accumulated_modes.copy()
            r_sig[cross.p_z_zhat_t_accumulated_modes > cross.alpha] = np.nan
            ax3.bar(cross._dsz.time - 0.3, cross.r_z_zhat_t_accumulated_modes[0, :], width, color='blue', label='1')
            ax3.bar(cross._dsz.time - 0.3, r_sig[0, :], width, color='blue', hatch='-')
            ax3.bar(cross._dsz.time, cross.r_z_zhat_t_accumulated_modes[1, :], width, color='green', label='2')
            ax3.bar(cross._dsz.time, r_sig[1, :], width, color='green', hatch='-')
            ax3.bar(cross._dsz.time + 0.3, cross.r_z_zhat_t_accumulated_modes[2, :], width, color='orange', label='3')
            ax3.bar(cross._dsz.time + 0.3, r_sig[2, :], width, color='orange', hatch='-')
            ax3.set_ylim(-0.65, 0.65)
            # ax3.set_ylabel('r',fontsize=20,weight='bold')
            ax04 = ax3.twinx()
            # ax02.plot(cross._dsz.time,rmse_r[2,:]*100/abs(index),color='orange')
            # ax02.plot(cross._dsz.time,rmse_r[1,:]*100/abs(index),color='green')
            # ax02.plot(cross._dsz.time,rmse_r[0,:]*100/abs(index),color='blue')
            mse = np.mean((cross.zhat_accumulated_modes - cross._dsz.data)**2, axis=1)
            mse_clim = np.std(cross._dsz.data, axis=0)
            msess = 1 - mse / mse_clim
            ax04.plot(cross._dsz.time, msess[2, :], color='orange')
            ax04.plot(cross._dsz.time, msess[1, :], color='green')
            ax04.plot(cross._dsz.time, msess[0, :], color='blue')
            # ax02.set_ylabel('%',fontsize=20,weight='bold')
            ax3.legend(loc='upper center', bbox_to_anchor=(0.7, -.25), ncol=3, columnspacing=1)
            # ax3.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.7, 1.2), ncol=3, columnspacing=1)
            ax3.grid()
            ax3.set_title('Skill (bars), \n MSESS (lines)', weight='bold', fontsize=8)
            # ax3.set_title('Skill (bars), \n MSESS (lines)', fontsize=20, weight='bold', x=0.2, y=1.02)
        else:
            sk_i = cross.r_z_zhat_s_accumulated_modes[n_mode - 1, :].reshape(nlat, nlon)
            # Skill modos
            ax3 = fig.add_subplot(spec[3, n_mode], projection=ccrs.PlateCarree())
            im = ax3.contourf(cross._dsz.lon, cross._dsz.lat, sk - sk_i, levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree())
            # ax2.contourf(cross._dsz.lon,cross._dsz.lat,skill_sig[i,:,:],levels=levels,cmap='coolwarm',hatches='.',transform = ccrs.PlateCarree())
            ax3.coastlines()
            if n_mode == 3:
                _cbar = fig.colorbar(im, ax3)
            gl = ax3.gridlines(draw_labels=True)
            gl.ylabels_right = False
            gl.xlabels_top = False
            ax3.set_extent([min(cross._dsz.lon), max(cross._dsz.lon), min(cross._dsz.lat), max(cross._dsz.lat)])
            ax3.set_title(f'ACC {np.arange(n_mode + 1) + 1}) - {np.arange(n_mode) + 1}', weight='bold', fontsize=8)
            # ax3.set_title(f'ACC {np.arange(n_mode + 1) + 1}) - {np.arange(n_mode) + 1}', fontsize=20, weight='bold', y=1.02)
        axs.append(ax3)
    return fig, tuple(axs)


def _plot_crossvalidation_default(
    cross: Crossvalidation,
    cmap: str,
    map_ticks: Optional[
        Union[npt.NDArray[np.float32], Sequence[float]]
    ]
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
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
    axs = tuple(
        plt.subplot(nrows * 100 + ncols * 10 + i,
                    projection=(ccrs.PlateCarree() if i == 1 else None))
        for i in range(1, ncols * nrows + 1)
    )

    nzlat = len(cross._dsz.lat)
    nzlon = len(cross._dsz.lon)
    # nztime = len(ts)

    # ------ r_z_zhat_s and p_z_zhat_s ------ #
    # Correlation map
    d = cross.r_z_zhat_s_accumulated_modes[-1, :].transpose().reshape((nzlat, nzlon))
    _mean = np.nanmean(d)
    _std = np.nanstd(d)
    mx = _mean + _std
    mn = _mean - _std
    _plot_map(
        d, cross._dsz.lat, cross._dsz.lon, fig, axs[0],
        'Correlation in space between z and zhat',
        cmap=cmap,
        ticks=(np.arange(round(mn * 10) / 10, floor(mx * 10) / 10 + .05, .1) if map_ticks is None and not np.isnan(_mean) and not np.isnan(_std) else map_ticks)
    )
    hatches = d.copy()
    hatches[((cross.p_z_zhat_s_accumulated_modes[-1, :] > cross.alpha) | (
                cross.r_z_zhat_s_accumulated_modes[-1, :] < 0)).transpose().reshape((nzlat, nzlon))] = np.nan

    axs[0].contourf(
        cross._dsz.lon, cross._dsz.lat, hatches,
        colors='none', hatches='..', extend='both',
        transform=ccrs.PlateCarree()
    )
    # ^^^^^^ r_z_zhat_s and p_z_zhat_s ^^^^^^ #

    # ------ r_z_zhat_t and p_z_zhat_t ------ #
    axs[1].bar(cross._dsz.time.values, cross.r_z_zhat_t_accumulated_modes[-1, :])
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1].scatter(
        cross._dsz.time[cross.p_z_zhat_t_accumulated_modes[-1, :] <= cross.alpha],
        cross.p_z_zhat_t_accumulated_modes[-1, :][cross.p_z_zhat_t_accumulated_modes[-1, :] <= cross.alpha]
    )
    axs[1].set_title('Correlation in space between z and zhat')
    axs[1].grid(True)
    # ^^^^^^ r_z_zhat_t and p_z_zhat_t ^^^^^^ #

    # ------ scf ------ #
    for mode in range(cross.scf.shape[0]):
        _plot_ts(
            time=cross._dsy.time.values,
            arr=cross.scf[mode],
            ax=axs[2],
            label=f'Mode {mode + 1}',
            title='Squared convariance fraction'
        )
    axs[2].legend()
    axs[2].grid(True)
    # ^^^^^^ scf ^^^^^^ #

    # ^^^^^^ Us ^^^^^^ #
    mean = cross.us.mean(2)
    std = np.std(cross.us, axis=2)
    for mode in range(mean.shape[0]):
        axs[3 + mode].grid(True)
        axs[3 + mode].errorbar(
            cross._dsy.time, mean[mode], yerr=np.abs(std[mode]), label='std', color='orange', linewidth=3, ecolor='purple'
        )
        axs[3 + mode].set_title(
            f'Us for mode {mode + 1}'
        )
        axs[3 + mode].legend()

    fig.suptitle(
        f'Z({cross._dsz.var}): {region2str(cross._dsz.region)}, '
        f'Y({cross._dsy.var}): {region2str(cross._dsy.region)}. '
        f'Alpha: {cross.alpha}',
        fontweight='bold'
    )

    fig.subplots_adjust(hspace=.4)

    return fig, axs


def calculate_time_correlation(
    z_land_array: LandArray,
    zhat_accumulated_modes: npt.NDArray[np.float_],
    zhat_separated_modes: Optional[npt.NDArray[np.float_]] = None,
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    nm, nz, nt = zhat_accumulated_modes.shape

    r_z_zhat_t_accumulated_modes = np.zeros([nm, nt], dtype=np.float32)
    p_z_zhat_t_accumulated_modes = np.zeros([nm, nt], dtype=np.float32)
    r_z_zhat_t_separated_modes = np.zeros([nm, nt], dtype=np.float32)
    p_z_zhat_t_separated_modes = np.zeros([nm, nt], dtype=np.float32)

    for j in range(nt):
        for mode in range(nm):
            if zhat_separated_modes is not None:
                rtt_sep = stats.pearsonr(zhat_separated_modes[mode, ~z_land_array.land_mask, j], z_land_array.not_land_values[:, j])  # serie de skill
                r_z_zhat_t_separated_modes[mode, j] = rtt_sep[0]
                p_z_zhat_t_separated_modes[mode, j] = rtt_sep[1]

            rtt_acc = stats.pearsonr(zhat_accumulated_modes[mode, ~z_land_array.land_mask, j],
                                     z_land_array.not_land_values[:, j])  # serie de skill
            r_z_zhat_t_accumulated_modes[mode, j] = rtt_acc[0]
            p_z_zhat_t_accumulated_modes[mode, j] = rtt_acc[1]

    return r_z_zhat_t_accumulated_modes, p_z_zhat_t_accumulated_modes, r_z_zhat_t_separated_modes, p_z_zhat_t_separated_modes


def calculate_space_correlation(
    z_land_array: LandArray,
    zhat_accumulated_modes: npt.NDArray[np.float_],
    zhat_separated_modes: Optional[npt.NDArray[np.float_]] = None,
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    nm, nz, nt = zhat_accumulated_modes.shape

    r_z_zhat_s_accumulated_modes = np.zeros([nm, nz], dtype=np.float32)
    p_z_zhat_s_accumulated_modes = np.zeros([nm, nz], dtype=np.float32)
    r_z_zhat_s_separated_modes = np.zeros([nm, nz], dtype=np.float32)
    p_z_zhat_s_separated_modes = np.zeros([nm, nz], dtype=np.float32)

    r_z_zhat_s_accumulated_modes[:, z_land_array.land_mask] = np.nan
    p_z_zhat_s_accumulated_modes[:, z_land_array.land_mask] = np.nan
    r_z_zhat_s_separated_modes[:, z_land_array.land_mask] = np.nan
    p_z_zhat_s_separated_modes[:, z_land_array.land_mask] = np.nan

    for i in np.nonzero(~z_land_array.land_mask)[0]:
        for mode in range(nm):
            if zhat_separated_modes is not None:
                rtt_sep = stats.pearsonr(zhat_separated_modes[mode, i, :], z_land_array.values[i, :])  # serie de skill
                r_z_zhat_s_separated_modes[mode, i] = rtt_sep[0]
                p_z_zhat_s_separated_modes[mode, i] = rtt_sep[1]

            rtt_acc = stats.pearsonr(zhat_accumulated_modes[mode, i, :], z_land_array.values[i, :])  # serie de skill
            r_z_zhat_s_accumulated_modes[mode, i] = rtt_acc[0]
            p_z_zhat_s_accumulated_modes[mode, i] = rtt_acc[1]

    return r_z_zhat_s_accumulated_modes, p_z_zhat_s_accumulated_modes, r_z_zhat_s_separated_modes, p_z_zhat_s_separated_modes
