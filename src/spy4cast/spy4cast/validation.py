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
from .crossvalidation import calculate_psi, calculate_time_correlation, calculate_space_correlation
from .mca import index_regression
from .. import Slise
from .._functions import debugprint, slise2str, time_from_here, time_to_here, _debuginfo
from .._procedure import _Procedure, _apply_flags_to_fig, _plot_map, _get_index_from_sy, _calculate_figsize, MAX_WIDTH, \
    MAX_HEIGHT, _plot_ts
from ..land_array import LandArray
from .preprocess import Preprocess
import xarray as xr


__all__ = [
    'Validation',
]


class Validation(_Procedure):
    """Perform validation methodology

    Parameters
    ----------

    Attributes
    ----------

    See Also
    --------
        MCA
        Crossvalidation
    """

    @property
    def var_names(self) -> Tuple[str, ...]:
        """Returns the variables contained in the object"""
        return (
            'psi',
            'zhat',
            'r_z_zhat_t_accumulated_modes',
            'p_z_zhat_t_accumulated_modes',
            'r_z_zhat_s_accumulated_modes',
            'p_z_zhat_s_accumulated_modes',
        )

    def __init__(
        self,
        training_mca: MCA,
        validating_dsy: Preprocess,
        validating_dsz: Preprocess,
    ) -> None:
        self.training_mca = training_mca
        self.validating_dsy = validating_dsy
        self.validating_dsz = validating_dsz

        _debuginfo(f"""Applying Validation
        Training
        ---------- 
        Shapes: Z{training_mca.zdata.shape} 
                Y{training_mca.ydata.shape} 
        Slises: Z {slise2str(training_mca.zslise)} 
                Y {slise2str(training_mca.yslise)}
        
        Validation
        ---------- 
        Shapes: Z{validating_dsz.shape} 
                Y{validating_dsy.shape} 
        Slises: Z {slise2str(self.validating_dsz.slise)} 
                Y {slise2str(self.validating_dsy.slise)}""", )

        if len(validating_dsz.time) != len(validating_dsy.time):
            raise ValueError(
                f'The number of years of the predictand must be the '
                f'same as the number of years of the predictor: '
                f'got {len(validating_dsy.time)} and '
                f'{len(validating_dsz.time)}'
            )

        time_from_here()

        # --- NEED TO BE FASTER!!!
        common_z_land_indices = np.array(list(set(validating_dsz.land_data.land_indices) | set(training_mca.z_land_array.land_indices)), dtype=np.int32)
        common_y_land_indices = np.array(list(set(validating_dsy.land_data.land_indices) | set(training_mca.y_land_array.land_indices)), dtype=np.int32)
        common_z_not_land_indices = np.array([i for i in range(training_mca.zdata.shape[0]) if i not in set(common_z_land_indices)], dtype=np.int32)
        common_y_not_land_indices = np.array([i for i in range(training_mca.ydata.shape[0]) if i not in set(common_y_land_indices)], dtype=np.int32)
        # ---

        self.psi = np.zeros([1, training_mca.ydata.shape[0], training_mca.zdata.shape[0]], dtype=np.float32)
        self.zhat = np.zeros([1, validating_dsz.land_data.shape[0], validating_dsz.time.shape[0]], dtype=np.float32)

        self.psi[:, common_y_land_indices, :] = np.nan
        self.psi[:, :, common_z_land_indices] = np.nan
        self.zhat[:, common_z_land_indices] = np.nan

        self.psi[0, ~np.isnan(self.psi[0])] = calculate_psi(
            self.training_mca.SUY[common_y_not_land_indices, :],
            self.training_mca.Us[:, :],
            self.training_mca.zdata[common_z_not_land_indices, :],
            self.training_mca.ytime.shape[0],
            self.training_mca.Us.shape[0],
            self.training_mca.ydata.shape[0]
        ).reshape(len(common_y_not_land_indices) * len(common_z_not_land_indices))

        self.zhat[0, common_z_not_land_indices, :] = np.dot(
            validating_dsy.land_data.not_land_values[:, :].T,
            self.psi[0, common_y_not_land_indices, :][:, common_z_not_land_indices]).T

        self.r_z_zhat_t_accumulated_modes, self.p_z_zhat_t_accumulated_modes, \
            _r_z_zhat_t_separated_modes, _p_z_zhat_t_separated_modes \
            = calculate_time_correlation(self.z_land_array, self.zhat)

        self.r_z_zhat_s_accumulated_modes, self.p_z_zhat_s_accumulated_modes, \
            _r_z_zhat_s_separated_modes, _p_z_zhat_s_separated_modes \
            = calculate_space_correlation(self.z_land_array, self.zhat)

        debugprint(f'\n\tTook: {time_to_here():.03f} seconds')

    @property
    def alpha(self) -> float:
        """Land Array for Y dataset"""
        return self.training_mca.alpha

    @property
    def y_land_array(self) -> LandArray:
        """Land Array for Y dataset"""
        return self.validating_dsy.land_data

    @property
    def z_land_array(self) -> LandArray:
        """Land Array for Z dataset"""
        return self.validating_dsz.land_data

    @property
    def ydata(self) -> npt.NDArray[np.float32]:
        """Matrix with the data of the predictor variable

        Returns
        -------
            npt.NDArray[np.float32]
        """
        return self.validating_dsy.land_data.values

    @property
    def yvar(self) -> str:
        """Predictor variable name

        Returns
        -------
            str
        """
        return self.validating_dsy.var

    @property
    def yslise(self) -> Slise:
        """Predictor variable slise

        Returns
        -------
            spy4cast.stypes.Slise
        """
        return self.validating_dsy.slise

    @property
    def ytime(self) -> xr.DataArray:
        """Vector with time values of the predictor variable

        Returns
        -------
            xarray.DataArray
        """
        return self.validating_dsy.time

    @property
    def ylat(self) -> xr.DataArray:
        """Vector with latitude values of the predictor variable

        Returns
        -------
            xarray.DataArray
        """
        return self.validating_dsy.lat

    @property
    def ylon(self) -> xr.DataArray:
        """Vector with longitude values of the predictor variable

        Returns
        -------
            xarray.DataArray
        """
        return self.validating_dsy.lon

    @property
    def zdata(self) -> npt.NDArray[np.float32]:
        """Matrix with the data of the predicting variable

        Returns
        -------
            npt.NDArray[np.float32]
        """
        return self.validating_dsz.land_data.values

    @property
    def zvar(self) -> str:
        """Predicting variable name

        Returns
        -------
            str
        """
        return self.validating_dsz.var

    @property
    def zslise(self) -> Slise:
        """Predicting variable slise

        Returns
        -------
            spy4cast.stypes.Slise
        """
        return self.validating_dsz.slise

    @property
    def ztime(self) -> xr.DataArray:
        """Vector with time values of the predicting variable

        Returns
        -------
            xarray.DataArray
        """
        return self.validating_dsz.time

    @property
    def zlat(self) -> xr.DataArray:
        """Vector with latitude values of the predicting variable

        Returns
        -------
            xarray.DataArray
        """
        return self.validating_dsz.lat

    @property
    def zlon(self) -> xr.DataArray:
        """Vector with longitude values of the predicting variable

        Returns
        -------
            xarray.DataArray
        """
        return self.validating_dsz.lon

    @classmethod
    def load(cls, prefix: str, dir: str = '.', *,
             validating_dsy: Optional[Preprocess] = None,
             validating_dsz: Optional[Preprocess] = None,
             training_mca: Optional[MCA] = None,
             **attrs: Any) -> 'Validation':
        """Load an Validation object from matrices and type

        Parameters
        ----------
        prefix : str
            Prefix of the files containing the information for the object
        dir : str
            Directory of the files
        validating_dsy : Preprocess
            Preprocessed dataset of the validating predictor variable
        validating_dsz : Preprocess
            Preprocessed dataset of the validating predicting variable
        training_mca : MCA
            Training mca

        Returns
        -------
            Validation
        """
        if len(attrs) != 0:
            raise TypeError('Load only takes three keyword arguments: validating_dsy, validating_dsz and training_mca')
        if validating_dsy is None or validating_dsz is None or training_mca is None:
            raise TypeError('To load an MCA object you must provide `validating_dsz`, `validating_dsy` and `training_mca` keyword arguments')
        if type(validating_dsy) != Preprocess or type(validating_dsz) != Preprocess or type(training_mca) != MCA:
            raise TypeError(
                f'Unexpected types ({type(validating_dsy)}, {type(validating_dsz)} and {type(training_mca)}) for `validating_dsz`, `validating_dsy` and `training_mca`. Expected type `Preprocess`, `Preprocess` and `MCA`')

        self: Validation = super().load(prefix, dir)

        self.validating_dsy = validating_dsy
        self.validating_dsz = validating_dsz
        self.training_mca = training_mca
        return self

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
            fig, axs = _plot_validation_default(self, cmap, map_ticks)
        elif version == "elena":
            if mca is None:
                raise TypeError("Expected argument `mca` for version `elena`")
            if map_ticks is not None:
                raise TypeError("Unexpected argument `map_ticks` for version `elena`")
            if cmap is not None:
                raise TypeError("Unexpected argument `cmap` for version `elena`")
            fig, axs = _plot_validation_elena(self, mca)
        else:
            raise ValueError(f"Version can only be one of: `elena`, `default`")

        if dir is None:
            dir = '.'
        if name is None:
            path = os.path.join(dir, f'crossvalidation-plot_z-{self.zvar}_y-{self.yvar}.png')
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
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program
        )

        return fig, (ax0, ax1, ax2)


def _plot_validation_elena(
    validation: Validation,
    mca: MCA,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    raise NotImplementedError


def _plot_validation_default(
    validation: Validation,
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

    fig: plt.Figure = plt.figure(figsize=_calculate_figsize(None, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))
    nrows = 3
    ncols = 2
    axs: Tuple[plt.Axes] = tuple(
        fig.add_subplot(nrows, ncols, i, projection=(ccrs.PlateCarree() if i == 1 else None))
        for i in range(1, ncols * nrows + 1)
    )

    nzlat = len(validation.zlat)
    nzlon = len(validation.zlon)
    # nztime = len(ts)

    # ------ r_z_zhat_s and p_z_zhat_s ------ #
    # Correlation map
    d = validation.r_z_zhat_s_accumulated_modes[-1, :].transpose().reshape((nzlat, nzlon))
    _mean = np.nanmean(d)
    _std = np.nanstd(d)
    mx = _mean + _std
    mn = _mean - _std
    _plot_map(
        d, validation.zlat, validation.zlon, fig, axs[0],
        'Correlation in space between z and zhat',
        cmap=cmap,
        ticks=(np.arange(round(mn * 10) / 10, floor(mx * 10) / 10 + .05, .1) if map_ticks is None and not np.isnan(_mean) and not np.isnan(_std) else map_ticks)
    )
    hatches = d.copy()
    hatches[((validation.p_z_zhat_s_accumulated_modes[-1, :] > validation.alpha) | (
            validation.r_z_zhat_s_accumulated_modes[-1, :] < 0)).transpose().reshape((nzlat, nzlon))] = np.nan

    axs[0].contourf(
        validation.zlon, validation.zlat, hatches,
        colors='none', hatches='..', extend='both',
        transform=ccrs.PlateCarree()
    )
    # ^^^^^^ r_z_zhat_s and p_z_zhat_s ^^^^^^ #

    # ------ r_z_zhat_t and p_z_zhat_t ------ #
    axs[1].bar(validation.ztime.values, validation.r_z_zhat_t_accumulated_modes[-1, :])
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1].scatter(
        validation.ztime[validation.p_z_zhat_t_accumulated_modes[-1, :] <= validation.alpha],
        validation.p_z_zhat_t_accumulated_modes[-1, :][validation.p_z_zhat_t_accumulated_modes[-1, :] <= validation.alpha]
    )
    axs[1].set_title('Correlation in space between z and zhat')
    axs[1].grid(True)
    # ^^^^^^ r_z_zhat_t and p_z_zhat_t ^^^^^^ #

    # ------ scf ------ #
    axs[2].bar(np.arange(len(validation.training_mca.scf)), validation.training_mca.scf)
    axs[2].set_title("Square covariance fraction")
    axs[2].set_xlabel("Mode")
    axs[2].grid(True)
    # ^^^^^^ scf ^^^^^^ #

    # ^^^^^^ Us ^^^^^^ #
    for mode in range(validation.training_mca.Us.shape[0]):
        axs[3 + mode].grid(True)
        axs[3 + mode].plot(
            validation.training_mca.ytime, validation.training_mca.Us[mode], label=f'mode {mode}', color='orange', linewidth=3
        )
        axs[3 + mode].set_title(
            f'Us for mode {mode + 1}'
        )
        axs[3 + mode].legend()

    fig.suptitle(
        f'Z({validation.zvar}): {slise2str(validation.zslise)}, '
        f'Y({validation.yvar}): {slise2str(validation.yslise)}. '
        f'Alpha: {validation.alpha}',
        fontweight='bold'
    )

    fig.subplots_adjust(hspace=.4)

    return fig, axs
