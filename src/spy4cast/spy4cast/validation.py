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
from .. import Region
from .._functions import debugprint, region2str, time_from_here, time_to_here, _debuginfo
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
        training_mca : MCA
            MCA perform with the training datasets
        validating_dsy : Preprocess
            Predictor field for validation
        validating_dsz : Preprocess
            Predictand field for validation
    Attributes
    ----------
        psi : npt.NDArray[np.float32]
            Psi calculated with the training MCA data. Dimension: 1 x training_y_space x training_z_space
        zhat : npt.NDArray[np.float32]
            Zhat predicted for the predictand. Dimension: 1 x validating_z_space x validating_z_time
        r_z_zhat_t_accumulated_modes : npt.NDArray[np.float32]
            Correlation in time for accumlating all modes selected (nm) between z and zhat. Dimension: 1 x valudating_z_time
        p_z_zhat_t_accumulated_modes : npt.NDArray[np.float32]
            Pvalue of the correlation in time for accumlating all modes selected (nm) between z and zhat. Dimension: 1 x valudating_z_time
        r_z_zhat_s_accumulated_modes : npt.NDArray[np.float32]
            Correlation in space for accumlating all modes selected (nm) between z and zhat. Dimension: 1 x valudating_z_space
        p_z_zhat_s_accumulated_modes : npt.NDArray[np.float32]
            Pvalue of the correlation in space for accumlating all modes selected (nm) between z and zhat. Dimension: 1 x valudating_z_space

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
        self._training_mca = training_mca
        self._validating_dsy = validating_dsy
        self._validating_dsz = validating_dsz

        _debuginfo(f"""Applying Validation
        Training
        ---------- 
        Shapes: Z{training_mca._dsz.data.shape} 
                Y{training_mca._dsy.data.shape} 
        Regions: Z {region2str(training_mca._dsz.region)} 
                Y {region2str(training_mca._dsy.region)}
        
        Validation
        ---------- 
        Shapes: Z{validating_dsz.shape} 
                Y{validating_dsy.shape} 
        Regions: Z {region2str(self._validating_dsz.region)} 
                Y {region2str(self._validating_dsy.region)}""", )

        if len(validating_dsz.time) != len(validating_dsy.time):
            raise ValueError(
                f'The number of years of the predictand must be the '
                f'same as the number of years of the predictor: '
                f'got {len(validating_dsy.time)} and '
                f'{len(validating_dsz.time)}'
            )

        time_from_here()

        common_z_land_mask = validating_dsz.land_data.land_mask | training_mca._dsz.land_data.land_mask
        common_y_land_mask = validating_dsy.land_data.land_mask | training_mca._dsy.land_data.land_mask

        self.psi = np.zeros([1, training_mca._dsy.data.shape[0], training_mca._dsz.data.shape[0]], dtype=np.float32)
        self.zhat = np.zeros([1, validating_dsz.land_data.shape[0], validating_dsz.time.shape[0]], dtype=np.float32)

        self.psi[:, common_y_land_mask, :] = np.nan
        self.psi[:, :, common_z_land_mask] = np.nan
        self.zhat[:, common_z_land_mask] = np.nan

        self.psi[0, ~np.isnan(self.psi[0])] = calculate_psi(
            self._training_mca.SUY[~common_y_land_mask, :],
            self._training_mca.Us[:, :],
            self._training_mca._dsz.data[~common_z_land_mask, :],
            self._training_mca.Us.shape[1],
            self._training_mca._dsy.data.shape[0],
            self._training_mca.Us.shape[0],
        ).reshape((~common_y_land_mask).sum() * (~common_z_land_mask).sum())

        self.zhat[0, ~common_z_land_mask, :] = np.dot(
            validating_dsy.land_data.not_land_values[:, :].T,
            self.psi[0, ~common_y_land_mask, :][:, ~common_z_land_mask]).T

        new_z_land_array = LandArray(self._validating_dsz.data)
        new_z_land_array.update_land(common_z_land_mask)
        self.r_z_zhat_t_accumulated_modes, self.p_z_zhat_t_accumulated_modes, \
            _r_z_zhat_t_separated_modes, _p_z_zhat_t_separated_modes \
            = calculate_time_correlation(new_z_land_array, self.zhat)

        self.r_z_zhat_s_accumulated_modes, self.p_z_zhat_s_accumulated_modes, \
            _r_z_zhat_s_separated_modes, _p_z_zhat_s_separated_modes \
            = calculate_space_correlation(new_z_land_array, self.zhat)

        debugprint(f'\n\tTook: {time_to_here():.03f} seconds')

    @property
    def training_mca(self) -> MCA:
        """Training mca introduced"""
        return self._training_mca

    @property
    def validating_dsy(self) -> Preprocess:
        """Preprocessed dataset introduced as validating predictor"""
        return self._validating_dsy

    @property
    def validating_dsz(self) -> Preprocess:
        """Preprocessed dataset introduced as validating predictand"""
        return self._validating_dsz

    @classmethod
    def load(cls, prefix: str, folder: str = '.', *,
             validating_dsy: Optional[Preprocess] = None,
             validating_dsz: Optional[Preprocess] = None,
             training_mca: Optional[MCA] = None,
             **attrs: Any) -> 'Validation':
        """Load an Validation object from matrices and type

        Parameters
        ----------
        prefix : str
            Prefix of the files containing the information for the object
        folder : str
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

        self: Validation = super().load(prefix, folder)

        self._validating_dsy = validating_dsy
        self._validating_dsz = validating_dsz
        self._training_mca = training_mca
        return self

    def plot(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        folder: Optional[str] = None,
        name: Optional[str] = None,
        cmap: Optional[str] = None,
        map_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        version: Literal["default", "elena"] = "default",
        mca: Optional[MCA] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot the Crossvalidation results

        Parameters
        ----------
        save_fig
            Saves the fig in with `folder` / `name` parameters
        show_plot
            Shows the plot
        halt_program
            Only used if `show_plot` is `True`. If `True` shows the plot if plt.show
            and stops execution. Else uses fig.show and does not halt program
        folder
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
        figsize
            Set figure size. See `plt.figure`

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
            fig, axs = _plot_validation_default(self, figsize, cmap, map_ticks)
        elif version == "elena":
            if mca is None:
                raise TypeError("Expected argument `mca` for version `elena`")
            if map_ticks is not None:
                raise TypeError("Unexpected argument `map_ticks` for version `elena`")
            if cmap is not None:
                raise TypeError("Unexpected argument `cmap` for version `elena`")
            fig, axs = _plot_validation_elena(self, figsize, mca)
        else:
            raise ValueError(f"Version can only be one of: `elena`, `default`")

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'crossvalidation-plot_z-{self._validating_dsz.var}_y-{self._validating_dsy.var}.png')
        else:
            path = os.path.join(folder, name)

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
        folder: Optional[str] = None,
        name: Optional[str] = None,
        cmap: str = 'bwr',
        yticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        zticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
        """Plots the map of Zhat

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
            Year to plot
        folder
            Directory to save fig if `save_fig` is `True`
        name
            Name of the fig saved if `save_fig` is `True`
        cmap
            Colormap for the predicting map
        yticks
            Ticks for the y map
        zticks
            Ticks for the z map
        figsize
            Set figure size. See `plt.figure`

        Returns
        -------
        plt.Figure
            Figure object from matplotlib

        Sequence[plt.Axes]
            Tuple of axes in figure
        """
        nts, nylat, nylon = len(self._validating_dsy.time), len(self._validating_dsy.lat), len(self._validating_dsy.lon)
        nts, nzlat, nzlon = len(self._validating_dsz.time), len(self._validating_dsz.lat), len(self._validating_dsz.lon)

        height = nylat + nzlat + nzlat
        width = max(nzlon, nylon)

        figsize = _calculate_figsize(height / width, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)
        ax0 = plt.subplot(311, projection=ccrs.PlateCarree(0 if self.validating_dsy.region.lon0 < self.validating_dsy.region.lonf else 180))
        ax1 = plt.subplot(312, projection=ccrs.PlateCarree(0 if self.validating_dsz.region.lon0 < self.validating_dsz.region.lonf else 180))
        ax2 = plt.subplot(313, projection=ccrs.PlateCarree(0 if self.validating_dsz.region.lon0 < self.validating_dsz.region.lonf else 180))

        zindex = _get_index_from_sy(self._validating_dsz.time, year)
        yindex = zindex
        y_year = self._validating_dsy.time.values[yindex]

        d0 = self._validating_dsy.data.transpose().reshape((nts, nylat, nylon))

        if self._validating_dsy.region.lon0 < self._validating_dsy.region.lonf:
            y_xlim = sorted((self._validating_dsy.lon.values[0], self._validating_dsy.lon.values[-1]))
        else:
            y_xlim = [self._validating_dsy.region.lon0 - 180, self._validating_dsy.region.lonf + 180]
        _plot_map(d0[yindex], self._validating_dsy.lat, self._validating_dsy.lon, fig, ax0, f'Y on year {y_year}', ticks=yticks, xlim=y_xlim)

        d1 = self.zhat.transpose().reshape((nts, nzlat, nzlon))
        d2 = self._validating_dsz.data.transpose().reshape((nts, nzlat, nzlon))

        n = 30
        _std = np.nanstd(d2[zindex])
        _m = np.nanmean(d2[zindex])
        bound = max(abs(_m - _std), abs(_m + _std))
        levels = np.linspace(-bound, bound, n)

        if self._validating_dsz.region.lon0 < self._validating_dsz.region.lonf:
            z_xlim = sorted((self._validating_dsz.lon.values[0], self._validating_dsz.lon.values[-1]))
        else:
            z_xlim = [self._validating_dsz.region.lon0 - 180, self._validating_dsz.region.lonf + 180]
        _plot_map(
            d1[zindex], self._validating_dsz.lat, self._validating_dsz.lon, fig, ax1, f'Zhat on year {year}',
            cmap=cmap, levels=levels, ticks=zticks, xlim=z_xlim,
        )
        _plot_map(
            d2[zindex], self._validating_dsz.lat, self._validating_dsz.lon, fig, ax2, f'Z on year {year}',
            cmap=cmap, levels=levels, ticks=zticks, xlim=z_xlim,
        )

        fig.suptitle(
            f'Z({self._validating_dsz.var}): {region2str(self._validating_dsz.region)}, '
            f'Y({self._validating_dsy.var}): {region2str(self._validating_dsy.region)}. '
            f'Alpha: {self._training_mca.alpha}',
            fontweight='bold'
        )

        fig.subplots_adjust(hspace=.4)

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'crossvalidation-plot_z-{self._validating_dsz.var}_y-{self._validating_dsy.var}.png')
        else:
            path = os.path.join(folder, name)

        _apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program
        )

        return fig, (ax0, ax1, ax2)


def _plot_validation_elena(
    validation: Validation,
    figsize: Optional[Tuple[float, float]],
    mca: MCA,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    raise NotImplementedError


def _plot_validation_default(
    validation: Validation,
    figsize: Optional[Tuple[float, float]],
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

    nlat, nlon = len(validation.validating_dsz.lat), (len(validation.validating_dsz.lon))

    figsize = _calculate_figsize(None, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
    fig: plt.Figure = plt.figure(figsize=figsize)

    ax00 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree(0 if validation.validating_dsz.region.lon0 < validation.validating_dsz.region.lonf else 180))
    if validation._validating_dsz.region.lon0 < validation._validating_dsz.region.lonf:
        z_xlim = sorted((validation._validating_dsz.lon.values[0], validation._validating_dsz.lon.values[-1]))
    else:
        z_xlim = [validation._validating_dsz.region.lon0 - 180, validation._validating_dsz.region.lonf + 180]
    _plot_map(
        arr=validation.r_z_zhat_s_accumulated_modes[-1, :].reshape((nlat, nlon)),
        lat=validation.validating_dsz.lat,
        lon=validation.validating_dsz.lon,
        fig=fig,
        ax=ax00,
        title='Correlation in space: z vs zhat',
        xlim=z_xlim,
        cmap=cmap,
        ticks=map_ticks,
    )

    ax01 = fig.add_subplot(1, 2, 2)
    ax01.bar(validation.validating_dsz.time, validation.r_z_zhat_t_accumulated_modes[-1, :])
    ax01.set_title('Correlation in time: z vs zhat')

    fig.suptitle(
        f'Z({validation.validating_dsz.var}): {region2str(validation.validating_dsz.region)}, '
        f'Y({validation.validating_dsy.var}): {region2str(validation.validating_dsy.region)}. '
        f'Alpha: {validation.training_mca.alpha}',
        fontweight='bold'
    )

    fig.subplots_adjust(hspace=.4)

    return fig, (ax00, ax01)
