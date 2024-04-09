import os
from math import floor
from typing import Tuple, Optional, Any, Union, Sequence, cast, Literal, List
from cartopy.util import add_cyclic_point

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib import ticker
from scipy import stats

from . import MCA
from .crossvalidation import calculate_psi, calculate_time_correlation, calculate_space_correlation
from .mca import index_regression
from .. import Region
from .._functions import debugprint, region2str, time_from_here, time_to_here, _debuginfo
from .._procedure import _Procedure, _apply_flags_to_fig, _plot_map, _get_index_from_sy, _calculate_figsize, MAX_WIDTH, \
    MAX_HEIGHT, _plot_ts, _get_xlim_from_region, _get_central_longitude_from_region
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

    Examples
    --------
    Run the MCA with a training_predicting and a training_predictor field

    >>> from spy4cast import Dataset, Region, Month
    >>> from spy4cast.spy4cast import MCA, Preprocess
    >>> t_y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
    ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 1990)))
    >>> t_z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
    ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 1991)))
    >>> mca = MCA(t_y, t_z, 3, 0.01)
    >>> # Validate on the same data but with a different period
    >>> v_y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
    ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 2000, 2010)))
    >>> v_z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
    ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 2001, 2011)))
    >>> val = Validation(mca, v_y, v_z)

    All the :doc:`/variables/validation` easily accesioble

    >>> cor = val.r_z_zhat_s_separated_modes.reshape((3, len(v_z.lat), len(v_z.lon)))  # 3 is the number of modes
    >>> # Plot with any plotting library
    >>> import matplotlib.pyplot as plt
    >>> import cartopy.crs as ccrs
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection=ccrs.PlateCarree())
    >>> ax.contourf(v_z.lon, v_z.lat, cor[0, :, :])
    >>> ax.coastlines()

    Save the data in .npy to use in a different run

    >>> val.save("saved_validation_", folder="saved_data")

    Reuse the previuosly ran data easily with one line

    >>> val = Validation.load(
    ...     "saved_validation_", folder="saved_data", 
    ...     validating_dsy=v_y, validating_dsz=v_z, training_mca=mca
    ... )  # IMPORTANT TO USE validating_dsy=, validating_dsz= and training_mca=

    Plot with one line and several options

    >>> # plot_type=pcolor to use pcolormesh, change the default cmap and figisze with a single option
    >>> # halt_program=False does not halt execution and lets us create two plots at the same time: crossvalidation 
    >>> val.plot_zhat(1990, show_plot=True, halt_program=False, cmap="jet", figsize=(20, 10), plot_type="pcolor")  
    >>> val.plot(show_plot=True, halt_program=True, cmap="jet", figsize=(20, 10), plot_type="pcolor")

    Attributes
    ----------
        psi_accumulated_modes : npt.NDArray[np.float32]
            Psi calculated with the training MCA data. Dimension: 1 x training_y_space x training_z_space
        zhat_accumulated_modes : npt.NDArray[np.float32]
            Zhat predicted for the predictand using all modes accumulated. Dimension: 1 x validating_z_space x validating_z_time
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
            'psi_accumulated_modes',
            'zhat_accumulated_modes',
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

        self.psi_accumulated_modes = np.zeros([1, training_mca._dsy.data.shape[0], training_mca._dsz.data.shape[0]], dtype=np.float32)
        self.zhat_accumulated_modes = np.zeros([1, validating_dsz.land_data.shape[0], validating_dsz.time.shape[0]], dtype=np.float32)

        self.psi_accumulated_modes[:, common_y_land_mask, :] = np.nan
        self.psi_accumulated_modes[:, :, common_z_land_mask] = np.nan
        self.zhat_accumulated_modes[:, common_z_land_mask] = np.nan

        self.psi_accumulated_modes[0, ~np.isnan(self.psi_accumulated_modes[0])] = calculate_psi(
            self._training_mca.SUY[~common_y_land_mask, :],
            self._training_mca.Us[:, :],
            self._training_mca._dsz.data[~common_z_land_mask, :],
            self._training_mca.Us.shape[1],
            self._training_mca._dsy.data.shape[0],
            self._training_mca.Us.shape[0],
            self._training_mca.scf
        ).reshape((~common_y_land_mask).sum() * (~common_z_land_mask).sum())

        self.zhat_accumulated_modes[0, ~common_z_land_mask, :] = np.dot(
            validating_dsy.land_data.not_land_values[:, :].T,
            self.psi_accumulated_modes[0, ~common_y_land_mask, :][:, ~common_z_land_mask]).T

        new_z_land_array = LandArray(self._validating_dsz.data)
        new_z_land_array.update_land(common_z_land_mask)
        self.r_z_zhat_t_accumulated_modes, self.p_z_zhat_t_accumulated_modes, \
            _r_z_zhat_t_separated_modes, _p_z_zhat_t_separated_modes \
            = calculate_time_correlation(new_z_land_array, self.zhat_accumulated_modes)

        self.r_z_zhat_s_accumulated_modes, self.p_z_zhat_s_accumulated_modes, \
            _r_z_zhat_s_separated_modes, _p_z_zhat_s_separated_modes \
            = calculate_space_correlation(new_z_land_array, self.zhat_accumulated_modes)

        debugprint(f'\n\tTook: {time_to_here():.03f} seconds')

    @property
    def training_mca(self) -> MCA:
        """Training mca used for validation"""
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
        """Load an Validation object from .npy files saved with Validation.save.

        Parameters
        ----------
        prefix : str
            Prefix of the files containing the information for the object
        folder : str
            Directory of the files
        validating_dsy : Preprocess
            ONLY KEYWORD ARGUMENT. Preprocessed dataset of the validating predictor variable
        validating_dsz : Preprocess
            Preprocessed dataset of the validating predicting variable
        training_mca : MCA
            ONLY KEYWORD ARGUMENT. Training mca

        Returns
        -------
            Validation

        Examples
        --------
        Load with Validation.load using the same validating datsets and training mca
        as when the methodology was run

        >>> val = Validation.load(
        ...     "saved_validation_", folder="saved_data", 
        ...     validating_dsy=validating_y, validating_dsz=validating_z, training_mca=validating_mca
        ... )  # IMPORTANT TO USE validating_dsy=, validating_dsz= and training_mca=

        Save: on a previous run the validation is calcuated
        
        >>> from spy4cast import Dataset, Region, Month
        >>> from spy4cast.spy4cast import Validation, MCA, Preprocess
        >>> t_y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 1990)))
        >>> t_z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 1991)))
        >>> training_mca = MCA(t_y, t_z, 3, 0.01)
        >>> # Validate on the same data but with a different period
        >>> validating_y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 2000, 2010)))
        >>> validating_z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 2001, 2011)))
        >>> val = Validation(training_mca, validating_y, validating_z)
        >>> val.save("saved_validation_", folder="data")  # Save the output

        Load: To avoid running the methodology again for plotting and analysis load the data directly

        >>> from spy4cast import Dataset, Region, Month
        >>> from spy4cast.spy4cast import Validation, MCA, Preprocess
        >>> t_y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 1990)))
        >>> t_z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 1991)))
        >>> training_mca = MCA(t_y, t_z, 3, 0.01)
        >>> # Validate on the same data but with a different period
        >>> validating_y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 2000, 2010)))
        >>> validating_z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 2001, 2011)))
        >>> val = Validation.load("saved_validation_", "data", 
        ...     training_mca=training_mca, validating_y=validating_y, validating_y=validating_z)

        Then you can plot as usual

        >>> val.plot(save_fig=True, name="cross.png")
        >>> val.plot_zhat(2004, save_fig=True, name="zhat_1999.png")

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
        map_levels: Optional[
            Union[npt.NDArray[np.float32], Sequence[float], bool]
        ] = None,
        version: Literal["default", 2] = "default",
        mca: Optional[MCA] = None,
        figsize: Optional[Tuple[float, float]] = None,
        nm: Optional[int] = None,
        plot_type: Literal["contour", "pcolor"] = "contour",
    ) -> Tuple[Tuple[plt.Figure], Tuple[plt.Axes, ...]]:
        """Plot the Validation results

        Parameters
        ----------
        save_fig
            Saves the fig using `folder` and `name` parameters
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
        map_levels
            Levels for the z map in version default
        version
            Select version from: `default` and `2`
        mca
            MCA results for version `2`
        figsize
            Set figure size. See `plt.figure`
        nm : int, optional
            Number of modes to use for the corssvalidation plot. Must be less than or equal
            to nm used to run the methodology. If -1 use all modes.
        plot_type : {"contour", "pcolor"}, defaut = "pcolor"
            Plot type. If `contour` it will use function `ax.contourf`, 
            if `pcolor` `ax.pcolormesh`.

        Returns
        -------
        Tuple[plt.Figure]
            Figures object from matplotlib

        Tuple[plt.Axes]
            Tuple of axes in figure

        Examples
        --------

        Plot and halt the program

        >>> val.plot(show_plot=True, halt_program=True)

        Save the plot 

        >>> val.plot(save_fig=True, name="val_plot.png")

        Plot with pcolormesh and be precise with the resolution

        >>> val.plot(save_fig=True, name="val.png", plot_type="pcolor")

        Plot and not halt the program

        >>> val.plot(show_plot=True)
        >>> # .... Compute a new validation for example
        >>> import matplotlib.pyplot as plt
        >>> plt.show()  # Will show the previously ran plot
        """
        fig: plt.Figure
        axs: Sequence[plt.Axes]
        if plot_type not in ("contour", "pcolor"):
            raise ValueError(f"Expected `contour` or `pcolor` for argument `plot_type`, but got {plot_type}")
        if version == "default":
            if mca is not None:
                raise TypeError("Unexpected argument `mca` for version `default`")
            if cmap is None:
                cmap = 'bwr'
            fig, axs = _plot_validation_default(self, figsize, cmap, map_ticks, map_levels, nm, plot_type)
        elif int(version) == 2:
            if mca is None:
                raise TypeError("Expected argument `mca` for version `2`")
            if map_ticks is not None:
                raise TypeError("Unexpected argument `map_ticks` for version `2`")
            if cmap is not None:
                raise TypeError("Unexpected argument `cmap` for version `2`")
            fig, axs = _plot_validation_2(self, figsize, mca)
        else:
            raise ValueError(f"Version can only be one of: `2`, `default`")

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

        return (fig, ), axs


    def plot_zhat(
        self,
        year: int,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        folder: Optional[str] = None,
        name: Optional[str] = None,
        cmap: str = 'bwr',
        y_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        z_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        y_levels: Optional[
            Union[npt.NDArray[np.float32], Sequence[float], bool]
        ] = None,
        z_levels: Optional[
            Union[npt.NDArray[np.float32], Sequence[float], bool]
        ] = None,
        figsize: Optional[Tuple[float, float]] = None,
        plot_type: Literal["contour", "pcolor"] = "contour",
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
        """Plots the map of Zhat

        Parameters
        ----------
        save_fig
            Saves the fig using `folder` and `name` parameters
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
        y_ticks
            Ticks for the y map
        z_ticks
            Ticks for the z map
        y_levels
            Levels for the map y
        z_levels
            Levels for the map z
        figsize
            Set figure size. See `plt.figure`
        plot_type : {"contour", "pcolor"}, defaut = "pcolor"
            Plot type. If `contour` it will use function `ax.contourf`, 
            if `pcolor` `ax.pcolormesh`.

        Returns
        -------
        plt.Figure
            Figure object from matplotlib

        Sequence[plt.Axes]
            Tuple of axes in figure

        Examples
        --------
        
        Plot a year prediction

        >>> val.plot_zhat(1990, show_plot=True, halt_program=True, save_fig=True, name="zhat_1990.png")

        """
        if plot_type not in ("contour", "pcolor"):
            raise ValueError(f"Expected `contour` or `pcolor` for argument `plot_type`, but got {plot_type}")
        
        nts, nylat, nylon = len(self._validating_dsy.time), len(self._validating_dsy.lat), len(self._validating_dsy.lon)
        nts, nzlat, nzlon = len(self._validating_dsz.time), len(self._validating_dsz.lat), len(self._validating_dsz.lon)

        height = nylat + nzlat + nzlat
        width = max(nzlon, nylon)

        figsize = _calculate_figsize(height / width, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 0.1, 1, 1, 0.1], hspace=0.7)

        # central longitude
        map_y, map_z = self._validating_dsy, self._validating_dsz
        central_longitude_y = _get_central_longitude_from_region(map_y.region.lon0, map_y.region.lonf)
        y_xlim = _get_xlim_from_region(map_y.region.lon0, map_y.region.lonf, central_longitude_y)

        central_longitude_z = _get_central_longitude_from_region(map_z.region.lon0, map_z.region.lonf)
        z_xlim = _get_xlim_from_region(map_z.region.lon0, map_z.region.lonf, central_longitude_z)

        ax0 = plt.subplot(gs[0], projection=ccrs.PlateCarree(central_longitude_y))
        ax1 = plt.subplot(gs[2], projection=ccrs.PlateCarree(central_longitude_z))
        ax2 = plt.subplot(gs[3], projection=ccrs.PlateCarree(central_longitude_z))


        zindex = _get_index_from_sy(self._validating_dsz.time, year)
        yindex = zindex
        y_year = self._validating_dsy.time.values[yindex]

        d0 = self._validating_dsy.data.transpose().reshape((nts, nylat, nylon))

        _plot_map(d0[yindex], self._validating_dsy.lat, self._validating_dsy.lon, fig, ax0, f'Y on year {y_year}', ticks=y_ticks, xlim=y_xlim,
                  add_cyclic_point=self._validating_dsy.region.lon0 >= self._validating_dsy.region.lonf, plot_type=plot_type,
                  levels=y_levels)

        d1 = self.zhat_accumulated_modes.transpose().reshape((nts, nzlat, nzlon))
        d2 = self._validating_dsz.data.transpose().reshape((nts, nzlat, nzlon))

        n = 20
        _m = np.nanmean([np.nanmean(d2), np.nanmean(d1)])
        _s = np.nanmean([np.nanstd(d2), np.nanstd(d1)])
        levels = z_levels if z_levels is not None else np.linspace(_m -2*_s, _m + 2*_s, n)

        _plot_map(
            d1[zindex], self._validating_dsz.lat, self._validating_dsz.lon, fig, ax1, f'Zhat on year {year}',
            cmap=cmap, levels=levels, ticks=z_ticks, xlim=z_xlim,
            add_cyclic_point=self._validating_dsz.region.lon0 >= self._validating_dsz.region.lonf, plot_type=plot_type,
        )
        _plot_map(
            d2[zindex], self._validating_dsz.lat, self._validating_dsz.lon, fig, ax2, f'Z on year {year}',
            cmap=cmap, levels=levels, ticks=z_ticks, xlim=z_xlim,
            add_cyclic_point=self._validating_dsz.region.lon0 >= self._validating_dsz.region.lonf, plot_type=plot_type,
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


def _plot_validation_2(
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
    ],
    map_levels: Optional[
        Union[npt.NDArray[np.float32], Sequence[float], bool]
    ],
    nm: Optional[int] = None,
    plot_type: Literal["contour", "pcolor"] = "contour",
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    figsize = _calculate_figsize(1.5/3, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(4, 6, height_ratios=[1, 0.08, 1, 0.08], width_ratios=[1, 1, 1, 1, 1, 1], wspace=1, hspace=0.5)

    # central longitude
    map_z = validation._validating_dsz
    central_longitude_z = _get_central_longitude_from_region(map_z.region.lon0, map_z.region.lonf)
    z_xlim = _get_xlim_from_region(map_z.region.lon0, map_z.region.lonf, central_longitude_z)

    axs = (
        fig.add_subplot(gs[0, 0:3], projection=ccrs.PlateCarree(central_longitude_z)),
        fig.add_subplot(gs[0:2, 3:6]),
        fig.add_subplot(gs[2, 0:3], projection=ccrs.PlateCarree(central_longitude_z)),
        fig.add_subplot(gs[2:4, 3:6]),
    )

    nzlat, nzlon = len(validation.validating_dsz.lat), (len(validation.validating_dsz.lon))

    # ------ r_z_zhat_s and p_z_zhat_s ------ #
    # Correlation map
    if nm is not None and not 1 <= nm <= validation.r_z_zhat_s_accumulated_modes.shape[0]:
        raise ValueError(f"Parameter `nm` must be positive an less than or equal to the number of modes used in the methodology, {validation.r_z_zhat_s_accumulated_modes.shape[0]}, but got {nm}")
    d = validation.r_z_zhat_s_accumulated_modes[(-1 if nm is None else nm - 1), :].transpose().reshape((nzlat, nzlon))
    _mean = np.nanmean(d)
    _std = np.nanstd(d)
    mx = _mean + _std
    mn = _mean - _std
    im = _plot_map(
        d, validation.validating_dsz.lat, validation.validating_dsz.lon, fig, axs[0],
        'Correlation in space between z and zhat',
        cmap=cmap,
        ticks=(np.arange(round(mn * 10) / 10, floor(mx * 10) / 10 + .05, .1) if map_ticks is None and not np.isnan(_mean) and not np.isnan(_std) else map_ticks),
        levels=map_levels,
        xlim=z_xlim,
        colorbar=False,
        add_cyclic_point=validation.validating_dsz.region.lon0 >= validation.validating_dsz.region.lonf,
        plot_type=plot_type,
    )

    hatches = d.copy()
    hatches[((validation.p_z_zhat_s_accumulated_modes[-1, :] > validation.training_mca.alpha) | (
                validation.r_z_zhat_s_accumulated_modes[-1, :] < 0)).transpose().reshape((nzlat, nzlon))] = np.nan
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 0:3]), orientation='horizontal', ticks=map_ticks)
    if map_ticks is None:
        tick_locator = ticker.MaxNLocator(nbins=5, prune='both', steps=[2, 5])
        #ticks = tick_locator.tick_values(vmin=cb.vmin, vmax=cb.vmax)
        #cb.ax.set_xticks(ticks)
        cb.ax.xaxis.set_major_locator(tick_locator)

    axs[0].contourf(
        validation.validating_dsz.lon, validation.validating_dsz.lat, hatches,
        colors='none', hatches='..', extend='both',
        transform=ccrs.PlateCarree()
    )
    # ^^^^^^ r_z_zhat_s and p_z_zhat_s ^^^^^^ #

    # ------ r_z_zhat_t and p_z_zhat_t ------ #
    axs[1].bar(validation.validating_dsz.time.values, validation.r_z_zhat_t_accumulated_modes[-1, :])
    axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axs[1].scatter(
        validation.validating_dsz.time[validation.p_z_zhat_t_accumulated_modes[-1, :] <= validation.training_mca.alpha],
        validation.r_z_zhat_t_accumulated_modes[-1, :][validation.p_z_zhat_t_accumulated_modes[-1, :] <= validation.training_mca.alpha]
    )
    axs[1].set_title('Correlation in space between z and zhat')
    axs[1].grid(True)
    # ^^^^^^ r_z_zhat_t and p_z_zhat_t ^^^^^^ #

    # RMSE
    lon = validation.validating_dsz.lon
    lat = validation.validating_dsz.lat
    time = validation.validating_dsz.time
    nlon, nlat, nt = len(lon), len(lat), len(time)
    zhat = validation.zhat_accumulated_modes[-1, :]  # space x time
    zdata = validation.validating_dsz.data  # space x time

    rmse_map = np.sqrt(np.nansum((zhat - zdata)**2, axis=1) / nt).reshape((nlat, nlon))

    im = _plot_map(
        d, validation.validating_dsz.lat, validation.validating_dsz.lon, fig, axs[2],
        'RMSE',
        cmap="Reds",
        ticks=None,
        levels=None,
        xlim=z_xlim,
        colorbar=False,
        add_cyclic_point=validation.validating_dsz.region.lon0 >= validation.validating_dsz.region.lonf,
        plot_type=plot_type,
    )
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[3, 0:3]), orientation='horizontal')

    # RMSE time series
    rmse_ts = np.sqrt(np.nansum((zhat - zdata)**2, axis=0) / (nlat * nlon))
    axs[3].bar(validation.validating_dsz.time.values, rmse_ts, color="orange")
    axs[3].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    axs[3].set_title('RMSE time series')
    axs[3].grid(True)

    fig.suptitle(
        f'Z({validation.validating_dsz.var}): {region2str(validation.validating_dsz.region)}, '
        f'Y({validation.validating_dsy.var}): {region2str(validation.validating_dsy.region)}. '
        f'Alpha: {validation.training_mca.alpha}',
        fontweight='bold'
    )

    fig.subplots_adjust(hspace=.4)

    return fig, axs

