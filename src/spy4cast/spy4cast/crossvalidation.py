import os
from math import floor
from typing import Tuple, Optional, Any, Union, Sequence, cast, Literal, List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
from scipy import stats

from . import MCA
from .mca import index_regression, calculate_psi
from .. import Region
from .._functions import debugprint, region2str, time_from_here, time_to_here, _debuginfo
from .._procedure import _Procedure, _apply_flags_to_fig, _plot_map, _get_index_from_sy, _calculate_figsize, MAX_WIDTH, \
    MAX_HEIGHT, _plot_ts, _get_central_longitude_from_region, _get_xlim_from_region
from ..land_array import LandArray
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
        sig : {'monte-carlo', 'test-t'}
            Signification technique: monte-carlo or test-t

    Examples
    --------
    Run the methodology with a predicting and a predictor field

    >>> from spy4cast import Dataset, Region, Month
    >>> from spy4cast.spy4cast import MCA, Preprocess
    >>> y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
    ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 2010)))
    >>> z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
    ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 2011)))
    >>> map_y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
    ...         Region(-80, 30, -70, 50, Month.JUN, Month.AUG, 1960, 2010)))
    >>> map_z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
    ...         Region(-60, 60, -150, 150, Month.DEC, Month.FEB, 1961, 2011)))
    >>> mca = MCA(y, z, 3, 0.01)

    All the :doc:`/variables/crossvalidation` easily accesioble

    >>> cor = cross.r_z_zhat_s_separated_modes.reshape((3, len(z.lat), len(z.lon)))  # 3 is the number of modes
    >>> # Plot with any plotting library
    >>> import matplotlib.pyplot as plt
    >>> import cartopy.crs as ccrs
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection=ccrs.PlateCarree())
    >>> ax.contourf(z.lon, z.lat, cor[0, :, :])
    >>> ax.coastlines()

    Save the data in .npy to use in a different run

    >>> cross.save("saved_cross_", folder="saved_data")

    Reuse the previuosly ran data easily with one line

    >>> cross = Crossvalidation.load("saved_cross_", folder="saved_data", dsy=y, dsz=z)  # IMPORTANT TO USE dsy= and dsz=

    Plot with one line and several options

    >>> # plot_type=pcolor to use pcolormesh, change the default cmap and figisze with a single option
    >>> # halt_program=False does not halt execution and lets us create two plots at the same time: crossvalidation 
    >>> cross.plot_zhat(1990, show_plot=True, halt_program=False, cmap="jet", figsize=(20, 10), plot_type="pcolor")  
    >>> cross.plot(show_plot=True, halt_program=True, cmap="jet", figsize=(20, 10), plot_type="pcolor")

    Attributes
    ----------
        zhat_separated_modes 
            Hindcast of field to predict using crosvalidation for each individual mode
        zhat_accumulated_modes 
            Hindcast of field to predict using crosvalidation for n modes (1, 1->2, ..., 1->nm)
        scf 
            Squared covariance fraction of the mca for each mode
        r_z_zhat_t_separated_modes 
            Correlation between zhat and Z for each time (time series) for each individual mode
        r_z_zhat_t_accumulated_modes 
            Correlation between zhat and Z for each time (time series) for n modes (1, 1->2, ..., 1->nm)
        p_z_zhat_t_separated_modes 
            P values of rt for each individual mode
        p_z_zhat_t_accumulated_modes 
            P values of rt for n modes (1, 1->2, ..., 1->nm)
        r_z_zhat_s_separated_modes 
            Correlation between time series (for each point) of zhat and z (map) for each individual mode
        r_z_zhat_s_accumulated_modes 
            Correlation between time series (for each point) of zhat and z (map) for n modes (1, 1->2, ..., 1->nm)
        p_z_zhat_s_separated_modes 
            P values of rr for each individual mode
        p_z_zhat_s_accumulated_modes 
            P values of rr for n modes (1, 1->2, ..., 1->nm)
        r_uv 
            Correlation score betweeen u and v for each mode
        r_uv_sig 
            Correlation score betweeen u and v for each mode where significative
        p_uv 
            P value of ruv
        suy 
            Correlation in space of the predictor with the singular vector. Dimension: y_space x time x nm
        suz 
            Correlation in space of the predictand with the singular vector. Dimension: z_space x time x nm
        suy_sig 
            Correlation in space of the predictor with the singular vector where pvalue is smaller than alpha. Dimension: y_space x time x nm
        suz_sig 
            Correlation in space of the predictand with the singular vector where pvalue is smaller than alpha. Dimension: z_space x time x nm
        us 
            Singular vectors of the predictor field. Dimension: nm x time x time
        vs 
            Singular vectors of the predictand field. Dimension: nm x time x time
        alpha 
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
    # psi_separated_modes: npt.NDArray[np.float32]
    # psi_accumulated_modes: npt.NDArray[np.float32]
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

            # 'psi_accumulated_modes',
            # 'psi_separated_modes',
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

        psi_separated_modes = np.zeros([nm, nt, ny, nz], dtype=np.float32)
        psi_accumulated_modes = np.zeros([nm, nt, ny, nz], dtype=np.float32)
        psi_separated_modes[:, :, self._dsy.land_data.land_mask, :][:, :, :, self._dsz.land_data.land_mask] = np.nan
        psi_accumulated_modes[:, :, self._dsy.land_data.land_mask, :][:, :, :, self._dsz.land_data.land_mask] = np.nan

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
                        psi_separated_modes[:, i, :, :], psi_accumulated_modes[:, i, :, :], \
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
                    psi_separated_modes[:, i, :, :], psi_accumulated_modes[:, i, :, :], \
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

    @property
    def dsy(self) -> Preprocess:
        """Preprocessed dataset introduced as predictor"""
        return self._dsy

    @property
    def dsz(self) -> Preprocess:
        """Preprocessed dataset introduced as predictand"""
        return self._dsz

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
        debugprint('\tyear:', year + 1, 'of', nt)
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
            psi_separated_modes[mode, ~np.isnan(psi_separated_modes[mode])] = calculate_psi(
                mca_out.SUY[~y2.land_mask, mode:mode + 1], 
                mca_out.Us[mode:mode + 1, :],
                z2.not_land_values, 
                nt - 1, 
                ny,
                nm,
                mca_out.scf[mode:mode + 1]
            ).reshape((~y2.land_mask).sum() * (~z2.land_mask).sum())
            psi_accumulated_modes[mode, ~np.isnan(psi_accumulated_modes[mode])] = calculate_psi(
                mca_out.SUY[~y2.land_mask, :mode + 1], 
                mca_out.Us[:mode + 1, :],
                z2.not_land_values,
                nt - 1,
                ny,
                mode + 1,
                mca_out.scf[:mode + 1]
            ).reshape((~y2.land_mask).sum() * (~z2.land_mask).sum())

            zhat_separated_modes[mode, ~z2.land_mask] = np.dot(np.transpose(y.not_land_values[:, year]), psi_separated_modes[mode, ~y2.land_mask, :][:, ~z2.land_mask])
            zhat_accumulated_modes[mode, ~z2.land_mask] = np.dot(np.transpose(y.not_land_values[:, year]), psi_accumulated_modes[mode, ~y2.land_mask, :][:, ~z2.land_mask])

        r_uv = np.zeros(nm, dtype=np.float32)
        r_uv_sig = np.zeros(nm, dtype=np.float32)
        p_uv = np.zeros(nm, dtype=np.float32)
        for m in range(nm):
            r_uv[m], p_uv[m], r_uv_sig[m], _, _ = index_regression(mca_out.Us[m, :], mca_out.Vs[m, :].T, alpha, sig, montecarlo_iterations)

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
        """Plot the Crossvalidation results

        Parameters
        ----------
        save_fig
            Saves the fig using `folder` and `name` parameters
        show_plot
            Shows the plot but does NOT stop the program. Calls `fig.show`. 
            If you want the behaviour of `plt.plot` add the halt_program option.
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
        figures : Tuple[plt.Figure]
            Figures objects from matplotlib. In this case just one figure

        ax : Tuple[plt.Axes]
            Tuple of axes in figure. In the default case: 2 axes

        Examples
        --------

        Plot and halt the program

        >>> cross.plot(show_plot=True, halt_program=True)

        Save the plot 

        >>> cross.plot(save_fig=True, name="cross_plot.png")

        Plot with pcolormesh and be precise with the resolution

        >>> cross.plot(save_fig=True, name="cross_plot.png", plot_type="pcolor")

        Plot cross result in a bigger region

        >>> from spy4cast import Dataset, Region, Month
        >>> from spy4cast.spy4cast import Crossvalidation, Preprocess
        >>> y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 2010)))
        >>> z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 2011)))
        >>> map_y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-80, 30, -70, 50, Month.JUN, Month.AUG, 1960, 2010)))
        >>> map_z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-60, 60, -150, 150, Month.DEC, Month.FEB, 1961, 2011)))
        >>> cross = Crossvalidation(y, z, 3, 0.01)
        >>> cross.plot(show_plot=True, halt_program=True, map_y=map_y, map_z=map_z)


        Plot and not halt the program

        >>> cross.plot(show_plot=True)
        >>> # .... Compute crossvalidation for example
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
                cmap = 'jet'
            fig, axs = _plot_crossvalidation_default(self, figsize, cmap, map_ticks, map_levels, nm, plot_type)
        elif int(version) == 2:
            if mca is None:
                raise TypeError("Expected argument `mca` for version `2`")
            if map_ticks is not None:
                raise TypeError("Unexpected argument `map_ticks` for version `2`")
            if map_levels is not None:
                raise TypeError("Unexpected argument `map_levels` for version `2`")
            if cmap is not None:
                raise TypeError("Unexpected argument `cmap` for version `2`")
            fig, axs = _plot_crossvalidation_2(self, figsize, mca)
        else:
            raise ValueError(f"Version can only be one of: `2`, `default`")

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'crossvalidation-plot_z-{self._dsz.var}_y-{self._dsy.var}.png')
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

        Examples
        --------
        
        Plot a year prediction

        >>> cross.plot_zhat(1990, show_plot=True, halt_program=True, save_fig=True, name="zhat_1990.png")

        Returns
        -------
        figure : plt.Figure
            Figure object from matplotlib. In this case just one

        axes : Sequence[plt.Axes]
            Tuple of axes in figure. In this case 3: y, z and zhat
        """
        if plot_type not in ("contour", "pcolor"):
            raise ValueError(f"Expected `contour` or `pcolor` for argument `plot_type`, but got {plot_type}")

        nts, nylat, nylon = len(self._dsy.time), len(self._dsy.lat), len(self._dsy.lon)
        nts, nzlat, nzlon = len(self._dsz.time), len(self._dsz.lat), len(self._dsz.lon)

        figsize = _calculate_figsize(1.5, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 0.1, 1, 1, 0.1], hspace=0.7)

        # central longitude
        map_y, map_z = self._dsy, self._dsz
        central_longitude_y = _get_central_longitude_from_region(map_y.region.lon0, map_y.region.lonf)
        y_xlim = _get_xlim_from_region(map_y.region.lon0, map_y.region.lonf, central_longitude_y)

        central_longitude_z = _get_central_longitude_from_region(map_z.region.lon0, map_z.region.lonf)
        z_xlim = _get_xlim_from_region(map_z.region.lon0, map_z.region.lonf, central_longitude_z)

        ax0 = plt.subplot(gs[0], projection=ccrs.PlateCarree(central_longitude_y))
        ax1 = plt.subplot(gs[2], projection=ccrs.PlateCarree(central_longitude_z))
        ax2 = plt.subplot(gs[3], projection=ccrs.PlateCarree(central_longitude_z))

        zindex = _get_index_from_sy(self._dsz.time, year)
        yindex = zindex
        y_year = self._dsy.time.values[yindex]

        d0 = self._dsy.data.transpose().reshape((nts, nylat, nylon))

        _plot_map(d0[yindex], self._dsy.lat, self._dsy.lon, fig, ax0, f'Y on year {y_year}', ticks=y_ticks, xlim=y_xlim, 
                  cax=fig.add_subplot(gs[1]), add_cyclic_point=self.dsy.region.lon0 >= self.dsy.region.lonf, plot_type=plot_type,
                  levels=y_levels)

        d1 = self.zhat_accumulated_modes[-1, :].transpose().reshape((nts, nzlat, nzlon))
        d2 = self._dsz.data.transpose().reshape((nts, nzlat, nzlon))

        n = 20
        _m = np.nanmean([np.nanmean(d2), np.nanmean(d1)])
        _s = np.nanmean([np.nanstd(d2), np.nanstd(d1)])
        levels = z_levels if z_levels is not None else np.linspace(_m -2*_s, _m + 2*_s, n)
        _plot_map(
            d1[zindex], self._dsz.lat, self._dsz.lon, fig, ax1, f'Zhat on year {year}',
            cmap=cmap, levels=levels, ticks=z_ticks, xlim=z_xlim, colorbar=False,
            add_cyclic_point=self.dsz.region.lon0 >= self.dsz.region.lonf, plot_type=plot_type,
        )
        _plot_map(
            d2[zindex], self._dsz.lat, self._dsz.lon, fig, ax2, f'Z on year {year}',
            cmap=cmap, levels=levels, ticks=z_ticks, xlim=z_xlim, cax=fig.add_subplot(gs[4]),
            add_cyclic_point=self.dsz.region.lon0 >= self.dsz.region.lonf, plot_type=plot_type,
        )

        fig.suptitle(
            f'Z({self._dsz.var}): {region2str(self._dsz.region)}, '
            f'Y({self._dsy.var}): {region2str(self._dsy.region)}. '
            f'Alpha: {self.alpha}',
            fontweight='bold'
        )

        fig.subplots_adjust(hspace=.4)

        if folder is None:
            folder = '.'
        if name is None:
            path = os.path.join(folder, f'crossvalidation-plot_z-{self._dsz.var}_y-{self._dsy.var}.png')
        else:
            path = os.path.join(folder, name)

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
        folder: str = '.',
        *,
        dsz: Optional[Preprocess] = None,
        dsy: Optional[Preprocess] = None,
        **attrs: Any
    ) -> 'Crossvalidation':
        """Load an Crossvalidation object from .npy files saved in Crossvalidation.save.

        Parameters
        ----------
        prefix : str
            Prefix of the files containing the information for the object
        folder : str
            Directory of the files
        dsy : Preprocess
            ONLY KEYWORD ARGUMENT. Preprocessed dataset of the predictor variable
        dsz : Preprocess
            ONLY KEYWORD ARGUMENT. Preprocessed dataset of the predicting variable

        Returns
        -------
            Crossvalidation

        Examples
        --------
        Load with just one line

        >>> cross = Crossvalidation.load(prefix="cross_", folder="saved_data")

        Save: on a previous run the crossvalidation is calcuated
        
        >>> from spy4cast import Dataset, Region, Month
        >>> from spy4cast.spy4cast import Crossvalidation, Preprocess
        >>> y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 2010)))
        >>> z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 2011)))
        >>> cross = Crossvalidation(y, z, 3, 0.01)
        >>> cross.save("saved_cross_", folder="data")  # Save the output

        Load: To avoid running the methodology again for plotting and analysis load the data directly

        >>> from spy4cast.spy4cast import Crossvalidation, Preprocess
        >>> from spy4cast import Dataset, Region, Month
        >>> y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 2010)))  # YOU SHOULD USE THE SAME REGION
        >>> z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 2011)))
        >>> cross = Crossvalidation.load("saved_cross_", folder="data", dsy=y, dsz=z)  # IMPORTANT TO USE dsy= and dsz=

        Then you can plot as usual

        >>> cross.plot(save_fig=True, name="cross.png")
        >>> cross.plot_zhat(1999, save_fig=True, name="zhat_1999.png")

        """
        if len(attrs) != 0:
            raise TypeError('Load only takes two keyword arguments: dsz and dsy')
        if dsz is None or dsy is None:
            raise TypeError('To load an Crossvalidation object you must provide `dsz` and `dsy` keyword arguments')
        if type(dsz) != Preprocess or type(dsy) != Preprocess:
            raise TypeError(f'Unexpected types ({type(dsz)} and {type(dsy)}) for `dsz` and `dsy`. Expected type `Preprocess`')

        self: Crossvalidation = super().load(prefix, folder)
        self._dsz = dsz
        self._dsy = dsy
        return self


def _plot_crossvalidation_2(
    cross: Crossvalidation,
    figsize: Optional[Tuple[float, float]],
    mca: MCA,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    figsize = (24, 8.5) if figsize is None else figsize
    fig = plt.figure(figsize=figsize)
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

        ax2 = fig.add_subplot(spec[2, n_mode], projection=ccrs.PlateCarree(0 if cross.dsz.region.lon0 < cross.dsz.region.lonf else 180))
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
            ax3 = fig.add_subplot(spec[3, n_mode], projection=ccrs.PlateCarree(0 if cross.dsz.region.lon0 < cross.dsz.region.lonf else 180))
            im = ax3.contourf(cross.dsz.lon, cross.dsz.lat, sk - sk_i, levels=levels, cmap='coolwarm', transform=ccrs.PlateCarree())
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
    map_z = cross._dsz

    central_longitude_z = _get_central_longitude_from_region(map_z.region.lon0, map_z.region.lonf)
    z_xlim = _get_xlim_from_region(map_z.region.lon0, map_z.region.lonf, central_longitude_z)

    axs = (
        fig.add_subplot(gs[0, 0:3], projection=ccrs.PlateCarree(central_longitude_z)),
        fig.add_subplot(gs[0:2, 3:6]),
        fig.add_subplot(gs[2, 0:3], projection=ccrs.PlateCarree(central_longitude_z)),
        fig.add_subplot(gs[2:4, 3:6]),
    )

    nzlat = len(cross._dsz.lat)
    nzlon = len(cross._dsz.lon)
    # nztime = len(ts)

    # ------ r_z_zhat_s and p_z_zhat_s ------ #
    # Correlation map
    if nm is not None and not 1 <= nm <= cross.r_z_zhat_s_accumulated_modes.shape[0]:
        raise ValueError(f"Parameter `nm` must be positive an less than or equal to the number of modes used in the methodology, {cross.r_z_zhat_s_accumulated_modes.shape[0]}, but got {nm}")
    d = cross.r_z_zhat_s_accumulated_modes[(-1 if nm is None else nm - 1), :].transpose().reshape((nzlat, nzlon))
    _mean = np.nanmean(d)
    _std = np.nanstd(d)
    mx = _mean + _std
    mn = _mean - _std
    im = _plot_map(
        d, cross._dsz.lat, cross._dsz.lon, fig, axs[0],
        'ACC map',
        cmap=cmap,
        ticks=(np.arange(round(mn * 10) / 10, floor(mx * 10) / 10 + .05, .1) if map_ticks is None and not np.isnan(_mean) and not np.isnan(_std) else map_ticks),
        levels=map_levels,
        xlim=z_xlim,
        colorbar=False,
        add_cyclic_point=cross.dsz.region.lon0 >= cross.dsz.region.lonf,
        plot_type=plot_type,
    )

    hatches = d.copy()
    hatches[((cross.p_z_zhat_s_accumulated_modes[-1, :] > cross.alpha) | (
                cross.r_z_zhat_s_accumulated_modes[-1, :] < 0)).transpose().reshape((nzlat, nzlon))] = np.nan
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 0:3]), orientation='horizontal', ticks=map_ticks)
    if map_ticks is None:
        tick_locator = ticker.MaxNLocator(nbins=5, prune='both', steps=[2, 5])
        #ticks = tick_locator.tick_values(vmin=cb.vmin, vmax=cb.vmax)
        #cb.ax.set_xticks(ticks)
        cb.ax.xaxis.set_major_locator(tick_locator)

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
        cross.r_z_zhat_t_accumulated_modes[-1, :][cross.p_z_zhat_t_accumulated_modes[-1, :] <= cross.alpha]
    )
    axs[1].set_title('ACC time series')
    axs[1].grid(True)
    # ^^^^^^ r_z_zhat_t and p_z_zhat_t ^^^^^^ #

    # RMSE
    lon = cross._dsz.lon
    lat = cross._dsz.lat
    time = cross._dsz.time
    nlon, nlat, nt = len(lon), len(lat), len(time)
    zhat = cross.zhat_accumulated_modes[-1, :]  # space x time
    zdata = cross._dsz.data  # space x time

    rmse_map = np.sqrt(np.nansum((zhat - zdata)**2, axis=1) / nt).reshape((nlat, nlon))

    im = _plot_map(
        rmse_map, cross._dsz.lat, cross._dsz.lon, fig, axs[2],
        'RMSE map',
        cmap="Reds",
        ticks=None,
        levels=None,
        xlim=z_xlim,
        colorbar=False,
        add_cyclic_point=cross.dsz.region.lon0 >= cross.dsz.region.lonf,
        plot_type=plot_type,
    )
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[3, 0:3]), orientation='horizontal')
    # ^^^^^^ r_z_zhat_s and p_z_zhat_s ^^^^^^ #
    
    # RMSE time series
    rmse_ts = np.sqrt(np.nansum((zhat - zdata)**2, axis=0) / (nlat * nlon))
    axs[3].bar(cross._dsz.time.values, rmse_ts, color="orange")
    axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    axs[3].set_title('RMSE time series')
    axs[3].grid(True)

    '''
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
    '''

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
