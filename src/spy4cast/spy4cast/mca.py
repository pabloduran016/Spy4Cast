import warnings
import os
import math
from typing import Optional, Tuple, Any, Sequence, Union, cast, Literal, List

import numpy as np
import numpy.typing as npt  # type: ignore
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec  # type: ignore
from matplotlib import ticker, patches
import cartopy.crs as ccrs
from scipy import sparse, signal
from scipy.signal.signaltools import axis_reverse
import scipy.sparse.linalg
import xarray as xr
from scipy.stats import stats

from .. import Region
from .._functions import time_from_here, time_to_here, region2str, _debuginfo, debugprint
from .._procedure import _Procedure, plot_map, _apply_flags_to_fig, _calculate_figsize, MAX_HEIGHT, MAX_WIDTH, plot_ts, \
    get_xlim_from_region, get_central_longitude_from_region, add_cyclic_point_to_data
from .preprocess import Preprocess


__all__ = [
    'MCA',
    'index_regression',
    'calculate_psi',
]

from ..land_array import LandArray


class MCA(_Procedure):
    """Maximum covariance analysis between y (predictor) and Z (predictand)

    Parameters
    ----------
        dsy : Preprocess
            Predictor
        dsz : Preprocess
            Predictand
        nm : int
            Number of modes
        alpha : float
            Significance level
        sig : {'monte-carlo', 'test-t'}
            Signification technique: monte-carlo or test-t
        montecarlo_iterations : optional, int
            Number of iterations for monte-carlo sig
        detrend : bool, default=True
            Detrend the y variable in the time axis
        num_svdvals : int or None, default=None
            If not None, approximate the sum of the singular values of the
            covariance matrix (used to calculate scf) to the sum of the
            largest `num_svdvals` singular values. Useful to speed up MCA
            if the `scf` is not needed or it is not needed to be precise.

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

    All the :doc:`/variables/mca` easily accesioble

    >>> y_regression = mca.RUY.reshape((len(y.lat), len(y.lon), 6))  # 6 is the number of modes
    >>> # Plot with any plotting library
    >>> import matplotlib.pyplot as plt
    >>> import cartopy.crs as ccrs
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection=ccrs.PlateCarree())
    >>> ax.contourf(y.lon, y.lat, y_regression[:, :, 0])
    >>> ax.coastlines()

    Save the data in .npy to use in a different run

    >>> mca.save("saved_mca_", folder="saved_data")

    Reuse the previuosly ran data easily with one line

    >>> mca = MCA.load("saved_mca_", folder="saved_data", dsy=y, dsz=z)  # IMPORTANT TO USE dsy= and dsz=

    Plot with one line and several options

    >>> mca.plot(show_plot=True, halt_program=True, cmap="jet", figsize=(20, 10))


    Attributes
    ----------
    psi
        Regression coefficient of the MCA model, so that ZHAT = PSI * Y
    RUY 
        Correlation of the predictor field. Dimension: y_space x nm
    RUY_sig
        Correlation of the predictor field where pvalue is smaller than alpha. Dimension: y_space x nm
    SUY
        Regression in space of the predictor with the singular vector. Dimension: y_space x nm
    SUY_sig
        Regression in space of the predictor with the singular vector where pvalue is smaller than alpha. Dimension: y_space x nm
    RUZ
        Correlation of the predictand field. Dimension: z_space x nm
    RUZ_sig 
        Correlation of the predictand field where pvalue is smaller than alpha. Dimension: z_space x nm
    SUZ 
        Regression in space of the predictand with the singular vector. Dimension: z_space x nm
    SUZ_sig 
        Regression in space of the predictand with the singular vector where pvalue is smaller than alpha. Dimension: z_space x nm
    pvalruy
        Pvalue of the correlation of the predictor field. Dimension: y_space x nm
    pvalruz
        Pvalue of the correlation of the predictand field. Dimension: z_space x nm
    Us 
        Singular vectors of the predictor field. Dimension: nm x time
    Vs
        Singular vectors of the predictand field. Dimension: nm x time
    scf
        Square covariance fraction of the singular values. Dimension: nm
    alpha
        Significance coeficient.
    """
    # TODO: Document MCA fields
    RUY: npt.NDArray[np.float32]
    RUY_sig: npt.NDArray[np.float32]
    SUY: npt.NDArray[np.float32]
    SUY_sig: npt.NDArray[np.float32]
    RUZ: npt.NDArray[np.float32]
    RUZ_sig: npt.NDArray[np.float32]
    SUZ: npt.NDArray[np.float32]
    SUZ_sig: npt.NDArray[np.float32]
    pvalruz: npt.NDArray[np.float32]
    pvalruy: npt.NDArray[np.float32]
    Us: npt.NDArray[np.float32]
    Vs: npt.NDArray[np.float32]
    scf: npt.NDArray[np.float32]
    alpha: float

    _psi: Optional[npt.NDArray[np.float32]]

    @property
    def var_names(self) -> Tuple[str, ...]:
        """Returns the variables contained in the object (RUY, SUY, scf, ...)"""
        return (
            'psi',
            'RUY',
            'RUY_sig',
            'SUY',
            'SUY_sig',
            'RUZ',
            'RUZ_sig',
            'SUZ',
            'SUZ_sig',
            'pvalruz',
            'pvalruy',
            'Us',
            'Vs',
            'scf',
            'alpha',
            'nm',
        )

    def __init__(
        self,
        dsy: Preprocess,
        dsz: Preprocess,
        nm: int,
        alpha: float,
        sig: Literal["test-t", "monte-carlo"] = 'test-t',
        montecarlo_iterations: Optional[int] = None,
        detrend: bool = True,
        num_svdvals: Optional[int] = None,
    ):
        self._dsz = dsz
        self._dsy = dsy

        _debuginfo(f"""Applying MCA 
    Shapes: Z{dsz.shape} 
            Y{dsy.shape} 
    Regions: Z {region2str(self._dsz.region)} 
            Y {region2str(self._dsy.region)}""", )
        here = time_from_here()

        if len(dsz.time) != len(dsy.time):
            raise ValueError(
                f'The number of years of the predictand must be the '
                f'same as the number of years of the predictor: '
                f'got {len(dsz.time)} and '
                f'{len(dsy.time)}'
            )

        self._mca(dsz.land_data, dsy.land_data, nm, alpha, sig, montecarlo_iterations, detrend, num_svdvals)
        debugprint(f'       Took: {time_to_here(here):.03f} seconds')

        self._psi = None
        # first you calculate the covariance matrix
        # c = np.nan_to_num(np.dot(y, np.transpose(z)), nan=NAN_VAL)

    @property
    def dsy(self) -> Preprocess:
        """Preprocessed dataset introduced as predictor"""
        return self._dsy

    @property
    def dsz(self) -> Preprocess:
        """Preprocessed dataset introduced as predictand"""
        return self._dsz

    @classmethod
    def from_land_arrays(
        cls,
        y: LandArray,
        z: LandArray,
        nm: int,
        alpha: float,
        sig: Literal["test-t", "monte-carlo"] = 'test-t',
        montecarlo_iterations: Optional[int] = None,
        detrend: bool = True,
        num_svdvals: Optional[int] = None,
    ) -> 'MCA':
        """
        Alternative constructor for mca that takes Land Array

        Parameters
        ----------
            y : LandArray
                Predictor (space x time)
            z : LandArray
                Predictand (space x time)
            nm : int
                Number of modes
            alpha : alpha
               Significance level
            sig : 'monte-carlo or 'test-t'
                Signification technique: monte-carlo or test-t
            montecarlo_iterations : optional, int
                Number of iterations for monte-carlo sig
            detrend : bool, default=True
                Detrend the y variable in the time axis
            num_svdvals : int or None, default=None
                If True, approximate the sum of the singular values of the
                covariance matrix (used to calculate scf) to the sum of the
                largest `num_svdvals` singular values. Useful to speed up MCA
                if the `scf` is not needed or it is not needed to be precise.

        Returns
        -------
            MCA
                MCA object with the methodology performed

        See Also
        --------
            MCA
        """
        m = cls.__new__(MCA)
        m._mca(z, y, nm, alpha, sig, montecarlo_iterations, detrend, num_svdvals)
        return m

    def _mca(
        self,
        z: LandArray,
        y: LandArray,
        nm: int,
        alpha: float,
        sig: str,
        montecarlo_iterations: Optional[int] = None,
        detrend: bool = True,
        num_svdvals: Optional[int] = None
    ) -> None:
        nz, nt = z.shape
        ny, nt = y.shape

        if detrend:
            y.values[~y.land_mask] = signal.detrend(y.not_land_values)  # detrend in time

        c = np.dot(y.not_land_values, np.transpose(z.not_land_values))
        if type(c) == np.ma.MaskedArray:
            c = c.data

        r, d, q = sparse.linalg.svds(c, k=nm, which='LM')  # Which LM = Large magnitude
        # Modes are reversed so we reverse them in r, d and q
        r = r[:, ::-1]
        d = d[::-1]
        q = q[::-1, :]

        # OLD WAY OF DOING SVD: REALLY SLOW
        # r, d, q = scipy.linalg.svd(c)
        # r = r[:, :nm]
        # q = r[:nm, :]

        if num_svdvals is None:
            svdvals = scipy.linalg.svdvals(c)
            sum_svdvals = np.sum(svdvals)
        elif num_svdvals <= nm:
            sum_svdvals = np.sum(d[:num_svdvals])
        else:
            _r, svdvals, _q = sparse.linalg.svds(c, k=num_svdvals, which='LM')
            sum_svdvals = np.sum(svdvals)

        scf = d / sum_svdvals

        # y había que transponerla si originariamente era (espacio, tiempo),
        # pero ATN_e es (tiempo, espacio) así
        # que no se transpone
        u = np.dot(np.transpose(y.not_land_values), r)
        # u = np.dot(np.transpose(y), r[:, :nm])
        # calculamos las anomalías estandarizadas
        v = np.dot(np.transpose(z.not_land_values), q.transpose())
        # v = np.dot(np.transpose(z), q[:, :nm])

        self.RUY = np.zeros([ny, nm], dtype=np.float32)
        self.RUY_sig = np.zeros([ny, nm], dtype=np.float32)
        self.SUY = np.zeros([ny, nm], dtype=np.float32)
        self.SUY_sig = np.zeros([ny, nm], dtype=np.float32)
        self.RUZ = np.zeros([nz, nm], dtype=np.float32)
        self.RUZ_sig = np.zeros([nz, nm], dtype=np.float32)
        self.SUZ = np.zeros([nz, nm], dtype=np.float32)
        self.SUZ_sig = np.zeros([nz, nm], dtype=np.float32)
        self.Us = ((u - u.mean(0)) / u.std(0)).transpose()  # Standarized anom across axis 0
        self.Vs = ((v - v.mean(0)) / v.std(0)).transpose()
        self.scf = scf
        self.alpha = alpha
        self.nm = nm

        self.pvalruy = np.zeros([ny, nm], dtype=np.float32)
        self.pvalruz = np.zeros([nz, nm], dtype=np.float32)
        for i in range(nm):
            (
                self.RUY[:, i],
                self.pvalruy[:, i],
                self.RUY_sig[:, i],
                self.SUY[:, i],
                self.SUY_sig[:, i]
            ) = index_regression(y, self.Us[i, :], alpha, sig, montecarlo_iterations)

            (
                self.RUZ[:, i],
                self.pvalruz[:, i],
                self.RUZ_sig[:, i],
                self.SUZ[:, i],
                self.SUZ_sig[:, i]
            ) = index_regression(z, self.Us[i, :], alpha, sig, montecarlo_iterations)

    @property
    def psi(self) -> npt.NDArray[np.float32]:
        if self._psi is None:
            z = self._dsz.land_data
            y = self._dsy.land_data
            nz, nt = z.shape
            ny, nt = y.shape
            self._psi = calculate_psi(self.SUY, self.Us, z.values, nt, ny, self.nm, self.scf)
        return self._psi

    @psi.setter
    def psi(self, value: npt.NDArray[np.float32]) -> None:
        self._psi = value

    def plot(
        self,
        *,
        save_fig: bool = False,
        show_plot: bool = False,
        halt_program: bool = False,
        cmap: str = 'bwr',
        signs: Optional[Sequence[bool]] = None,
        folder: Optional[str] = None,
        name: Optional[str] = None,
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
        nm: Optional[int] = None,
        map_y: Optional[Preprocess] = None,
        map_z: Optional[Preprocess] = None,
        rect_color: Union[Tuple[int, int, int], str] = "r",
        sig: Optional[Literal["monte-carlo", "test-t"]] = None,
        montecarlo_iterations: Optional[int] = None,
        plot_type: Literal["contour", "pcolor"] = "contour",
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        central_longitude_y: Optional[float] = None,
        central_longitude_z: Optional[float] = None,
        y_xlim: Optional[Tuple[float, float]] = None,
        z_xlim: Optional[Tuple[float, float]] = None,
        variable: Literal["s", "r"] = "r",
        ruy_ticks: Optional[Union[npt.NDArray[np.float32], Sequence[float]]] = None,
        ruz_ticks: Optional[Union[npt.NDArray[np.float32], Sequence[float]]] = None,
        ruy_levels: Optional[Union[npt.NDArray[np.float32], Sequence[float], bool]] = None,
        ruz_levels: Optional[Union[npt.NDArray[np.float32], Sequence[float], bool]] = None,
    ) -> Tuple[Tuple[plt.Figure, ...], Tuple[plt.Axes, ...]]:
        """Plot the MCA results

        Parameters
        ----------
        save_fig
            Saves the fig using `folder` and `name` parameters
        show_plot
            Shows the plot but does NOT stop the program. Calls `fig.show`. 
            If you want the behaviour of `plt.plot` add the halt_program option.
        halt_program
            Only used if `show_plot` is `True`. If `True` shows the plot if plt.show
            and stops execution. Else uses `fig.show` and does not halt program
        cmap
            Colormap for the predicting maps
        signs
            Sequence of `True` or `False` values of same length as `nm`. Where `True` the
            mode output will be multipled by -1.
        folder
            Directory to save fig if `save_fig` is `True`
        name
            Name of the fig saved if `save_fig` is `True`
        y_ticks
            Ticks for the maps of the Y output
        z_ticks
            Ticks for the maps of the Z output
        y_levels
            Levels for the maps of the Y output
        z_levels
            Levels for the maps of the Z output
        figsize
            Set figure size. See `plt.figure`
        nm : int
            Number of modes to plot
        map_y : Preprocess
            Y to plot the map
        map_z : Preprocess
            Z to plot the map
        rect_color: (r, g, b) or string, default = "r"
            Color of the rectangle when using map_y and map_z that outlines the region where the methodology was originally ran
        sig : {"monte-carlo", "test-t"}
            Significance method when map_y or map_z is set
        montecarlo_iterations : int
            when monte-carlo sig and map_y or map_z is set
        plot_type : {"contour", "pcolor"}, defaut = "pcolor"
            Plot type. If `contour` it will use function `ax.contourf`, 
            if `pcolor` `ax.pcolormesh`.
        height_ratios: list[float], optional
            Height ratios passed in to matplotlib.gridspec.Gridspec
        width_ratios: list[float], optional
            Width ratios passed in to matplotlib.gridspec.Gridspec
        central_longitude_y : float, optional
            Longitude used to center the `y` map
        central_longitude_z : float, optional
            Longitude used to center the `z` map
        y_xlim : tuple[float, float], optional
            Xlim for the `y` map passed into ax.set_extent
        z_xlim : tuple[float, float], optional
            Xlim for the `z` map passed into ax.set_extent
        variable : {"s", "r"}, default="r"
            If "r", plot RUY and RUZ (correlation)
            If "s", plot SUY and SUZ (regression)

        Note
        ----
            `ruy_ticks`, `ruz_ticks`, `ruy_levels` and `ruz_levels` are deprecated

        Returns
        -------
        figures : Tuple[plt.Figure]
            Figures objects from matplotlib. One figure per page of MCA with 3 modes per page

        ax : Tuple[plt.Axes]
            Tuple of axes in figure. In this case 3 axes per mode: Us/Vs, Y, Z

        Examples
        -------

        Plot and halt the program

        >>> mca.plot(show_plot=True, halt_program=True)

        Save the plot 

        >>> mca.plot(save_fig=True, name="mca_plot.png")

        Plot with pcolormesh and be precise with the resolution

        >>> mca.plot(save_fig=True, name="mca_plot.png", plot_type="pcolor")

        Plot MCA result in a bigger region

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
        >>> mca.plot(show_plot=True, halt_program=True, map_y=map_y, map_z=map_z)


        Plot and not halt the program

        >>> mca.plot(show_plot=True)
        >>> # .... Compute crossvalidation for example
        >>> import matplotlib.pyplot as plt
        >>> plt.show()  # Will show the previously ran plot
        """
        nm = self.nm if nm is None else nm
        if nm > self.nm:
            raise ValueError(f"Can not draw more modes ({nm}) than the ones used for then methodology ({self.nm})")
        if signs is not None:
            if len(signs) != nm:
                raise TypeError(f'Expected signs to be a sequence of the same length as number of modes ({nm})')
            # if any(type(x) != bool for x in signs):
            #     raise TypeError(f'Expected signs to be a sequence of boolean: either True or False, got {signs}')

        if montecarlo_iterations is not None and map_y is None and map_z is None and sig != 'monte-carlo':
            assert False
        if sig is not None and map_y is None and map_z is None:
            assert False
        sig = 'test-t' if sig is None else sig

        if plot_type not in ("contour", "pcolor"):
            raise ValueError(f"Expected `contour` or `pcolor` for argument `plot_type`, but got {plot_type}")

        if variable not in ("r", "s"):
            raise ValueError(f"Expected `r` or `s` for argument `variable`, but got {variable}")

        if ruy_ticks is not None:
            if y_ticks is not None:
                raise ValueError(f"Supplied both `y_ticks` and `ruy_ticks`, but only one can be used. Please use `y_ticks`, "
                                 "as `y_ticks` is deprecated")
            y_ticks = ruy_ticks
            warnings.warn(f"`ruy_ticks` is deprecated. Please use `y_ticks` instead")
        if ruz_ticks is not None:
            if z_ticks is not None:
                raise ValueError(f"Supplied both `z_ticks` and `ruz_ticks`, but only one can be used. Please use `z_ticks`, "
                                 "as `z_ticks` is deprecated")
            z_ticks = ruz_ticks
            warnings.warn(f"`ruz_ticks` is deprecated. Please use `z_ticks` instead")
        if ruy_levels is not None:
            if y_levels is not None:
                raise ValueError(f"Supplied both `y_levels` and `ruy_levels`, but only one can be used. Please use `y_levels`, "
                                 "as `y_levels` is deprecated")
            y_levels = ruy_levels
            warnings.warn(f"`ruy_levels` is deprecated. Please use `y_levels` instead")
        if ruz_levels is not None:
            if z_levels is not None:
                raise ValueError(f"Supplied both `z_levels` and `ruz_levels`, but only one can be used. Please use `z_levels`, "
                                 "as `z_levels` is deprecated")
            z_levels = ruz_levels
            warnings.warn(f"`ruz_levels` is deprecated. Please use `z_levels` instead")

        if map_y is None:
            map_y = self.dsy
            if variable == "r":
                uy, uy_sig = self.RUY, self.RUY_sig
            elif variable == "s":
                uy, uy_sig = self.SUY, self.SUY_sig
            else:
                assert False, "Unreachable"
        else:
            if map_y.time.shape[0] != self.dsy.time.shape[0]:
                raise ValueError(f"`map_y` has to have the same amount of years (length of the time dimension) as the "
                                 f"data used to run MCA. Got {map_y.time.shape[0]}, expected {self.dsy.time.shape[0]}")
            ny = map_y.shape[0]
            uy = np.zeros([ny, nm], dtype=np.float32)
            uy_sig = np.zeros([ny, nm], dtype=np.float32)
            for i in range(nm):
                ruy, _, ruy_sig, suy, suy_sig = index_regression(map_y.land_data, self.Us[i, :], self.alpha, sig, montecarlo_iterations)
                uy[:, i] = ruy if variable == "r" else suy
                uy_sig[:, i] = ruy_sig if variable == "r" else suy_sig
        if map_z is None:
            map_z = self.dsz
            if variable == "r":
                uz, uz_sig = self.RUZ, self.RUZ_sig
            elif variable == "s":
                uz, uz_sig = self.SUZ, self.SUZ_sig
            else:
                assert False, "Unreachable"
        else:
            if map_z.time.shape[0] != self.dsz.time.shape[0]:
                raise ValueError(f"`map_z` has to have the same amount of years (length of the time dimension) as the "
                                 f"data used to run MCA. Got {map_z.time.shape[0]}, expected {self.dsz.time.shape[0]}")
            nz = map_z.shape[0]
            uz = np.zeros([nz, nm], dtype=np.float32)
            uz_sig = np.zeros([nz, nm], dtype=np.float32)
            for i in range(nm):
                ruz, _, ruz_sig, suz, suz_sig = index_regression(map_z.land_data, self.Us[i, :], self.alpha, sig, montecarlo_iterations)
                uz[:, i] = ruz if variable == "r" else suz
                uz_sig[:, i] = ruz_sig if variable == "r" else suz_sig

        figs = []
        axs: List[Tuple[plt.Axes, ...]] = []
        n_pages = math.ceil(nm / 3)
        for i in range(n_pages):
            mode0 = 3 * i
            modef = min(nm - 1, 3 * (i + 1) - 1)
            fig_i, axs_i = _new_mca_page(
                self, cmap=cmap, signs=signs, y_ticks=y_ticks, z_ticks=z_ticks, 
                y_levels=y_levels, z_levels=z_levels, figsize=figsize, mode0=mode0, modef=modef,
                uy=uy, uy_sig=uy_sig, uz=uz, uz_sig=uz_sig, variable=variable,
                map_y=map_y, map_z=map_z, plot_type=plot_type, 
                rect_color=rect_color, height_ratios=height_ratios, width_ratios=width_ratios,
                central_longitude_y=central_longitude_y, y_xlim=y_xlim, central_longitude_z=central_longitude_z, z_xlim=z_xlim
            )

            if folder is None:
                folder = '.'
            if name is None:
                path = os.path.join(folder, f'mca-plot_z-{self._dsz.var}_y-{self._dsy.var}.png')
            else:
                path = os.path.join(folder, name)

            if n_pages > 1:
                path_name, path_extension = os.path.splitext(path)
                path = f"{path_name}_{i}{path_extension}"

            _apply_flags_to_fig(
                fig_i, path,
                save_fig=save_fig,
                show_plot=show_plot,
                halt_program=False,
            )

            figs.append(fig_i)
            axs.extend(axs_i)

        if show_plot and halt_program:
            plt.show(block=True)

        return tuple(figs), tuple(axs)


    @classmethod
    def load(cls, prefix: str, folder: str = '.', *,
             dsy: Optional[Preprocess] = None,
             dsz: Optional[Preprocess] = None,
             **attrs: Any) -> 'MCA':
        """Load an MCA object from .npy files saved in MCA.save.

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
            MCA

        Examples
        --------
        Load with just one line

        >>> mca = MCA.load(prefix="mca_", folder="saved_data")

        Save: on a previous run the MCA is calcuated
        
        >>> from spy4cast import Dataset, Region, Month
        >>> from spy4cast.spy4cast import MCA, Preprocess
        >>> y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 2010)))
        >>> z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 2011)))
        >>> mca = MCA(y, z, 3, 0.01)
        >>> mca.save("saved_mca_", folder="data")  # Save the output

        Load: To avoid running the methodology again for plotting and analysis load the data directly

        >>> from spy4cast.spy4cast import MCA, Preprocess
        >>> from spy4cast import Dataset, Region, Month
        >>> y = Preprocess(Dataset("dataset_y.nc").open("y").slice(
        ...         Region(-50, 10, -50, 20, Month.JUN, Month.AUG, 1960, 2010)))  # YOU SHOULD USE THE SAME REGION
        >>> z = Preprocess(Dataset("dataset_z.nc").open("z").slice(
        ...         Region(-30, 30, -120, 120, Month.DEC, Month.FEB, 1961, 2011)))
        >>> mca = MCA.load("saved_mca_", folder="data", dsy=y, dsz=z)  # IMPORTANT TO USE dsy= and dsz=

        Then you can plot as usual

        >>> mca.plot(save_fig=True, name="mca.png")

        """
        if len(attrs) != 0:
            raise TypeError('Load only takes two keyword arguments: dsy and dsz')
        if dsz is None or dsy is None:
            raise TypeError('To load an MCA object you must provide `dsz` and `dsy` keyword arguments')
        if type(dsz) != Preprocess or type(dsy) != Preprocess:
            raise TypeError(f'Unexpected types ({type(dsz)} and {type(dsy)}) for `dsz` and `dsy`. Expected type `Preprocess`')

        self: MCA = super().load(prefix, folder)

        self._dsz = dsz
        self._dsy = dsy
        return self


def _new_mca_page(
    mca: MCA, 
    cmap: str,
    signs: Optional[Sequence[bool]],
    y_ticks: Optional[
        Union[npt.NDArray[np.float32], Sequence[float]]
    ],
    z_ticks: Optional[
        Union[npt.NDArray[np.float32], Sequence[float]]
    ],
    y_levels: Optional[
        Union[npt.NDArray[np.float32], Sequence[float], bool]
    ],
    z_levels: Optional[
        Union[npt.NDArray[np.float32], Sequence[float], bool]
    ],
    figsize: Optional[Tuple[float, float]],
    mode0: int,
    modef: int,
    uy: npt.NDArray[np.float_],
    uy_sig: npt.NDArray[np.float_],
    uz: npt.NDArray[np.float_],
    uz_sig: npt.NDArray[np.float_],
    variable: Literal["r", "s"],
    map_y: Preprocess,
    map_z: Preprocess,
    rect_color: Union[Tuple[int, int, int], str],
    plot_type: Literal["contour", "pcolor"],
    width_ratios: Optional[List[float]],
    height_ratios: Optional[List[float]],
    central_longitude_y: Optional[float],
    central_longitude_z: Optional[float],
    y_xlim: Optional[Tuple[float, float]],
    z_xlim: Optional[Tuple[float, float]],
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    nm = modef - mode0 + 1

    y_wratio = (map_y.region.lonf if map_y.region.lonf > map_y.region.lon0 else map_y.region.lonf + 360) - map_y.region.lon0
    z_wratio = (map_z.region.lonf if map_z.region.lonf > map_z.region.lon0 else map_z.region.lonf + 360) - map_z.region.lon0
    gs = gridspec.GridSpec(nm + 1, 3, 
                           height_ratios=(height_ratios if height_ratios is not None else [*[1]*nm, 0.15]),
                           width_ratios=(width_ratios if width_ratios is not None else 
                                         [0.9*max(y_wratio, z_wratio), y_wratio, z_wratio]),
                           hspace=0.7)

    figsize = _calculate_figsize((nm + 1)/2/3, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT) if figsize is None else figsize
    fig: plt.Figure = plt.figure(figsize=figsize)

    # central longitude
    central_longitude_y = central_longitude_y if central_longitude_y is not None else \
        get_central_longitude_from_region(map_y.region.lon0, map_y.region.lonf)
    y_xlim = y_xlim if y_xlim is not None else \
        get_xlim_from_region(map_y.region.lon0, map_y.region.lonf, central_longitude_y)

    central_longitude_z = central_longitude_z if central_longitude_z is not None else \
        get_central_longitude_from_region(map_z.region.lon0, map_z.region.lonf)
    z_xlim = z_xlim if z_xlim is not None else \
        get_xlim_from_region(map_z.region.lon0, map_z.region.lonf, central_longitude_z)

    axs = (
        *(fig.add_subplot(gs[i, 0]) for i in range(nm)),
        *(fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree(central_longitude_y)) for i in range(nm)),
        *(fig.add_subplot(gs[i, 2], projection=ccrs.PlateCarree(central_longitude_z)) for i in range(nm)),
    )

    for i in range(nm):
        mode = mode0 + i

        # Us and Vs Time series
        ax_ts = axs[i]

        us = mca.Us[mode, :]
        vs = mca.Vs[mode, :]
        if signs is not None and signs[mode]:
            us *= -1
            vs *= -1
        plot_ts(
            time=mca._dsy.time.values,
            arr=us,
            ax=ax_ts,
            title=f'Us Vs mode {mode + 1}',
            color='green',
            label='Us',
        )
        plot_ts(
            time=mca._dsz.time.values,
            arr=vs,
            ax=ax_ts,
            title=None,
            color='blue',
            label='Vs'
        )
        ax_ts.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax_ts.set_xlim(xmin=min(mca._dsy.time.values[0], mca._dsz.time.values[0]), 
                       xmax=max(mca._dsy.time.values[-1], mca._dsz.time.values[-1]))
        if i == 0:
            ax_ts.legend()
        ax_ts.grid(True)

        # UY and UZ map
        for j, (var_name, u, u_sig, lats, lons, cm, ticks, levels, region, add_cyclic_point, original_region, central_longitude, xlim) in enumerate((
            ("RUY" if variable == "r" else "SUY", uy, uy_sig, map_y.lat, map_y.lon, 'bwr', y_ticks, y_levels, map_y.region, map_y.region.lon0 >= map_y.region.lonf, mca.dsy.region, central_longitude_y, y_xlim),
            ("RUZ" if variable == "r" else "SUZ", uz, uz_sig, map_z.lat, map_z.lon, cmap, z_ticks, z_levels, map_z.region, map_z.region.lon0 >= map_z.region.lonf, mca.dsz.region, central_longitude_z, z_xlim)
        )):
            ylim = sorted((lats.values[-1], lats.values[0]))

            if levels is None:
                levels = np.linspace(-1, +1, 20)

            ax_map = axs[nm * (j + 1) + i]
            title = f'{var_name} mode {mode + 1}. ' \
                    f'SCF={mca.scf[mode]*100:.01f}%'

            t = u[:, mode].transpose().reshape((len(lats), len(lons)))
            th = u_sig[:, mode].transpose().reshape((len(lats), len(lons)))

            if signs is not None:
                if signs[mode]:
                    t *= -1

            im = plot_map(
                t, lats, lons, fig, ax_map, title,
                levels=levels, xlim=xlim, ylim=ylim, cmap=cm, ticks=ticks,
                colorbar=False, add_cyclic_point=add_cyclic_point, plot_type=plot_type,
            )
            if i == nm - 1:
                cb = fig.colorbar(im, cax=fig.add_subplot(gs[nm, j + 1]), orientation='horizontal', ticks=ticks,)
                if ticks is None:
                    tick_locator = ticker.MaxNLocator(nbins=5, prune='both', symmetric=True)
                    cb.ax.xaxis.set_major_locator(tick_locator)
                #cb.ax.xaxis.set_tick_params(rotation=20)
            hlons = lons
            if add_cyclic_point:
                th, hlons = add_cyclic_point_to_data(th, coord=hlons.values)
            ax_map.contourf(
                hlons, lats, th, colors='none', hatches='..', extend='both',
                transform=ccrs.PlateCarree()
            )
            width = original_region.lonf - original_region.lon0 if original_region.lonf > original_region.lon0 else \
                    original_region.lonf + 360 - original_region.lon0 
            assert width >= 0
            height = original_region.latf - original_region.lat0 if original_region.latf > original_region.lat0 else \
                    original_region.latf + 360 - original_region.lat0 
            assert height >= 0
            rect = patches.Rectangle(
                xy=[original_region.lon0, original_region.lat0],
                width=width,
                height=height,
                facecolor='none', edgecolor=rect_color,
                transform=ccrs.PlateCarree())
            ax_map.add_patch(rect)

    axs[0].legend(loc='upper left')

    fig.suptitle(
        f'Z({mca._dsz.var}): {region2str(mca._dsz.region)}, '
        f'Y({mca._dsy.var}): {region2str(mca._dsy.region)}. '
        f'Alpha: {mca.alpha}',
        fontweight='bold'
    )

    fig.subplots_adjust(hspace=.4)

    return fig, axs


def index_regression(
    data: Union[LandArray, npt.NDArray[np.float_]],
    index: npt.NDArray[np.float32],
    alpha: float,
    sig: str,
    montecarlo_iterations: Optional[int] = None
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32]
]:
    """Create correlation (pearson correlation) and regression

    Parameters
    ----------
        data : npt.NDArray[np.float32]
            Data to perform the methodology in (space x time)
        index : npt.NDArray[np.float32]
            Unidimensional array: temporal series
        alpha : float
            Significance level
        sig : 'monte-carlo' or 'test-t'
            Type of sygnificance
        montecarlo_iterations : optional, int
            Number of iterations for monte-carlo sig

    Returns
    -------
        Cor : npt.NDArray[np.float32] (space)
            Correlation map
        Pvalue : npt.NDArray[np.float32] (space)
            Map with p values
        Cor_sig : npt.NDArray[np.float32] (space)
            Significative corrrelation map
        reg : npt.NDArray[np.float32] (space)
            Regression map
        reg_sig : npt.NDArray[np.float32] (space)
            Significative regression map
    """
    if len(data.shape) == 2:
        ns, nt = data.shape
    elif len(data.shape) == 1:
        ns, nt = 1, data.shape[0]
    else:
        assert False, "Unreachable"
    cor = np.zeros(ns, dtype=np.float32)
    pvalue = np.zeros(ns, dtype=np.float32)
    if type(data) == LandArray:
        reg = data.values.dot(index) / nt
        cor[data.land_mask] = np.nan
        pvalue[data.land_mask] = np.nan
        not_land_values = data.not_land_values
        land_mask = data.land_mask
    else:
        assert type(data) == np.ndarray
        reg = np.array([np.dot(data, index) / nt])
        not_land_values = data
        land_mask = np.zeros(1, dtype=np.bool_)


    if sig == 'test-t':
        if len(data.shape) == 2:
            result = np.apply_along_axis(stats.pearsonr, 1, not_land_values, index)
            cor[~land_mask] = result[:, 0]
            pvalue[~land_mask] = result[:, 1]
        elif len(data.shape) == 1:
            result = stats.pearsonr(not_land_values, index)
            cor[~land_mask] = result[0]
            pvalue[~land_mask] = result[1]
        else:
            assert False, "Unreachable"

        cor_sig = cor.copy()
        reg_sig = reg.copy()
        cor_sig[pvalue > alpha] = np.nan
        reg_sig[pvalue > alpha] = np.nan
    elif sig == 'monte-carlo':
        if montecarlo_iterations is None:
            raise ValueError('Expected argument `montecarlo_iteration` for `monte-carlo` sig')

        corp = np.empty([ns, montecarlo_iterations])
        for p in range(montecarlo_iterations):
            if len(data.shape) == 2:
                corp[~land_mask, p] = pearsonr_2d(not_land_values, np.random.permutation(index))
            elif len(data.shape) == 1:
                corp[~land_mask, p] = stats.pearsonr(not_land_values, np.random.permutation(index))[0]
            else:
                assert False, "Unreachable"

        if len(data.shape) == 2:
            result = np.apply_along_axis(stats.pearsonr, 1, not_land_values, index)
            cor[~land_mask] = result[:, 0]
        elif len(data.shape) == 1:
            result = stats.pearsonr(not_land_values, index)
            cor[~land_mask] = result[0]
        else:
            assert False, "Unreachable"

        for nn in range(ns):
            hcor = np.count_nonzero((cor[nn] > 0) & (corp[nn, :] < cor[nn]) | (cor[nn] < 0) & (corp[nn, :] > cor[nn]))
            # nivel de confianza
            pvalue[nn] = 1 - hcor / montecarlo_iterations

        cor_sig = cor.copy()
        reg_sig = reg.copy()
        cor_sig[pvalue > alpha] = np.nan
        reg_sig[pvalue > alpha] = np.nan
    else:
        raise ValueError(f'Unrecognized `sig` parameter: {sig}')

    return cor, pvalue, cor_sig, reg, reg_sig


def pearsonr_2d(y: npt.NDArray[np.float_], x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:, None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:, None], 2), axis=1))
    rho: npt.NDArray[np.float_] = upper / lower
    return rho


def calculate_psi(
    suy: npt.NDArray[np.float32],
    us: npt.NDArray[np.float32],
    z: npt.NDArray[np.float32],
    nt: int,
    ny: int,
    nm: int,
    scf: npt.NDArray[np.float_],
) -> npt.NDArray[np.float32]:
    # (((SUY * inv(Us * Us')) * Us) * Z') * nt * nm / ny
    # suy = suy * scf[np.newaxis, :]
    return cast(
        npt.NDArray[np.float32],
        np.dot(np.dot(np.dot(suy, np.linalg.inv(np.dot(us, np.transpose(us)))), us), np.transpose(z)) * nt / ny)
    # (((SUY * inv(Us * Us')) * Us) * Z') / (ny * nm**2)
    # return cast(
    #     npt.NDArray[np.float32],
    #     np.dot(np.dot(np.dot(suy, np.linalg.inv(np.dot(us, np.transpose(us)))), us), np.transpose(z)) * nm / ny) 

