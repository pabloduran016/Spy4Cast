import os
from typing import Optional, Tuple, Any, Sequence, Union, cast, Literal

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from scipy import sparse, signal
import scipy.sparse.linalg
import xarray as xr
from scipy.stats import stats

from .. import Region
from .._functions import time_from_here, time_to_here, region2str, _debuginfo, debugprint
from .._procedure import _Procedure, _plot_map, _apply_flags_to_fig, _calculate_figsize, MAX_HEIGHT, MAX_WIDTH, _plot_ts
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

    Attributes
    ----------
        psi : npt.NDArray[np.float32]
            Regression coefficient of the MCA model, so that ZHAT = PSI * Y
        RUY : npt.NDArray[np.float32]
            Regression of the predictor field. Dimension: y_space x nm
        RUY_sig : npt.NDArray[np.float32]
            Regression of the predictor field where pvalue is smaller than alpha. Dimension: y_space x nm
        SUY : npt.NDArray[np.float32]
            Correlation in space of the predictor with the singular vector. Dimension: y_space x nm
        SUY_sig : npt.NDArray[np.float32]
            Correlation in space of the predictor with the singular vector where pvalue is smaller than alpha. Dimension: y_space x nm
        RUZ : npt.NDArray[np.float32]
            Regression of the predictand field. Dimension: z_space x nm
        RUZ_sig : npt.NDArray[np.float32]
            Regression of the predictand field where pvalue is smaller than alpha. Dimension: z_space x nm
        SUZ : npt.NDArray[np.float32]
            Correlation in space of the predictand with the singular vector. Dimension: z_space x nm
        SUZ_sig : npt.NDArray[np.float32]
            Correlation in space of the predictand with the singular vector where pvalue is smaller than alpha. Dimension: z_space x nm
        pvalruy : npt.NDArray[np.float32]
            Pvalue of the correlation of the predictor field. Dimension: y_space x nm
        pvalruz : npt.NDArray[np.float32]
            Pvalue of the correlation of the predictand field. Dimension: z_space x nm
        Us : npt.NDArray[np.float32]
            Singular vectors of the predictor field. Dimension: nm x time
        Vs : npt.NDArray[np.float32]
            Singular vectors of the predictand field. Dimension: nm x time
        scf : npt.NDArray[np.float32]
            Square covariance fraction of the singular values. Dimension: nm
        alpha : float
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
    psi: npt.NDArray[np.float32]

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
        )

    def __init__(
        self,
        dsy: Preprocess,
        dsz: Preprocess,
        nm: int,
        alpha: float,
        sig: str = 'test-t',
        montecarlo_iterations: Optional[int] = None,
    ):
        self._dsz = dsz
        self._dsy = dsy

        _debuginfo(f"""Applying MCA 
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

        self._mca(dsz.land_data, dsy.land_data, nm, alpha, sig, montecarlo_iterations)
        debugprint(f'       Took: {time_to_here():.03f} seconds')

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

        Returns
        -------
            MCA
                MCA object with the methodology performed

        See Also
        --------
            MCA
        """
        m = cls.__new__(MCA)
        m._mca(z, y, nm, alpha, sig, montecarlo_iterations)
        return m

    def _mca(
        self,
        z: LandArray,
        y: LandArray,
        nm: int,
        alpha: float,
        sig: str,
        montecarlo_iterations: Optional[int] = None
    ) -> None:
        nz, nt = z.shape
        ny, nt = y.shape

        c = np.dot(signal.detrend(y.not_land_values), np.transpose(z.not_land_values))
        if type(c) == np.ma.MaskedArray:
            c = c.data

        # NEW WAY OF PEFORMING SVD: WAAAAAAAY FASTER (> 20 times)
        r, _d, q = sparse.linalg.svds(c, k=nm, which='LM')  # Which LM = Large magnitude
        # Modes are reversed so we reverse them in r, d and q
        r = r[:, ::-1]
        _d = _d[::-1]
        q = q[::-1, :]

        # OLD WAY OF DOING SVD: REALLY SLOW
        # r, d, q = scipy.linalg.svd(c)
        # r = r[:, :nm]
        # q = r[:nm, :]

        svdvals = scipy.linalg.svdvals(c)
        scf = svdvals[:nm] / np.sum(svdvals)

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
        self.psi = calculate_psi(self.SUY, self.Us, z.values, nm, ny)

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
        suy_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        suz_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        suy_levels: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        suz_levels: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot the MCA results

        Parameters
        ----------
        save_fig
            Saves the fig in with `folder` / `name` parameters
        show_plot
            Shows the plot
        halt_program
            Only used if `show_plot` is `True`. If `True` shows the plot if plt.show
            and stops execution. Else uses fig.show and does not halt program
        cmap
            Colormap for the predicting maps
        signs
            Sequence of `True` or `False` values of same length as `nm`. Where `True` the
            mode output will be multipled by -1.
        folder
            Directory to save fig if `save_fig` is `True`
        name
            Name of the fig saved if `save_fig` is `True`
        suy_ticks
            Ticks for the maps of the SUY output
        suz_ticks
            Ticks for the maps of the SUZ output
        suy_levels
            Levels for the maps of the SUY output
        suz_levels
            Levels for the maps of the SUZ output

        Returns
        -------
        plt.Figure
            Figure object from matplotlib

        Sequence[plt.Axes]
            Tuple of axes in figure
        """
        nm = 3
        if signs is not None:
            if len(signs) != nm:
                raise TypeError(f'Expected signs to be a sequence of the same length as number of modes ({nm})')
            if any(type(x) != bool for x in signs):
                raise TypeError(f'Expected signs to be a sequence of boolean: either True or False, got {signs}')

        nrows = 3
        ncols = 3

        fig: plt.Figure = plt.figure(figsize=_calculate_figsize(None, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))

        axs = (
            fig.add_subplot(nrows, ncols, 1),
            fig.add_subplot(nrows, ncols, 4),
            fig.add_subplot(nrows, ncols, 7),
            fig.add_subplot(nrows, ncols, 2, projection=ccrs.PlateCarree(0 if self.dsy.region.lon0 < self.dsy.region.lonf else 180)),
            fig.add_subplot(nrows, ncols, 5, projection=ccrs.PlateCarree(0 if self.dsy.region.lon0 < self.dsy.region.lonf else 180)),
            fig.add_subplot(nrows, ncols, 8, projection=ccrs.PlateCarree(0 if self.dsy.region.lon0 < self.dsy.region.lonf else 180)),
            fig.add_subplot(nrows, ncols, 3, projection=ccrs.PlateCarree(0 if self.dsz.region.lon0 < self.dsz.region.lonf else 180)),
            fig.add_subplot(nrows, ncols, 6, projection=ccrs.PlateCarree(0 if self.dsz.region.lon0 < self.dsz.region.lonf else 180)),
            fig.add_subplot(nrows, ncols, 9, projection=ccrs.PlateCarree(0 if self.dsz.region.lon0 < self.dsz.region.lonf else 180)),
        )

        # Plot timeseries
        ax: plt.Axes
        for i, ax in enumerate(axs[:3]):
            # # ax.margins(0)
            _plot_ts(
                time=self._dsy.time.values,
                arr=self.Us[i, :],
                ax=ax,
                title=f'Us Vs mode {i + 1}',
                color='green',
                label='Us'
            )
            _plot_ts(
                time=self._dsz.time.values,
                arr=self.Vs[i, :],
                ax=ax,
                title=None,
                color='blue',
                label='Vs'
            )
            ax.legend()
            ax.grid(True)
        axs[0].legend(loc='upper left')

        # suy = SUY
        # suy[suy == 0.0] = np.nan

        n = 20
        for i, (var_name, su, ru, lats, lons, cm, ticks, levels, region) in enumerate((
                ('SUY', self.SUY, self.RUY_sig, self._dsy.lat, self._dsy.lon, 'bwr', suy_ticks, suy_levels, self._dsy.region),
                ('SUZ', self.SUZ, self.RUZ_sig, self._dsz.lat, self._dsz.lon, cmap, suz_ticks, suz_levels, self._dsz.region)
        )):
            if levels is None:
                _std = np.nanstd(su)
                _m = np.nanmean(su)
                bound = max(abs(_m - _std), abs(_m + _std))
                levels = np.linspace(-bound, bound, n)
            if region.lon0 < region.lonf:
                xlim = sorted((lons.values[0], lons.values[-1]))
            else:
                xlim = [region.lon0 - 180, region.lonf + 180]
            ylim = sorted((lats.values[-1], lats.values[0]))

            current_axes = axs[3 * (i + 1):3 * (i + 1) + 3]
            for j, ax in enumerate(current_axes):
                title = f'{var_name} mode {j + 1}. ' \
                        f'SCF={self.scf[j]*100:.01f}'

                t = su[:, j].transpose().reshape((len(lats), len(lons)))
                th = ru[:, j].transpose().reshape((len(lats), len(lons)))

                if signs is not None:
                    if signs[j]:
                        t *= -1

                im = _plot_map(
                    t, lats, lons, fig, ax, title,
                    levels=levels, xlim=xlim, ylim=ylim, cmap=cm, ticks=ticks,
                    colorbar=j == 2,
                )
                ax.contourf(
                    lons, lats, th, colors='none', hatches='..', extend='both',
                    transform=ccrs.PlateCarree()
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
            path = os.path.join(folder, f'mca-plot_z-{self._dsz.var}_y-{self._dsy.var}.png')
        else:
            path = os.path.join(folder, name)

        _apply_flags_to_fig(
            fig, path,
            save_fig=save_fig,
            show_plot=show_plot,
            halt_program=halt_program,
        )

        return fig, axs

    @classmethod
    def load(cls, prefix: str, folder: str = '.', *,
             dsy: Optional[Preprocess] = None,
             dsz: Optional[Preprocess] = None,
             **attrs: Any) -> 'MCA':
        """Load an MCA object from matrices and type

        Parameters
        ----------
        prefix : str
            Prefix of the files containing the information for the object
        folder : str
            Directory of the files
        dsy : Preprocess
            Preprocessed dataset of the predictor variable
        dsz : Preprocess
            Preprocessed dataset of the predicting variable

        Returns
        -------
            MCA
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


def index_regression(
    data: LandArray,
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
    ns, nt = data.shape
    reg = data.values.dot(index) / nt
    cor = np.zeros(ns, dtype=np.float32)
    cor[data.land_mask] = np.nan
    pvalue = np.zeros(ns, dtype=np.float32)
    pvalue[data.land_mask] = np.nan


    if sig == 'test-t':
        result = np.apply_along_axis(stats.pearsonr, 1, data.not_land_values, index)
        cor[~data.land_mask] = result[:, 0]
        pvalue[~data.land_mask] = result[:, 1]

        cor_sig = cor.copy()
        reg_sig = reg.copy()
        cor_sig[pvalue > alpha] = np.nan
        reg_sig[pvalue > alpha] = np.nan
    elif sig == 'monte-carlo':
        if montecarlo_iterations is None:
            raise ValueError('Expected argument `montecarlo_iteration` for `monte-carlo` sig')

        corp = np.empty([ns, montecarlo_iterations])
        for p in range(montecarlo_iterations):
            corp[~data.land_mask, p] = pearsonr_2d(data.not_land_values, np.random.permutation(index))

        result = np.apply_along_axis(stats.pearsonr, 1, data.not_land_values, index)
        cor[~data.land_mask] = result[:, 0]

        for nn in range(ns):
            hcor = np.count_nonzero((cor[nn] > 0) & (corp[nn, :] < cor[nn]) | (cor[nn] < 0) & (corp[nn, :] > cor[nn]))
            # nivel de confianza
            pvalue[nn] = hcor / montecarlo_iterations

        cor_sig = cor.copy()
        reg_sig = reg.copy()
        cor_sig[pvalue >= (1 - alpha)] = np.nan
        reg_sig[pvalue >= (1 - alpha)] = np.nan
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
    nm: int,
    ny: int,
) -> npt.NDArray[np.float32]:
    # (((SUY * inv(Us * Us')) * Us) * Z') * nt * nm / ny
    return cast(
        npt.NDArray[np.float32],
        np.dot(np.dot(np.dot(suy, np.linalg.inv(np.dot(us, np.transpose(us)))), us), np.transpose(z)) * nm / ny)
