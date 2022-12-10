import os
from typing import Optional, Tuple, Any, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from scipy import sparse, signal
import scipy.sparse.linalg
import xarray as xr
from scipy.stats import stats

from .. import Slise, F
from .._functions import time_from_here, time_to_here, slise2str, _debuginfo, debugprint
from .._procedure import _Procedure, _plot_map, _apply_flags_to_fig, _calculate_figsize, MAX_HEIGHT, MAX_WIDTH, _plot_ts
from .preprocess import Preprocess


__all__ = [
    'MCA',
    'index_regression',
]


class MCA(_Procedure):
    """Maximum covariance analysis between y (predictor)
    and Z (predictand)

    Parameters
    ----------
        dsy : Preprocess
            predictor
        dsz : Preprocess
            Predictand
        nm : int
            Number of modes
        alpha : float
            Significance level
        sig : {'monte-carlo', 'test-t'}
            Signification technique: monte-carlo or test-t
        dsy_index_regression : optional, Preprocess
            Predictor to send to index regression. Degault is the same as y
        dsz_index_regression : optional, Preprocess
            Predictand to send to index regression. Degault is the same as z
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

    @property
    def var_names(self) -> Tuple[str, ...]:
        return (
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
        dsy_index_regression: Optional[Preprocess] = None,
        dsz_index_regression: Optional[Preprocess] = None,
    ):
        self._dsz = dsz
        self._dsy = dsy

        _debuginfo(f"""Applying MCA 
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

        z_index_regression = dsz_index_regression.data if dsz_index_regression is not None else None
        y_index_regression = dsy_index_regression.data if dsy_index_regression is not None else None

        self._mca(dsz.data, dsy.data, nm, alpha, sig, z_index_regression, y_index_regression)
        debugprint(f'       Took: {time_to_here():.03f} seconds')

        # first you calculate the covariance matrix
        # c = np.nan_to_num(np.dot(y, np.transpose(z)), nan=NAN_VAL)

    @classmethod
    def from_nparrays(
        cls,
        z: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        nm: int,
        alpha: float,
        sig: str = 'test-t',
        z_index_regression: Optional[npt.NDArray[np.float32]] = None,
        y_index_regression: Optional[npt.NDArray[np.float32]] = None,
    ) -> 'MCA':
        """
        Alternative constructor for mca that takes np arrays

        Parameters
        ----------
            z : array-like
                Predictand (space x time)
            y : array-like
                Predictor (space x time)
            nm : int
                Number of modes
            alpha : alpha
               Significance level
            sig : {'monte-carlo', 'test-t'}
                Signification technique: monte-carlo or test-t
            z_index_regression : optional, array-like
                Predictand (space x time) to send to index regression. Degault is the same as z
            y_index_regression : optional, array-like
                Predictor (space x time) to send to index regression. Degault is the same as y

        Returns
        -------
            MCA
                MCA object with the methodology performed

        See Also
        --------
            MCA
        """
        m = cls.__new__(MCA)
        m._mca(z, y, nm, alpha, sig, z_index_regression, y_index_regression)
        return m

    def _mca(
        self,
        z: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        nm: int,
        alpha: float,
        sig: str,
        z_index_regression: Optional[npt.NDArray[np.float32]] = None,
        y_index_regression: Optional[npt.NDArray[np.float32]] = None,
    ) -> None:
        if y_index_regression is None:
            y_index_regression = y
        if z_index_regression is None:
            z_index_regression = z

        nz, nt = z.shape
        ny, nt = y.shape

        c = np.dot(signal.detrend(y), np.transpose(z))
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
        u = np.dot(np.transpose(y), r)
        # u = np.dot(np.transpose(y), r[:, :nm])
        # calculamos las anomalías estandarizadas
        v = np.dot(np.transpose(z), q.transpose())
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
            ) = index_regression(y_index_regression, self.Us[i, :], alpha, sig)

            (
                self.RUZ[:, i],
                self.pvalruz[:, i],
                self.RUZ_sig[:, i],
                self.SUZ[:, i],
                self.SUZ_sig[:, i]
            ) = index_regression(z_index_regression, self.Us[i, :], alpha, sig)

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
        flags: F = F(0),
        cmap: str = 'bwr',
        signs: Optional[Sequence[bool]] = None,
        dir: Optional[str] = None,
        name: Optional[str] = None,
        suy_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
        suz_ticks: Optional[
            Union[npt.NDArray[np.float32], Sequence[float]]
        ] = None,
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot MCA results

        Parameters
        ----------
            flags : spy4cast.stypes.F
                ...
            cmap : str
                ...
            signs : optional, array-like[bool]
                Indicates for each mode wether to change the sign
            dir : str
                ....
            name : str
                ...
            suy_ticks : str
                ...
            suz_ticks : str
                ...
        """
        nm = 3
        if signs is not None:
            if len(signs) != nm:
                raise TypeError(f'Expected signs to be a sequence of the same length as number of modes ({nm})')
            if any(type(x) != bool for x in signs):
                raise TypeError(f'Expected signs to be a sequence of boolean: either True or False, got {signs}')

        nrows = 3
        ncols = 3

        fig = plt.figure(figsize=_calculate_figsize(None, maxwidth=MAX_WIDTH, maxheight=MAX_HEIGHT))

        axs = [
            plt.subplot(nrows * 100 + ncols * 10 + i,
                        projection=(ccrs.PlateCarree() if i > 3 else None))
            for i in range(1, ncols * nrows + 1)
        ]

        # Plot timeseries
        for i, ax in enumerate(axs[:3]):
            # # ax.margins(0)
            _plot_ts(
                time=self.ytime.values,
                arr=self.Us[i, :],
                ax=ax,
                title=f'Us Vs mode {i + 1}',
                color='green',
                label='Us'
            )
            _plot_ts(
                time=self.ztime.values,
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
        for i, (var_name, su, ru, lats, lons, cm, ticks) in enumerate((
                ('SUY', self.SUY, self.RUY_sig, self.ylat, self.ylon, 'bwr', suy_ticks),
                ('SUZ', self.SUZ, self.RUZ_sig, self.zlat, self.zlon, cmap, suz_ticks)
        )):
            _std = np.nanstd(su)
            _m = np.nanmean(su)
            bound = max(abs(_m - _std), abs(_m + _std))
            levels = np.linspace(-bound, bound, n)
            xlim = sorted((lons.values[0], lons.values[-1]))
            ylim = sorted((lats.values[-1], lats.values[0]))

            for j, ax in enumerate(axs[3 * (i + 1):3 * (i + 1) + 3]):
                title = f'{var_name} mode {j + 1}. ' \
                        f'SCF={self.scf[j]*100:.01f}'

                t = su[:, j].transpose().reshape((len(lats), len(lons)))
                th = ru[:, j].transpose().reshape((len(lats), len(lons)))

                if signs is not None:
                    if signs[j]:
                        t *= -1

                _plot_map(
                    t, lats, lons, fig, ax, title,
                    levels=levels, xlim=xlim, ylim=ylim, cmap=cm, ticks=ticks,
                )
                ax.contourf(
                    lons, lats, th, colors='none', hatches='..', extend='both',
                    transform=ccrs.PlateCarree()
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
            path = os.path.join(dir, f'mca-plot_z-{self.zvar}_y-{self.yvar}.png')
        else:
            path = os.path.join(dir, name)

        _apply_flags_to_fig(
            fig, path, F(flags)
        )

        return fig, axs

    @classmethod
    def load(cls, prefix: str, dir: str = '.', *,
             dsz: Optional[Preprocess] = None,
             dsy: Optional[Preprocess] = None,
             **attrs: Any) -> 'MCA':
        if len(attrs) != 0:
            raise TypeError('Load only takes two keyword arguments: dsy and dsz')
        if dsz is None or dsy is None:
            raise TypeError('To load an MCA object you must provide `dsz` and `dsy` keyword arguments')
        if type(dsz) != Preprocess or type(dsy) != Preprocess:
            raise TypeError(f'Unexpected types ({type(dsz)} and {type(dsy)}) for `dsz` and `dsy`. Expected type `Preprocess`')

        self: MCA = super().load(prefix, dir)

        self._dsz = dsz
        self._dsy = dsy
        return self


def index_regression(
    data: npt.NDArray[np.float32],
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
        sig : str, {'monte-carlo', 'test-t'}
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
    reg = data.dot(index) / (nt - 1)

    if sig == 'test-t':
        result = np.apply_along_axis(stats.pearsonr, 1, data, index)
        cor = result[:, 0]
        pvalue = result[:, 1]

        cor_sig = cor.copy()
        reg_sig = reg.copy()
        cor_sig[pvalue > alpha] = np.nan
        reg_sig[pvalue > alpha] = np.nan
    elif sig == 'monte-carlo':
        if montecarlo_iterations is None:
            raise ValueError('Expected argument `montecarlo_iteration` for `monte-carlo` sig')

        corp = np.empty([ns, montecarlo_iterations])
        for p in range(montecarlo_iterations):
            corp[:, p] = pearsonr_2d(data, np.random.permutation(index))

        result = np.apply_along_axis(stats.pearsonr, 1, data, index)
        cor = result[:, 0]

        pvalue = np.zeros([ns, ], dtype=np.float32)
        for nn in range(ns):
            hcor = np.count_nonzero((cor[nn] > 0) & (corp[nn, :] < cor[nn]) | (cor[nn] < 0) & (corp[nn, :] > cor[nn]))
            # nivel de confianza
            pvalue[nn] = hcor / montecarlo_iterations

        cor_sig = cor.copy()
        reg_sig = reg.copy()
        cor_sig[pvalue >= (1 - alpha)] = np.nan
        reg_sig[pvalue >= (1 - alpha)] = np.nan
    else:
        assert False, 'Unreachable'

    return cor, pvalue, cor_sig, reg, reg_sig


def pearsonr_2d(y: npt.NDArray[np.float_], x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:, None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:, None], 2), axis=1))
    rho: npt.NDArray[np.float_] = upper / lower
    return rho
