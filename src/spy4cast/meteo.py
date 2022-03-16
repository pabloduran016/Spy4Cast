"""
Module that contains a collection of functions used in methodologies
"""

import pandas as pd
import xarray as xr
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from scipy import stats
import scipy
import numpy.typing as npt



__all__ = [
    'CrossvalidationOut',
    'MCAOut',
    'clim',
    'anom',
    'npanom',
    'mca',
    'index_regression',
    'crossvalidation_mp',
    'crossvalidation',
]

NAN_VAL = 1e4

@dataclass
class CrossvalidationOut:
    """Dataclass that is the output of the crossvalidation methodology

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


@dataclass
class MCAOut:
    """Dataclass that is the output of the MCA methodology

    Attributes
    ----------
    RUY : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    RUY_sig : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    SUY : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    SUY_sig : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    RUZ : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    RUZ_sig : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    SUZ : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    SUZ_sig : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    Us : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    Vs : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    scf : npt.NDArray[np.float32]
        .. todo:: Not documented yet
    alpha : float
        Correlation factor
    """
    # TODO: Add docs for MCAOut field
    RUY: npt.NDArray[np.float32]
    RUY_sig: npt.NDArray[np.float32]
    SUY: npt.NDArray[np.float32]
    SUY_sig: npt.NDArray[np.float32]
    RUZ: npt.NDArray[np.float32]
    RUZ_sig: npt.NDArray[np.float32]
    SUZ: npt.NDArray[np.float32]
    SUZ_sig: npt.NDArray[np.float32]
    Us: npt.NDArray[np.float32]
    Vs: npt.NDArray[np.float32]
    scf: npt.NDArray[np.float32]
    alpha: float


def clim(array: xr.DataArray, dim: str = 'time') -> xr.DataArray:
    """Function that performs the climatology of a xarray Dataset

    The climatology is the average across a given axis

    Parameters
    ----------
        array : xr.DataArray
            Xarray DataArray where you wish to perform the climatology

        dim : str, default='time'
            Dimension where the climatology is going to be performed on

    See Also
    --------
    plotters.ClimerTS, plotters.ClimerMap

    Raises
    ------
        TypeError
            If array is not an instance of `xr.DataArray`
        ValueError
            If dim is not `month`, `time` or `year`
    """
    if not isinstance(array, xr.DataArray):
        raise TypeError(f"Expected type xarray.DataArray, got {type(array)}")
    if dim == 'year' or dim == 'month':
        months = list(array.groupby('time.month').groups.keys())  # List of month values
        nm = len(months)
        # Create index to reshape time variable
        ind = pd.MultiIndex.from_product(
            (months, array.time[nm - 1::nm].data),
            names=('month', 'year')
        )
        # Reshape time variable
        assert len(array.shape) == 2 or len(array.shape) == 1,\
            f'Clim implemented only for 1 and 2 dimensional arrays, for now'
        arr = array.assign_coords(
            time=('time', ind)
        ).unstack('time').transpose('year', 'month')
        rv: xr.DataArray =  arr.mean(dim=dim)
    elif dim == 'time':  # Apply across year and month
        assert 'time' in array.dims
        rv = array.mean(dim=dim)
    else:
        raise ValueError(f'Invalid dim {dim}')
    return rv


def anom(
        array: xr.DataArray, st: bool = False
) -> xr.DataArray:

    """Function to calculate the anomalies on a xarray DataArray

    The anomaly is the time variable minus the mean across all times of a given point

    Parameters
    ----------
        array : xr.DataArray
            Array to process the anomalies. Must have a dimension called `time`
        st : bool, default=False
            Indicates whether the anomaly should standarized. Divide by the standard deviation

    Raises
    ------
        TypeError
            If array is not an instance of `xr.DataArray`
        ValueError
            If the number of dimension of the array is not either 3 (map) or 1 (time series)

    See Also
    --------
    npanom
    """
    # print(f'[INFO] <meteo.Meteo.anom()> called, st: {st}')
    if not isinstance(array, xr.DataArray):
        raise TypeError(f"Invalid type for array: {type(array)}")

    assert 'time' in array.dims, 'Cant\'t recognise time key in array'
    # List of month values
    months_set = set(array.groupby('time.month').groups.keys())
    nm = len(months_set)
    months = array['time.month'][:nm].data
    # Create index to reshape time variab le
    ind = pd.MultiIndex.from_product(
        (array.time[nm - 1::nm]['time.year'].data, months),
        names=('year', 'month')
    )
    assert len(array.time) == len(ind)
    if len(array.shape) == 3:  # 3d array
        # Reshape time variable
        lat_key = 'latitude' if 'latitude' in array.dims else 'lat'
        lon_key = 'longitude' if 'longitude' in array.dims else 'lon'
        assert lat_key in array.dims and lon_key in array.dims,\
            'Can\'t recognise keys'
        arr = array.assign_coords(time=('time', ind))
        # arr must be a DataArray with dims=(months, year, lat, lon)
        a = arr.groupby('year').mean()
        b: xr.DataArray = a - a.mean('year')
        if st:
            # print('[INFO] <meteo.Meteo.anom()> standarzing')
            rv: xr.DataArray = b / b.std()
            return rv
        return b
    elif len(array.shape) == 1:  # time series
        assert 'latitude' not in array.dims and 'longitude' not in array.dims,\
            'Unidimensional arrays time must be the only dimension'
        arr = array.assign_coords(
            time=('time', ind)
        ).unstack('time').transpose('year', 'month')
        a = arr.mean('month')
        b = a - a.mean('year')
        if st:
            # print('[INFO] <meteo.Meteo.anom()> standarzing')
            rv = b / b.std()
            return rv
        return b
    else:
        raise ValueError(
            'Invalid dimensions of array from anom methodology'
        )


def npanom(
    array: npt.NDArray[np.float32],
    axis: int = 0,
    st: bool = False
) -> npt.NDArray[np.float32]:

    """Function to calculate the anomalies on a numpy array

    The anomaly is the time variable minus the mean across all times
    of a given point

    Parameters
    ----------
        array : npt.NDArray[np.float32]
            Array to process the anomalies.
        axis : int, default=0
            Axis where to perform teh anomaly, ussually the time axis
        st : bool, default=False
            Indicates whether the anomaly should standarized.
            Divide by the standard deviation

    See Also
    --------
    anom
    """
    b: npt.NDArray[np.float32] = array - array.mean(axis=axis)
    if st:
        rv: npt.NDArray[np.float32] = b / b.std(axis=axis)
        return rv
    return b


def mca(
    z: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    nm: int, alpha: float
) -> MCAOut:

    """"Maximum covariance analysis between y (predictor)
    and Z (predictand)

    Parameters
    ----------
        z : npt.NDArray[np.float32]
            Predictand array (space x time)
        y : npt.NDArray[np.float32]
            predictor array (space x time)
        nm : int
            Number of modes
        alpha : float
            Significance level

    Returns
    -------
        MCAOut

    See Also
    --------
    MCAOut
    """
    nz, nt = z.shape
    ny, nt = y.shape
    # nyr = int(nt / nmes)

    # first you calculate the covariance matrix
    # c = np.nan_to_num(np.dot(y, np.transpose(z)), nan=NAN_VAL)
    c = np.dot(y, np.transpose(z))
    if type(c) == np.ma.MaskedArray:
        c = c.data
    r, d, q = scipy.linalg.svd(c)

    # y había que transponerla si originariamente era (espacio, tiempo),
    # pero ATN_e es (tiempo, espacio) así
    # que no se transpone
    u = np.dot(np.transpose(y), r[:, :nm])
    # u = np.dot(np.transpose(y), r[:, :nm])
    # calculamos las anomalías estandarizadas
    v = np.dot(np.transpose(z), q[:, :nm])
    # v = np.dot(np.transpose(z), q[:, :nm])
    out = MCAOut(
        RUY=np.zeros([ny, nm], dtype=np.float32),
        RUY_sig=np.zeros([ny, nm], dtype=np.float32),
        SUY=np.zeros([ny, nm], dtype=np.float32),
        SUY_sig=np.zeros([ny, nm], dtype=np.float32),
        RUZ=np.zeros([nz, nm], dtype=np.float32),
        RUZ_sig=np.zeros([nz, nm], dtype=np.float32),
        SUZ=np.zeros([nz, nm], dtype=np.float32),
        SUZ_sig=np.zeros([nz, nm], dtype=np.float32),
        # Standarized anom across axis 0
        Us=((u - u.mean(0)) / u.std(0)).transpose(),
        # Standarized anom across axis 0
        Vs=((v - v.mean(0)) / v.std(0)).transpose(),
        scf=(d / np.sum(d))[:nm],
        alpha=alpha,
    )
    pvalruy = np.zeros([ny, nm], dtype=np.float32)
    pvalruz = np.zeros([nz, nm], dtype=np.float32)
    for i in range(nm):
        out.RUY[:, i],\
            pvalruy[:, i],\
            out.RUY_sig[:, i],\
            out.SUY[:, i],\
            out.SUY_sig[:, i] \
            = index_regression(y, out.Us[i, :], alpha)

        out.RUZ[:, i],\
            pvalruz[:, i],\
            out.RUZ_sig[:, i],\
            out.SUZ[:, i],\
            out.SUZ_sig[:, i] \
            = index_regression(z, out.Us[i, :], alpha)
    # return cast(MCA_OUT, out)
    return out


def index_regression(
    data: npt.NDArray[np.float32],
    index: npt.NDArray[np.float32],
    alpha: float
) -> Tuple[npt.NDArray[np.float32], ...]:

    """Create correlation (pearson correlation) and regression

    Parameters
    ----------
        data : npt.NDArray[np.float32]
            Data to perform the methodology in (space x time)
        index : npt.NDArray[np.float32]
            Unidimensional array: temporal series
        alpha : float
            Significance level

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

    Cor = np.zeros([ns, ], dtype=np.float32)
    Pvalue = np.zeros([ns, ], dtype=np.float32)
    for nn in range(ns):  # Pearson correaltion for every point in the map
        Cor[nn], Pvalue[nn] = stats.pearsonr(data[nn, :], index)

    Cor_sig = Cor.copy()
    Cor_sig[Pvalue > alpha] = np.nan

    reg = data.dot(index) / (nt - 1)
    reg_sig = reg.copy()
    reg_sig[Pvalue > alpha] = np.nan
    return Cor, Pvalue, Cor_sig, reg, reg_sig


def _crossvalidate_year(
    year: int,
    z: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    nt: int,
    ny: int,
    yrs: npt.NDArray[np.int32],
    nm: int,
    alpha: float
) -> Tuple[npt.NDArray[np.float32], ...]:
    """Function of internal use that processes a single year for
    crossvalidation"""

    print('year:', year, 'of', nt)
    z2 = z[:, yrs != year]
    y2 = y[:, yrs != year]
    mca_out = mca(z2, y2, nm, alpha)
    scf = mca_out.scf  #
    psi = np.dot(
            np.dot(
                np.dot(
                    mca_out.SUY, np.linalg.inv(
                        np.dot(mca_out.Us, np.transpose(mca_out.Us))
                    )
                ), mca_out.Us
            ), np.transpose(z2)
    ) * nt * nm / ny
    zhat = np.dot(np.transpose(y[:, year]), psi)
    r_uv = np.zeros(nm, dtype=np.float32)
    p_uv = np.zeros(nm, dtype=np.float32)
    for m in range(nm):
        r_uv[m], p_uv[m] = stats.pearsonr(mca_out.Us[m, :], mca_out.Vs[m, :])

    return scf, zhat, r_uv, p_uv, mca_out.Us


def crossvalidation_mp(
    y: npt.NDArray[np.float32],
    z: npt.NDArray[np.float32],
    nm: int, alpha: float
) -> CrossvalidationOut:

    """Perform crossvalidation methodology with multiprocessing

    Parameters
    ----------
        y : npt.NDArray[np.float32]
            Predictor field (space x time)
        z : npt.NDArray[np.float32]
            Predictand field (space x time)
        nm : npt.NDArray[np.float32]
            Number of modes
        alpha : float
            Significance coeficient

    Returns
    -------
        CrossvalidationOut

    See Also
    --------
    CrossvalidationOut, mca, crossvalidation
    """
    nz, ntz = z.shape
    ny, nty = y.shape

    assert ntz == nty
    nt = ntz

    zhat = np.zeros_like(z, dtype=np.float32)
    scf = np.zeros([nm, nt], dtype=np.float32)
    r_uv = np.zeros([nm, nt], dtype=np.float32)
    p_uv = np.zeros([nm, nt], dtype=np.float32)
    # crosvalidated year on axis 2
    us = np.zeros([nm, nt, nt], dtype=np.float32)

    yrs = np.arange(nt)

    import multiprocessing as mp
    # Step 1: Init multiprocessing.Pool()
    count = mp.cpu_count()
    with mp.Pool(count) as pool:
        # print(f'Starting pool with {count=}')
        processes = []
        for i in yrs:
            # print(f'applying async on process {i=}')
            p = pool.apply_async(_crossvalidate_year, kwds={
                'year': i, 'z': z, 'y': y, 'nt': nt, 'ny': ny, 'yrs': yrs,
                'nm': nm, 'alpha': alpha
            })
            processes.append(p)

        for i in yrs:
            values = processes[i].get()
            scf[:, i], zhat[:, i], r_uv[:, i], p_uv[:, i], \
                us[:, [x for x in range(nt) if x != i], i] = values

    # Step 3: Don't forget to close

    # for i in yrs:
    #     scf[:, i], zhat[:, i], r_uv[:, i], p_uv[:, i], \
    #         = _crossvalidate_year(year=i, z=z, y=y, nt=nt, ny=ny,
    #         yrs=yrs, nmes=nmes, nm=nm, alpha=alpha)

    r_z_zhat_t = np.zeros(nt, dtype=np.float32)
    p_z_zhat_t = np.zeros(nt, dtype=np.float32)
    for j in range(nt):
        rtt = stats.pearsonr(zhat[:, j], z[:, j])  # skill series
        r_z_zhat_t[j] = rtt[0]
        p_z_zhat_t[j] = rtt[1]

    r_z_zhat_s = np.zeros([nz], dtype=np.float32)
    p_z_zhat_s = np.zeros([nz], dtype=np.float32)
    for i in range(nz):
        rs = stats.pearsonr(zhat[i, :], z[i, :])
        r_z_zhat_s[i] = rs[0]
        p_z_zhat_s[i] = rs[1]
    # rs = np.zeros(nz, dtype=np.float32)
    # rs_sig = np.zeros(nz, dtype=np.float32)
    # rmse = np.zeros(nz, dtype=np.float32)

    return CrossvalidationOut(
        zhat=zhat,  # Hindcast of field to predict using crosvalidation
        scf=scf,  # Squared covariance fraction of the mca for each mode
        # Correlation between zhat and Z for each time (time series)
        r_z_zhat_t=r_z_zhat_t,
        p_z_zhat_t=p_z_zhat_t,  # P values of rt
        # Correlation between time series (for each point) of zhat and z (map)
        r_z_zhat_s=r_z_zhat_s,
        p_z_zhat_s=p_z_zhat_s,  # P values of rr
        r_uv=r_uv,  # Correlation score betweeen u and v for each mode
        p_uv=p_uv,  # P value of ruv
        us=us,  # crosvalidated year on axis 2
        alpha=alpha,  # Correlation factor
    )


def crossvalidation(
    y: npt.NDArray[np.float32],
    z: npt.NDArray[np.float32],
    nm: int, alpha: float
) -> CrossvalidationOut:

    """Perform crossvalidation methodology

    Parameters
    ----------
        y : npt.NDArray[np.float32]
            Predictor field (space x time)
        z : npt.NDArray[np.float32]
            Predictand field (space x time)
        nm : npt.NDArray[np.float32]
            Number of modes
        alpha : float
            Significance coeficient

    Returns
    -------
        CrossvalidationOut

    See Also
    --------
    CrossvalidationOut, mca, crossvalidation_mp
    """
    nz, ntz = z.shape
    ny, nty = y.shape

    assert ntz == nty
    nt = ntz

    zhat = np.zeros_like(z, dtype=np.float32)
    scf = np.zeros([nm, nt], dtype=np.float32)
    r_uv = np.zeros([nm, nt], dtype=np.float32)
    p_uv = np.zeros([nm, nt], dtype=np.float32)
    # crosvalidated year on axis 2
    us = np.zeros([nm, nt, nt], dtype=np.float32)
    # estimación de zhat para cada año
    yrs = np.arange(nt)

    for i in yrs:
        scf[:, i], zhat[:, i], r_uv[:, i], p_uv[:, i],\
            us[:, [x for x in range(nt) if x != i], i] = _crossvalidate_year(
                year=i, z=z, y=y, nt=nt, ny=ny, yrs=yrs,
                nm=nm, alpha=alpha
            )

    r_z_zhat_t = np.zeros(nt, dtype=np.float32)
    p_z_zhat_t = np.zeros(nt, dtype=np.float32)
    for j in range(nt):
        rtt = stats.pearsonr(zhat[:, j], z[:, j])  # serie de skill
        r_z_zhat_t[j] = rtt[0]
        p_z_zhat_t[j] = rtt[1]

    r_z_zhat_s = np.zeros([nz], dtype=np.float32)
    p_z_zhat_s = np.zeros([nz], dtype=np.float32)
    for i in range(nz):
        r_z_zhat_s[i], p_z_zhat_s[i] = stats.pearsonr(zhat[i, :], z[i, :])
    # rs = np.zeros(nz, dtype=np.float32)
    # rs_sig = np.zeros(nz, dtype=np.float32)
    # rmse = np.zeros(nz, dtype=np.float32)

    return CrossvalidationOut(
        zhat=zhat,  # Hindcast of field to predict using crosvalidation
        scf=scf,  # Squared covariance fraction of the mca for each mode
        r_z_zhat_t=r_z_zhat_t,  # Correlation between zhat and Z for
        # each time (time series)
        p_z_zhat_t=p_z_zhat_t,  # P values of rt
        r_z_zhat_s=r_z_zhat_s,  # Correlation between time series
        # (for each point) of zhat and z (map)
        p_z_zhat_s=p_z_zhat_s,  # P values of rr
        r_uv=r_uv,  # Correlation score betweeen u and v for each mode
        p_uv=p_uv,  # P value of ruv
        us=us,  # crosvalidated year on axis 2
        alpha=alpha,  # Correlation factor
    )
