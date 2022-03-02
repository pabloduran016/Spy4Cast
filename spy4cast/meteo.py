import pandas as pd
import xarray as xr
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from scipy import stats
import scipy
import numpy.typing as npt


NAN_VAL =1e4


@dataclass
class CrossvalidationOut:
    zhat: npt.NDArray[np.float64]
    scf: npt.NDArray[np.float64]
    r_z_zhat_t: npt.NDArray[np.float64]
    p_z_zhat_t: npt.NDArray[np.float64]
    r_z_zhat_s: npt.NDArray[np.float64]
    p_z_zhat_s: npt.NDArray[np.float64]
    r_uv: npt.NDArray[np.float64]
    p_uv: npt.NDArray[np.float64]
    us: npt.NDArray[np.float64]
    alpha: float

@dataclass
class MCAOut:
    RUY: npt.NDArray[np.float64]
    RUY_sig: npt.NDArray[np.float64]
    SUY: npt.NDArray[np.float64]
    SUY_sig: npt.NDArray[np.float64]
    RUZ: npt.NDArray[np.float64]
    RUZ_sig: npt.NDArray[np.float64]
    SUZ: npt.NDArray[np.float64]
    SUZ_sig: npt.NDArray[np.float64]
    Us: npt.NDArray[np.float64]
    Vs: npt.NDArray[np.float64]
    scf: npt.NDArray[np.float64]


class Meteo:
    """Class containing functions to carry out on the dataset"""

    @classmethod
    def clim(cls, array: xr.DataArray, dim: str = 'time') -> xr.DataArray:
        if isinstance(array, xr.DataArray):
            if dim == 'year' or dim == 'month':
                months = list(array.groupby('time.month').groups.keys())  # List of month values
                nm = len(months)
                # Create index to reshape time variable
                ind = pd.MultiIndex.from_product((months, array.time[nm - 1::nm].data), names=('month', 'year'))
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
        raise ValueError(f"Expected type xarray.DataArray, got {type(array)}")

    @classmethod
    def anom(cls, array: xr.DataArray, st: bool = False) -> xr.DataArray:
        """
        Function to calculate the anomalies
        :param array: Array to process the anomalies. Fisrt dimension must be time
        :param st: bool to indicate whether the method should standarize
        """
        # print(f'[INFO] <meteo.Meteo.anom()> called, st: {st}')
        if isinstance(array, xr.DataArray):
            assert 'time' in array.dims, 'Cant\'t recognise time key in array'
            months_set = set(array.groupby('time.month').groups.keys())  # List of month values
            nm = len(months_set)
            months = array['time.month'][:nm].data
            # Create index to reshape time variab le
            ind = pd.MultiIndex.from_product((array.time[nm - 1::nm]['time.year'].data, months), names=('year', 'month'))
            assert len(array.time) == len(ind)
            if len(array.shape) == 3:  # 3d array
                # Reshape time variable
                lat_key = 'latitude' if 'latitude' in array.dims else 'lat'
                lon_key = 'longitude' if 'longitude' in array.dims else 'lon'
                assert lat_key in array.dims and lon_key in array.dims, 'Can\'t recognise keys'
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
                raise ValueError('Invalid dimensions of array from anom methodology')
        raise ValueError(f"Invalid type for array: {type(array)}")

    @staticmethod
    def npanom(array: npt.NDArray[np.float64], axis: int = 0, st: bool = False) -> npt.NDArray[np.float64]:
        """
        Function to calculate the anomalies
        :param array: Array to process the anomalies. space x time
        :param st: bool to indicate whether the method should standarize
        """
        b: npt.NDArray[np.float64] = array - array.mean(axis=axis)
        if st:
            rv: npt.NDArray[np.float64] = b / b.std(axis=axis)
            return rv
        return b

    @classmethod
    def mca(cls, z: npt.NDArray[np.float64], y: npt.NDArray[np.float64], nmes: int, nm: int, alpha: float) -> MCAOut:
        """"
        Maximum covariance analysis between y (predictor) and Z (predictand)

        :param z: predictand
        :param y: predictor. space x time.
        :param nm: number of modes
        :param nmes: es datos meses metes al año. Si haces la media estacional es 1, pero si metes enero y feb por separado es 2
        :param alpha: significant level
        :return: RUY, RUY_sig, SUY, SUY_sig, RUZ, RUZ_sig, SUZ, SUZ_sig, us, Vs, scf
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

        # y había que transponerla si originariamente era (espacio, tiempo), pero ATN_e es (tiempo, espacio) así
        # que no se transpone
        u = np.dot(np.transpose(y), r[:, :nm])
        # u = np.dot(np.transpose(y), r[:, :nm])
        # calculamos las anomalías estandarizadas
        v = np.dot(np.transpose(z), q[:, :nm])
        # v = np.dot(np.transpose(z), q[:, :nm])
        out = MCAOut(
            RUY=np.zeros([ny, nm]),
            RUY_sig=np.zeros([ny, nm]),
            SUY=np.zeros([ny, nm]),
            SUY_sig=np.zeros([ny, nm]),
            RUZ=np.zeros([nz, nm]),
            RUZ_sig=np.zeros([nz, nm]),
            SUZ=np.zeros([nz, nm]),
            SUZ_sig=np.zeros([nz, nm]),
            Us=((u - u.mean(0)) / u.std(0)).transpose(),  # Standarized anom across axis 0
            Vs=((v - v.mean(0)) / v.std(0)).transpose(),  # Standarized anom across axis 0
            scf=(d / np.sum(d))[:nm],
        )
        pvalruy = np.zeros([ny, nm])
        pvalruz = np.zeros([nz, nm])
        for i in range(nm):
            out.RUY[:, i], pvalruy[:, i], out.RUY_sig[:, i], out.SUY[:, i], out.SUY_sig[:, i] = cls.index_regression(y, out.Us[i, :], alpha)
            out.RUZ[:, i], pvalruz[:, i], out.RUZ_sig[:, i], out.SUZ[:, i], out.SUZ_sig[:, i] = cls.index_regression(z, out.Us[i, :], alpha)
        # return cast(MCA_OUT, out)
        return out

    @staticmethod
    def index_regression(data: npt.NDArray[np.float64], index: npt.NDArray[np.float64], alpha: float
                         ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Create correlation (using pearson correlation) and regression
        :param data: `data` (space, time)
        :param index: array of temporal series
        :param alpha: Significant level
        :return: Correlation map, map with p values, significative corrrelation map, regression map, significative regression map
        """
        n1, n2 = data.shape
        # inicializamos las matrices
        Cor = np.zeros([n1, ])
        Pvalue = np.zeros([n1, ])
        for nn in range(n1):  # para cada punto del espacio hacemos la correlación de Pearson
            bb = stats.pearsonr(data[nn, :], index)  # bb tiene dos salidas: la primera es corre y la segunda es p-value que es el nivel de confianza
            # asociado al valor de la correlación tras aplicar un test-t
            Cor[nn] = bb[0]
            Pvalue[nn] = bb[1]
        # generamos una variable que es para que no se muestren mas que los valores de Cor cuando la correlacion
        # es significativa
        Cor_sig = Cor.copy()
        Cor_sig[Pvalue > alpha] = np.nan
        # generamos el mapa de regresión mediante multiplicación matricial. Ojo con las dimensiones!!
        reg = data.dot(index) / (n2 - 1)
        # igualmente, hacemos una máscara para que sólo se muestre el mapa de regresión cuando es significativo
        reg_sig = reg.copy()
        reg_sig[Pvalue > alpha] = np.nan
        return Cor, Pvalue, Cor_sig, reg, reg_sig

    @staticmethod
    def _crossvalidate_year(year: int, z: npt.NDArray[np.float64], y: npt.NDArray[np.float64], nt: int, ny: int, yrs: npt.NDArray[np.int32],
                            nmes: int, nm: int, alpha: float) -> Tuple[npt.NDArray[np.float64], ...]:
        print('year:', year, 'of', nt)
        z2 = z[:, yrs != year]
        y2 = y[:, yrs != year]
        mca_out = Meteo.mca(z2, y2, nmes, nm, alpha)
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
        zhat = np.dot(np.transpose(y[:, year]), psi) #
        r_uv = np.zeros(nm)
        p_uv = np.zeros(nm)
        for m in range(nm):
            r_uv[m], p_uv[m] = stats.pearsonr(mca_out.Us[m, :], mca_out.Vs[m, :]) #

        return scf, zhat, r_uv, p_uv, mca_out.Us

    @classmethod
    def crossvalidation_mp(cls, y: npt.NDArray[np.float64], z: npt.NDArray[np.float64], nmes: int, nm: int, alpha: float) -> CrossvalidationOut:
        nz, ntz = z.shape
        ny, nty = y.shape

        assert ntz == nty
        nt = ntz

        zhat = np.zeros_like(z)
        scf = np.zeros([nm, nt])
        r_uv = np.zeros([nm, nt])
        p_uv = np.zeros([nm, nt])
        us = np.zeros([nm, nt - 1, nt])  # crosvalidated year on axis 1
        # estimación de zhat para cada año
        yrs = np.arange(nt)

        import multiprocessing as mp
        # Step 1: Init multiprocessing.Pool()
        count = mp.cpu_count()
        with mp.Pool(count) as pool:
            # print(f'Starting pool with {count=}')
            processes = []
            for i in yrs:
                # print(f'applying async on process {i=}')
                p = pool.apply_async(cls._crossvalidate_year, kwds={
                    'year': i, 'z': z, 'y': y, 'nt': nt, 'ny': ny, 'yrs': yrs, 'nmes': nmes,
                    'nm': nm, 'alpha': alpha
                })
                processes.append(p)

            for i in yrs:
                values = processes[i].get()
                scf[:, i], zhat[:, i], r_uv[:, i], p_uv[:, i], us[:, :, i] = values

        # Step 3: Don't forget to close

        # for i in yrs:
        #     scf[:, i], zhat[:, i], r_uv[:, i], p_uv[:, i], \
        #         = cls._crossvalidate_year(year=i, z=z, y=y, nt=nt, ny=ny, yrs=yrs, nmes=nmes, nm=nm, alpha=alpha)

        r_z_zhat_t = np.zeros(nt)
        p_z_zhat_t = np.zeros(nt)
        for j in range(nt):
            rtt = stats.pearsonr(zhat[:, j], z[:, j])  # serie de skill
            r_z_zhat_t[j] = rtt[0]
            p_z_zhat_t[j] = rtt[1]

        r_z_zhat_s = np.zeros([nz])
        p_z_zhat_s = np.zeros([nz])
        for i in range(nz):
            rs = stats.pearsonr(zhat[i, :], z[i, :])  # bb tiene dos salidas: la primera es corre y la segunda es p-value que es el nivel de confianza
            # asociado al valor de la correlación tras aplicar un test-t
            r_z_zhat_s[i] = rs[0]
            p_z_zhat_s[i] = rs[1]
        # rs = np.zeros(nz)
        # rs_sig = np.zeros(nz)
        # rmse = np.zeros(nz)

        return CrossvalidationOut(
            zhat=zhat,  # Hindcast of field to predict using crosvalidation
            scf=scf,  # Squared covariance fraction of the mca for each mode
            r_z_zhat_t=r_z_zhat_t,  # Correlation between zhat and Z for each time (time series)
            p_z_zhat_t=p_z_zhat_t,  # P values of rt
            r_z_zhat_s=r_z_zhat_s,  # Correlation between time series (for each point) of zhat and z (map)
            p_z_zhat_s=p_z_zhat_s,  # P values of rr
            r_uv=r_uv,  # Correlation score betweeen u and v for each mode
            p_uv=p_uv,  # P value of ruv
            us=us,  # crosvalidated year on axis 2
            alpha=alpha,  # Correlation factor
        )

    @classmethod
    def crossvalidation(cls, y: npt.NDArray[np.float64], z: npt.NDArray[np.float64], nmes: int, nm: int, alpha: float) -> CrossvalidationOut:
        nz, ntz = z.shape
        ny, nty = y.shape

        assert ntz == nty
        nt = ntz

        zhat = np.zeros_like(z)
        scf = np.zeros([nm, nt])
        r_uv = np.zeros([nm, nt])
        p_uv = np.zeros([nm, nt])
        us = np.zeros([nm, nt - 1, nt])  # crosvalidated year on axis 2
        # estimación de zhat para cada año
        yrs = np.arange(nt)

        # results = [cls._crossvalidate_year(**{
        #     'year': i, 'z': z, 'y': y, 'nt': nt, 'ny': ny, 'yrs': yrs, 'nmes': nmes, 'nm': nm,
        #     'alpha': alpha
        # }) for i in yrs]
        #
        # for i in yrs:
        #     scf[:, i], zhat[:, i], r_uv[:, i], p_uv[:, i] = results[i]

        for i in yrs:
            scf[:, i], zhat[:, i], r_uv[:, i], p_uv[:, i], us[:, :, i] \
                = cls._crossvalidate_year(year=i, z=z, y=y, nt=nt, ny=ny, yrs=yrs, nmes=nmes, nm=nm, alpha=alpha)

        r_z_zhat_t = np.zeros(nt)
        p_z_zhat_t = np.zeros(nt)
        for j in range(nt):
            rtt = stats.pearsonr(zhat[:, j], z[:, j])  # serie de skill
            r_z_zhat_t[j] = rtt[0]
            p_z_zhat_t[j] = rtt[1]

        r_z_zhat_s = np.zeros([nz])
        p_z_zhat_s = np.zeros([nz])
        for i in range(nz):
            rs = stats.pearsonr(zhat[i, :], z[i, :])  # bb tiene dos salidas: la primera es corre y la segunda es p-value que es el nivel de confianza
            # asociado al valor de la correlación tras aplicar un test-t
            r_z_zhat_s[i] = rs[0]
            p_z_zhat_s[i] = rs[1]
        # rs = np.zeros(nz)
        # rs_sig = np.zeros(nz)
        # rmse = np.zeros(nz)

        return CrossvalidationOut(
            zhat=zhat,  # Hindcast of field to predict using crosvalidation
            scf=scf,  # Squared covariance fraction of the mca for each mode
            r_z_zhat_t=r_z_zhat_t,  # Correlation between zhat and Z for each time (time series)
            p_z_zhat_t=p_z_zhat_t,  # P values of rt
            r_z_zhat_s=r_z_zhat_s,  # Correlation between time series (for each point) of zhat and z (map)
            p_z_zhat_s=p_z_zhat_s,  # P values of rr
            r_uv=r_uv,  # Correlation score betweeen u and v for each mode
            p_uv=p_uv,  # P value of ruv
            us=us,  # crosvalidated year on axis 2
            alpha=alpha,  # Correlation factor
        )
