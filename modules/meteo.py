import pandas as pd
import xarray as xr
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from scipy import stats
import numpy.typing as npt


NAN_VAL =1e4


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
    scf: List[float]


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
        nyr = int(nt / nmes)

        # first you calculate the covariance matrix
        c = np.nan_to_num(np.dot(y, np.transpose(z)), nan=NAN_VAL)
        r, d, q = np.linalg.svd(c)

        # y había que transponerla si originariamente era (espacio, tiempo), pero ATN_e es (tiempo, espacio) así
        # que no se transpone
        u = np.dot(np.transpose(y), r[:, 1:1 + nm])
        # u = np.dot(np.transpose(y), r[:, :nm])
        # calculamos las anomalías estandarizadas
        v = np.dot(np.transpose(z), q[:, 1:1 + nm])
        # v = np.dot(np.transpose(z), q[:, :nm])

        out = MCAOut(
            RUY=np.ma.empty([nz, nm]),
            RUY_sig=np.ma.empty([nz, nm]),
            SUY=np.ma.empty([nz, nm]),
            SUY_sig=np.ma.empty([nz, nm]),
            RUZ=np.ma.empty([nz, nm]),
            RUZ_sig=np.ma.empty([nz, nm]),
            SUZ=np.ma.empty([nz, nm]),
            SUZ_sig=np.ma.empty([nz, nm]),
            Us=((u - u.mean(0)) / u.std(0)).transpose(),  # Standarized anom across axis 0
            Vs=((v - v.mean(0)) / v.std(0)).transpose(),  # Standarized anom across axis 0
            scf=d / np.sum(d),
        )
        pvalruy = np.ma.empty([ny, nm])
        pvalruz = np.ma.empty([nz, nm])
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
        Cor = np.ma.empty([n1, ])
        Pvalue = np.ma.empty([n1, ])
        for nn in range(n1):  # para cada punto del espacio hacemos la correlación de Pearson
            bb = stats.pearsonr(data[nn, :],
                                index)  # bb tiene dos salidas: la primera es corre y la segunda es p-value que es el nivel de confianza
            # asociado al valor de la correlación tras aplicar un test-t
            Cor[nn] = bb[0]
            Pvalue[nn] = bb[1]
        # generamos una variable que es para que no se muestren mas que los valores de Cor cuando la correlacion
        # es significativa
        Cor_sig = np.ma.masked_where(Pvalue > alpha, Cor)
        # generamos el mapa de regresión mediante multiplicación matricial. Ojo con las dimensiones!!
        reg = data.dot(index) / (n2 - 1)
        # igualmente, hacemos una máscara para que sólo se muestre el mapa de regresión cuando es significativo
        reg_sig = np.ma.masked_where(Pvalue > alpha, reg)
        return Cor, Pvalue, Cor_sig, reg, reg_sig

