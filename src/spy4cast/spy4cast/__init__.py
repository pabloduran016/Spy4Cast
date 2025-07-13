from .mca import MCA, index_regression
from .preprocess import Preprocess
from .crossvalidation import Crossvalidation
from .validation import Validation
import numpy as np

__all__ = [
    'MCA',
    'Preprocess',
    'Crossvalidation',
    'Validation',
    'index_regression',
    'PCA_reduce_dimensionality',
]


def PCA_reduce_dimensionality(x: Preprocess, pca_modes: int = 10) -> Preprocess:
    """ 
    Returns a new Preprocess instance that has reduced dimensionality with 
    PCA based dimensionality reduction

    Paramenters
    -----------

    x : Preprocess
        Data to reduce the dimensionality
    pca_modes : int, default=10
        Number of modes that the PCA will use

    Returns
    -------
        Preprocess
        
    Example
    -------

    >>> z_ds = Dataset("HadISST_sst_chopped.nc", "./datasets").open("sst")
    >>> z_ds = z_ds.slice(
    ...     Region(lat0=-30, latf=30, lon0=-200, lonf=-60, month0=Month.DEC, monthf=Month.FEB,  
    ...            # year0 and yearf refer to monthf so the slice will span from DEC 1970 to FEB 2020 
    ...            year0=1971, yearf=2020),
    ...     skip=1  
    ... )
    >>> z = PCA_reduce_dimensionality(Preprocess(z_ds))
    >>> mca = MCA(y, z, 3, .1)
    """
    mca = MCA(x, x, nm=pca_modes, alpha=.1)
    x_r = x.copy(data=np.dot(mca.SUY, mca.Us))
    return x_r

