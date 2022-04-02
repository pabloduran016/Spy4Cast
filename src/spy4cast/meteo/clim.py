from .._procedure import _Procedure
import xarray as xr
import pandas as pd


class Clim(_Procedure):
    '''
        """A proker and a plotter that performs the climatology
    of a variable on a time series

    See Also
    --------
    ClimerMap, AnomerTS
    """

    def apply(self, **kw: Any) -> 'ClimerTS':
        """Method that applies the climatology.
        Does not accept any keyword arguments

        See Also
        --------
        spy4cast.meteo.clim
        """
        if len(kw) != 0:
            raise TypeError('`apply` does not accept any keyword arguments')
        self._data = self._data.mean(dim=self._lon_key).mean(dim=self._lat_key)
        self._data = clim(self._data, dim='month')
        self._time_key = 'year'
        self._plot_data = 'CLIM ' + self._plot_data

    '''

    '''
        """A proker and a plotter that performs the
    climatology of a variable on a map

    See Also
    --------
    ClimerTS, AnomerMap
    """

    def apply(self, **kw: Any) -> 'ClimerMap':
        """Method that applies the climatology.
        Does not accept any keyword arguments

        See Also
        --------
        spy4cast.meteo.clim
        """
        if len(kw) != 0:
            raise TypeError(
                '`apply` does not accept any keyword arguments'
            )
        self._data = clim(self._data)
        self._plot_data = 'CLIM ' + self._plot_data
        return self
'''


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
