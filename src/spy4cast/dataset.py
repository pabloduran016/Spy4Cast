import os
import sys
import traceback
from datetime import datetime
from typing import Optional, TypeVar, cast, Tuple

import pandas as pd

from .errors import DatasetError, DatasetNotFoundError, VariableSelectionError
from .stypes import *
import xarray as xr


_T = TypeVar('_T', bound='Dataset')


_INVALID_VARS = {
    'lon', 'longitude', 'lon_bnds', 'lat', 'latitude', 'lat_bnds',
    'time', 'time_bnds', 'average_DT'
}


class Dataset:
    """This class enables you to load, slice and save the data in a
    netcdf4 file confortably.

    It is concatenable, meaning that the output of most methods is the object
    itself so that you can concatenate methods easily.

    Args
    ----
        name : str, default=''
            Directory where the dataset you want to use is located
        dir : str, default='dataset.nc'
            Name of the dataset
        chunks : `stypes.ChunkType`, optional
            Argument passed when loading the datasets
            (see `chunks` in dask library)

    Example
    -------
        >>> Dataset(
        ...     'example.nc', 'data/', 'var'
        ... ).open().slice(
        ...     Slise(-20, 20, -10, 0, Month.Jan, Month.FEB, 1870, 2000)
        ... ).save_nc()
    """

    _data: xr.DataArray
    _ds: xr.Dataset
    _lat_key: str
    _lon_key: str
    _time_key: str
    _slise: Slise
    _var: str

    def __init__(
        self,
        name: str,
        dir: str = '.',
        chunks: Optional[ChunkType] = None
    ):
        self.name: str = name
        self.dir: str = dir
        self._chunks: Optional[ChunkType]  = chunks

    @property
    def time(self) -> xr.DataArray:
        """Returns the time variable of the data evaluated.
        They key used is recognised automatically
        """

        return self._data[self._time_key]

    @property
    def lat(self) -> xr.DataArray:
        """Returns the latitude variable of the data evaluated.
        They key used is recognised automatically
        """

        return self._data[self._lat_key]

    @property
    def lon(self) -> xr.DataArray:
        """Returns the latitude variable of the data evaluated.
        They key used is recognised automatically
        """

        return self._data[self._lon_key]

    @property
    def var(self) -> str:
        """Return the variable used.

        Raises
        ------
            ValueError
                If the dataset has not been opened yet
        """
        if not hasattr(self, '_var'):
            raise ValueError('Dataset has not been opened yet')

        return self._var

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape variable of the data evaluated."""
        return cast(Tuple[int, ...], self._data.shape)

    @property
    def timestamp0(self) -> TimeStamp:
        return self._ds.indexes['time'][0]

    @property
    def timestampf(self) -> TimeStamp:
        return self._ds.indexes['time'][-1]

    @property
    def slise(self) -> Slise:
        if not hasattr(self, '_slise'):
            self._slise = Slise(
                lat0=self.lat.values[0],
                latf=self.lat.values[-1],
                lon0=self.lon.values[0],
                lonf=self.lon.values[-1],
                month0=Month(self.timestamp0.month),
                monthf=Month(self.timestampf.month),
                year0=self.timestamp0.year,
                yearf=self.timestampf.year
            )
        return self._slise

    def open(self: _T, var: Optional[str] = None) -> _T:
        """Opens dataset without loading it into memory

        Parameters
        ----------
            var : optional, str
                Variable that can be specified. If not it is detected automatically

        Raises
        ------
            errors.DatasetError
                If there is an error while opening the dataset
            errors.DatasetNotFoundError
                If the dataset name or dir is not valid
            errors.VariableSelectionError
                If teh variable selected does not exist or can not be inferred
        """
        if hasattr(self, '_data'):
            return self

        try:
            self._ds = xr.open_dataset(
                os.path.join(self.dir, self.name),
                mask_and_scale=False,
                chunks=self._chunks
            )
        except ValueError:
            try:
                self._ds = xr.open_dataset(
                    os.path.join(self.dir, self.name),
                    mask_and_scale=False,
                    decode_times=False,
                    chunks=self._chunks
                )

                ts0 = datetime.strptime(
                    self._ds.time.attrs['units'].split()[2],
                    '%Y-%m-%d'
                )
                tsf = ts0 + pd.DateOffset(months=len(self._ds.time))
                self._ds = self._ds.assign_coords(
                    time=pd.date_range(
                        ts0, tsf, freq='M'
                    )
                )
            except Exception as e:
                traceback.print_exc()
                print(
                    f"[ERROR] <{self.__class__.__name__}> Could not load "
                    f"dataset {self.name}",
                    file=sys.stderr
                )
                raise DatasetError from e
        except FileNotFoundError as e:
            raise DatasetNotFoundError from e

        if var is not None:
            self._var = var

        self._roll_lon()
        self._detect_vars()

        # Check if values are in Kelvin
        if self._ds.variables[self.var].attrs['units'] == 'K':
            self._data = getattr(self._ds, self.var) - 273.15
        else:
            self._data = getattr(self._ds, self.var)

        # Fill nan
        if self._data.attrs.get('missing_value') is not None:
            self._data = self._data.where(
                lambda e: e != self._data.attrs['missing_value']
            )

        return self

    def _roll_lon(self) -> None:
        """
        Roll longitude if it iis from 0 to 360
        """
        if max(self._ds[self._lon_key]) > 180:
            self._ds = self._ds.assign_coords({
                self._lon_key: (
                    ((self._ds[self._lon_key] + 180) % 360) - 180
                )
            })

    def _detect_vars(self) -> None:
        """
        Detect variables in dataset
        """
        if 'longitude' in self._ds.variables:
            self._lon_key = 'longitude'
        elif 'lon' in self._ds.variables:
            self._lon_key = 'lon'
        else:
            raise DatasetError(
                f'Can\'t recognise dataset longitude variable key: '
                f'{self.name}\n'
                f'NOTE: variables={self._ds.variables}'
            )

        if 'latitude' in self._ds.variables:
            self._lat_key = 'latitude'
        elif 'lat' in self._ds.variables:
            self._lat_key = 'lat'
        else:
            raise DatasetError(
                f'Can\'t recognise dataset latitude variable key: '
                f'{self.name}\n'
                f'NOTE: variables={self._ds.variables}'
            )


        d_keys = [
            str(e) for e in self._ds.variables.keys() if str(e) not in _INVALID_VARS
        ]
        if not hasattr(self, 'var'):
            if len(d_keys) == 0:
                raise ValueError(
                    f'Could not detect variable for dtaset with name: {self.name}'
                )

            self._var = d_keys[0]

        if self.var not in self._ds.variables:
            raise VariableSelectionError(
                f'{self.var}', valid_vars=d_keys
            )


    def slice(self: _T, slise: Slise) -> _T:
        raise NotImplementedError

    def save_nc(self: _T) -> _T:
        raise NotImplementedError
