import os
import sys
import traceback
from datetime import datetime
from typing import Optional, TypeVar, cast, Tuple

import numpy as np
import pandas as pd

from ._functions import mon2str
from .errors import DatasetError, DatasetNotFoundError, VariableSelectionError, TimeBoundsSelectionError, \
    SelectedYearError
from .stypes import *
import xarray as xr
import numpy.typing as npt


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
        ...     'example.nc', 'data/'
        ... ).open('var').slice(
        ...     Slise(-20, 20, -10, 0, Month.JAN, Month.FEB, 1870, 2000)
        ... ).save_nc('data-sliced.nc')
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

    @classmethod
    def from_xrarray(cls, array: xr.DataArray) -> 'Dataset':
        ds = cls.__new__(cls)
        ds._data = array
        for var in ('lat', 'latitude'):
            if var in array.dims:
                ds._lat_key = var
                break
        for var in ('lon', 'longitude'):
            if var in array.dims:
                ds._lon_key = var
                break
        for var in ('year', 'time'):
            if var in array.dims:
                ds._time_key = var
                break
        return ds

    @property
    def data(self) -> xr.DataArray:
        """
        Returns the data contained
        """
        if not hasattr(self, '_data'):
            raise ValueError('Dataset has not been opened yet')
        return self._data

    @data.setter
    def data(self, val: xr.DataArray) -> None:
        self._data = val

    @property
    def time(self) -> xr.DataArray:
        """Returns the time variable of the data evaluated.
        They key used is recognised automatically
        """

        return self.data[self._time_key]

    @property
    def lat(self) -> xr.DataArray:
        """Returns the latitude variable of the data evaluated.
        They key used is recognised automatically
        """

        return self.data[self._lat_key]

    @property
    def lon(self) -> xr.DataArray:
        """Returns the latitude variable of the data evaluated.
        They key used is recognised automatically
        """

        return self.data[self._lon_key]

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
        return cast(Tuple[int, ...], self.data.shape)

    @property
    def timestamp0(self) -> TimeStamp:
        """Initial timestamp of the data"""
        return self.data.indexes[self._time_key][0]

    @property
    def timestampf(self) -> TimeStamp:
        """Final timestamp of the data"""
        return self.data.indexes[self._time_key][-1]

    @property
    def slise(self) -> Slise:
        """
        Returns the actual slise of the data
        """
        if not hasattr(self, '_slise'):
            self._slise = Slise(
                lat0=self.lat.values[0],
                latf=self.lat.values[-1],
                lon0=self.lon.values[0],
                lonf=self.lon.values[-1],
                month0=Month(self.timestamp0.month),
                monthf=Month(self.timestampf.month),
                year0=self.timestamp0.year,
                yearf=self.timestampf.year,
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

                try:
                    ts0 = datetime.strptime(
                        self._ds.time.attrs['units'].split()[2],
                        '%Y-%m-%d'
                    )
                except ValueError:
                    # This may occur in some datasets with the time variable starting from year 0
                    values = self._ds.time.attrs['units'].split()[2].split('-')
                    if len(values) != 3:
                        raise
                    year, month, day = map(int, values)
                    if year < 1667:
                        print(f"[WARNING] Can not load dataset with initial year being {year}. "
                              "Loading it with year=2000 so keep it in mind for slises", file=sys.stderr)
                        year = 2000
                    ts0 = datetime(year=year, month=month, day=day)

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

        self._detect_vars()
        self._roll_lon()

        # Check if values are in Kelvin
        if self._ds.variables[self.var].attrs['units'] == 'K':
            self.data = getattr(self._ds, self.var) - 273.15
        else:
            self.data = getattr(self._ds, self.var)

        # Fill nan
        if self.data.attrs.get('missing_value') is not None:
            self.data = self.data.where(
                lambda e: e != self.data.attrs['missing_value']
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
        if not hasattr(self, '_var'):
            if len(d_keys) == 0:
                raise ValueError(
                    f'Could not detect variable for dtaset with name: {self.name}'
                )

            self._var = d_keys[0]

        if self.var not in self._ds.variables:
            raise VariableSelectionError(
                f'{self.var}', valid_vars=d_keys
            )

        self._time_key = 'time'
        if self._time_key not in self._ds.variables:
            raise DatasetError(
                f'Could not detect time key in dataset variables '
                f'({", ".join(str(e) for e in self._ds.variables.keys())})'
            )

    def slice(self: _T, slise: Slise, skip: int = 0) -> _T:
        """Method that slices the dataset accorging to a Slise.

        Args
        ----
            slise : Slise
                Slise to use
            skip : int
                Amount of points to skip in the matrix

        Note
        ----
            It first calls `check_slise` method

        Note
        ----
            If the season contains months from different years
            (NOV-DEC-JAN-FEB for example) the initial year is applied
            to the month which comes at last (FEB). In this example, the
            data that will be used for NOV is on year before the initial
            year so keep this in mind if your dataset doesn't contain
            that specific year.

        See Also
        --------
        stypes.Slise
        """

        self._check_slise(slise)
        self._slise = slise

        # Time slise
        fro = pd.to_datetime(self.time.values[0])
        to = fro + pd.DateOffset(months=len(self.time))
        time = pd.date_range(start=fro, end=to, freq='M')
        if len(time) == len(self.time) + 1:
            time = time[:-1]

        if slise.month0 <= slise.monthf:
            timemask = (
                    (time.month >= slise.month0) &
                    (time.month <= slise.monthf) &
                    (time.year >= slise.year0) &
                    (time.year <= slise.yearf)
            )
        else:
            timemask = (
                    (
                            (time.month >= slise.month0) &
                            (time.year >= (slise.year0 - 1)) &
                            (time.year <= (slise.yearf - 1))
                    ) | (
                            (time.month <= slise.monthf) &
                            (time.year >= slise.year0) &
                            (time.year <= slise.yearf)
                    )
            )

        # Space slise
        latmask = (self.lat >= slise.lat0) & (self.lat <= slise.latf)
        lonmask = (self.lon >= slise.lon0) & (self.lon <= slise.lonf)

        self.data = self.data[{
            self._time_key: timemask,
            self._lat_key: latmask,
            self._lon_key: lonmask,
        }]

        latskipmask: npt.NDArray[np.bool_] = np.zeros(
            len(self.lat)
        ).astype(bool)

        latskipmask[::skip + 1] = True

        lonskipmask: npt.NDArray[np.bool_] = np.zeros(
            len(self.lon)
        ).astype(bool)

        lonskipmask[::skip + 1] = True

        self.data = self.data[{
            self._lat_key: latskipmask,
            self._lon_key: lonskipmask,
        }]

        return self

    def _check_slise(
            self: _T, slise: Slise
    ) -> _T:
        """Checks if the slise selected (only time-related part),
        if provided, is valid for the given dataset.

        Args
        ----
            slise : Slise
                Slise use for slicing (see `stypes.Slise`)

        Raises
        ------
            ValueError
                if the dataset ha not been loaded
            VariableSelectionError
                if the variable selected is not valid
            TimeBoundsSelectionError
                if the time slise is not valid
            SelectedYearError
                if the selected_year (if provided) is not valid

        See Also
        --------
        stypes.Slise
        """
        if not hasattr(self, '_data'):
            raise ValueError(
                'The dataset has not been loaded yet. Call load_dataset()'
            )

        assert type(slise.year0) == int, \
            f'Invalid type for initial_year: {type(slise.year0)}'
        if slise.year0 > self.timestampf.year:
            raise TimeBoundsSelectionError(
                f"Initial year not valid. Dataset finishes in "
                f"{self.timestampf.year}, got {slise.year0} as "
                f"initial year"
            )
        if slise.year0 < self.timestamp0.year:
            raise TimeBoundsSelectionError(
                f"Initial year not valid. Dataset starts in "
                f"{self.timestamp0.year}, got {slise.year0}"
            )
        assert type(slise.yearf) == int,\
            f"Invalid type for final_year: {type(slise.yearf)}"
        if slise.yearf > self.timestampf.year:
            raise TimeBoundsSelectionError(
                f"Final Year out of bounds. Dataset finishes in "
                f"{self.timestampf.year}, got {slise.yearf}"
            )
        if slise.yearf < self.timestamp0.year:
            raise TimeBoundsSelectionError(
                f"Final year not valid. Dataset starts in "
                f"{self.timestamp0.year}, got {slise.year0}"
            )
        assert type(slise.yearf) == int,\
            "Invalid type for final_year: %s" % type(slise.yearf)
        assert type(slise.monthf) == int or type(slise.monthf) == Month, \
            "Invalid type for final_month: %s" % type(slise.monthf)
        if slise.yearf >= self.timestampf.year and \
                slise.monthf > self.timestampf.month:
            raise TimeBoundsSelectionError(
                f"Final Month out of bounds. Dataset finishes in "
                f"{mon2str(Month(self.timestampf.month))} "
                f"{self.timestampf.year}, got "
                f"{mon2str(Month(slise.monthf))} {slise.yearf}"
            )
        assert type(slise.yearf) == int, \
            "Invalid type for final_year: %s" % type(slise.yearf)
        assert type(slise.year0) == int,\
            f"Invalid type for initial_year: {type(slise.year0)}"
        if slise.year0 > slise.yearf:
            raise TimeBoundsSelectionError(
                f"Initial year bigger than final year\n"
                f'NOTE: initial_year={slise.year0}, '
                f'final_year={slise.yearf}'
            )
        assert type(slise.month0) == int or type(slise.month0) == Month, \
            f"Invalid type for initial_month: {type(slise.month0)}"
        if not 1 <= slise.month0 <= 12:
            raise TimeBoundsSelectionError(
                'Initial month not valid, must be int from 0 to 11'
            )
        assert type(slise.monthf) == int or type(slise.monthf) == Month, \
            "Invalid type for final_month: %s" % type(slise.monthf)
        if not 1 <= slise.monthf <= 12:
            raise TimeBoundsSelectionError(
                'Final month not valid, must be int from 0 to 11'
            )
        if slise.month0 > slise.monthf and \
                slise.year0 - 1 < self.timestamp0.year:
            raise TimeBoundsSelectionError(
                f'Initial year not valid, remember that when selecting '
                f'month slice that combines years, the initial year '
                f'backtracks one unit\n'
                f'NOTE: dataset initial timestamp : {self.timestamp0}'
            )
        if slise.sy is not None and slise.sy != 0:
            if not slise.year0 <= slise.sy <= slise.yearf:
                raise SelectedYearError(slise.sy)
        return self

    def save_nc(self: _T, name: str, dir: str = '.') -> _T:
        """Saves the data as a netcdf4 file
        """

        fig_data_path = os.path.join(dir, name)

        # REMOVES NAN VALUES TO PREVENT ERRORS
        if self._lat_key in self.data.dims and \
                self._lon_key in self.data.dims:
            to_save = self.data[{
                self._lat_key: ~np.isnan(self.lat),
                self._lon_key: ~np.isnan(self.lon)
            }]
        else:
            to_save = self.data[~np.isnan(self.data)]

        self._ds.assign_coords({
            self.var: to_save
        }).to_netcdf(fig_data_path)

        return self
