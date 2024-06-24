from copy import copy
import os
import traceback
import warnings
from datetime import datetime
from typing import Optional, TypeVar, cast, Tuple

import numpy as np
import pandas as pd

from ._functions import mon2str, _error, _warning, _debuginfo
from .errors import DatasetError, DatasetNotFoundError, VariableSelectionError, TimeBoundsSelectionError, \
    SelectedYearError
from .stypes import *
import xarray as xr
import numpy.typing as npt


warnings.filterwarnings('ignore')


__all__ = [
    'Dataset',
]


_INVALID_VARS = {
    'lon', 'longitude', 'lon_bnds', 'lat', 'latitude', 'lat_bnds',
    'time', 'time_bnds', 'average_DT'
}
_T = TypeVar('_T', bound='Dataset')


class Dataset:
    """This class enables you to load, slice and save the data in a
    netcdf4 file confortably.

    It is concatenable, meaning that the output of most methods is the object
    itself so that you can concatenate methods easily.

    Args
    ----
        name : str, default=''
            Directory where the dataset you want to use is located
        folder : str, default='dataset.nc'
            Name of the dataset
        chunks : `stypes.ChunkType`, optional
            Argument passed when loading the datasets
            (see `chunks` in dask library)

    Example
    -------
        >>> Dataset(
        ...     'example.nc', 'data/'
        ... ).open('var').slice(
        ...     Region(-20, 20, -10, 0, Month.JAN, Month.FEB, 1870, 2000)
        ... ).save_nc('data-sliced.nc')
    """

    _data: xr.DataArray
    _ds: xr.Dataset
    _lat_key: str
    _lon_key: str
    _time_key: str
    _region: Region
    _var: str

    def __init__(
        self,
        name: str,
        folder: str = '.',
        chunks: Optional[ChunkType] = None
    ):
        self.name: str = name
        self.folder: str = folder
        self._chunks: Optional[ChunkType] = chunks

    @classmethod
    def from_xrarray(cls, array: xr.DataArray) -> 'Dataset':
        """Create a dataset from a xarray"""
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
        t0 = self.data[self._time_key].values[0]
        try:
            return cast(TimeStamp, pd.to_datetime(t0))
        except TypeError:
            _warning('Could not convert initial timestamp to pandas TimeStamp')
            return cast(TimeStamp, pd.Timestamp(str(t0)))

    @property
    def timestampf(self) -> TimeStamp:
        """Final timestamp of the data"""
        tf = self.data[self._time_key].values[-1]
        try:
            return cast(TimeStamp, pd.to_datetime(tf))
        except TypeError:
            _warning('Could not convert final timestamp to pandas TimeStamp')
            return cast(TimeStamp, pd.Timestamp(str(tf)))

    def _detect_region(self) -> Region:
        """
        Detect region
        """
        return Region(
            lat0=self.lat.values[0],
            latf=self.lat.values[-1],
            lon0=self.lon.values[0],
            lonf=self.lon.values[-1],
            month0=Month(self.timestamp0.month),
            monthf=Month(self.timestampf.month),
            year0=self.timestamp0.year,
            yearf=self.timestampf.year,
        )

    @property
    def region(self) -> Region:
        """
        Returns the actual region of the data
        """
        if not hasattr(self, '_region'):
            self._region = self._detect_region()
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        new_region = copy(region)
        while new_region.lon0 > 180:
            new_region.lon0 -= 360
        while new_region.lonf > 180:
            new_region.lonf -= 360
        while new_region.lon0 < -180:
            new_region.lon0 += 360
        while new_region.lonf < -180:
            new_region.lonf += 360

        self._region = new_region


    def open(self: _T, var: Optional[str] = None, region: Optional[Region] = None) -> _T:
        """Opens dataset without loading it into memory

        .. warning::

            If the dataset says that the dataset starts from a year smaller than 1678
            (minimum for pandas), it will be loaded with initial year = 2000

        Parameters
        ----------
            var : optional, str
                Variable that can be specified. If not it is detected automatically
            region: optional, Region
                Fill in this field the dataset is already sliced and you want to specify a certain region

        Raises
        ------
            errors.DatasetError
                If there is an error while opening the dataset
            errors.DatasetNotFoundError
                If the dataset name or folder is not valid
            errors.VariableSelectionError
                If teh variable selected does not exist or can not be inferred
        """
        if hasattr(self, '_data'):
            return self

        try:
            self._ds = xr.open_dataset(
                os.path.join(self.folder, self.name),
                mask_and_scale=True,
                chunks=self._chunks
            )
        except ValueError:
            try:
                self._ds = xr.open_dataset(
                    os.path.join(self.folder, self.name),
                    mask_and_scale=True,
                    decode_times=False,
                    chunks=self._chunks
                )

                # TODO: Some datasets may not even have the 'units' attr
                date_type = self._ds.time.attrs['units'].split()[0]
                if date_type != 'months':
                    # TODO: Look for a dataset that suffers from this. To add into the tests
                    raise ValueError(f'Datasets can only be loaded without decoded times if they are monthly data, got: {date_type}')

                try:
                    ts0 = datetime.strptime(
                        self._ds.time.attrs['units'].split()[2],
                        '%Y-%m-%d'
                    )
                except ValueError:
                    # TODO: Look for a dataset that suffers from this. To add into the tests
                    # This may occur in some datasets with the time variable starting from year 0
                    values = self._ds.time.attrs['units'].split()[2].split('-')
                    if len(values) != 3:
                        raise
                    year, month, day = map(int, values)
                    if year < 1678:
                        _warning(
                            f"Can not load dataset with initial year being {year}. "
                            "Loading it with year=2000 so keep it in mind for regions"
                        )
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
                _error(
                    f"<{self.__class__.__name__}> Could not load dataset {self.name}"
                )
                raise DatasetError from e
        except FileNotFoundError as e:
            raise DatasetNotFoundError from e

        if var is not None:
            self._var = var

        self._detect_vars()

        if region is not None:
            self.region = region
        else:
            self.region = Region(
                lat0=self._ds[self._lat_key].values[0],
                latf=self._ds[self._lat_key].values[-1],
                lon0=self._ds[self._lon_key].values[0],
                lonf=self._ds[self._lon_key].values[-1],
                month0=Month(pd.to_datetime(self._ds[self._time_key].values[0]).month),
                monthf=Month(pd.to_datetime(self._ds[self._time_key].values[-1]).month),
                year0=pd.to_datetime(self._ds[self._time_key].values[0]).year,
                yearf=pd.to_datetime(self._ds[self._time_key].values[-1]).year,
            )

        self._roll_lon()

        # Check if values are in Kelvin
        if 'units' in self._ds.variables[self.var].attrs and self._ds.variables[self.var].attrs['units'] == 'K':
            # TODO: Look for a dataset that suffers from this. To add into the tests
            self.data = getattr(self._ds, self.var) - 273.15
        else:
            self.data = getattr(self._ds, self.var)

        # Fill nan
        if self.data.attrs.get('missing_value') is not None:
            self.data = self.data.where(
                lambda x: x != self.data.attrs['missing_value']
            )

        self.data = self.data.assign_coords(year=('time', self.data['time.year'].data))

        return self

    def _roll_lon(self) -> None:
        """
        Roll longitude if it is from 0 to 360
        """
        if self._ds[self._lon_key].max() > 180:
            self._ds = self._ds.assign_coords({
                self._lon_key: (
                    ((self._ds[self._lon_key] + 180) % 360) - 180
                )
            }).sortby(self._lon_key)

    def _detect_vars(self) -> None:
        """
        Detect variables in dataset
        """
        ds_dims = self._ds.dims
        not_found_lon = False
        if 'longitude' in ds_dims:
            self._lon_key = 'longitude'
        elif 'lon' in ds_dims:
            self._lon_key = 'lon'
        else:
            not_found_lon = True

        not_found_lat = False
        if 'latitude' in ds_dims:
            self._lat_key = 'latitude'
        elif 'lat' in ds_dims:
            self._lat_key = 'lat'
        else:
            not_found_lat = True

        '''
        # There are some datasets that sore latitude
        # and longitude in other datavars that are accessed by an
        # index in the dimension i and j. We solve this by asserting
        # that the columns in the longitude matrix are the same and the
        # rows in the latitude matrix too and create new coordinates that
        # follow the usual expected structure
        if not_found_lat and not_found_lon and (
            'i' in ds_dims and
            'j' in ds_dims and
            'latitude' in self._ds.variables and
            'longitude' in self._ds.variables
        ):
            lon = self._ds.longitude
            lat = self._ds.latitude
            assert all((lat[:, col] == lat[:, col + 1]).all() for col in range(lat.shape[1] - 1))
            assert all((lon[row, :] == lon[row + 1, :]).all() for row in range(lon.shape[0] - 1))
            # i: longitude
            # j: longitude
            self._ds = self._ds.assign_coords({
                'unidimensional_latitude': lat[:, 0],
                'unidimensional_longitude': lon[0, :]
            })
            self._lat_key = 'unidimensional_latitude'
            self._lon_key = 'unidimensional_longitude'
        '''
        if not_found_lon:
            raise DatasetError(
                f'Can\'t recognise dataset longitude dimension key: '
                f'{self.name}\n'
                f'NOTE: dims={ds_dims}'
            )
        elif not_found_lat:
            raise DatasetError(
                f'Can\'t recognise dataset latitude dimension key: '
                f'{self.name}\n'
                f'NOTE: dims={ds_dims}'
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
            _debuginfo(f'Detected variable {self._var}')

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

    def slice(self: _T, region: Region, skip: int = 0) -> _T:
        """Method that slices the dataset accorging to a Region.

        Args
        ----
            region : spy4cast.stypes.Region
                spy4cast.stypes.Region to use
            skip : int
                Amount of points to skip in the matrix

        Note
        ----
            It first calls `check_region` method

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
        stypes.Region
        """

        self.region = region

        self._check_region(region)

        # Time region
        fro = self.timestamp0
        to = fro + pd.DateOffset(months=len(self.time))
        time = pd.date_range(start=fro, end=to, freq='M')
        if len(time) == len(self.time) + 1:
            time = time[:-1]

        if region.month0 <= region.monthf:
            timemask = (
                    (time.month >= region.month0) &
                    (time.month <= region.monthf) &
                    (time.year >= region.year0) &
                    (time.year <= region.yearf)
            )
        else:
            timemask = (
                    (
                        (time.month >= region.month0)     &
                        (time.year >= (region.year0 - 1)) &
                        (time.year <= (region.yearf - 1))
                    ) | (
                        (time.month <= region.monthf) &
                        (time.year >= region.year0)   &
                        (time.year <= region.yearf)
                    )
            )

        # Space region
        if region.lat0 > region.latf:
            latmask = ~((self.lat >= region.latf) & (self.lat <= region.lat0))
        else:
            latmask = (self.lat >= region.lat0) & (self.lat <= region.latf)
        if region.lon0 > region.lonf:
            lonmask = ~((self.lon >= region.lonf) & (self.lon <= region.lon0))
        else:
            lonmask = (self.lon >= region.lon0) & (self.lon <= region.lonf)

        self.data = self.data[{
            self._time_key: timemask,
            self._lat_key: latmask,
            self._lon_key: lonmask,
        }]

        if region.month0 <= region.monthf:
            season_size = region.monthf - region.month0 + 1
        else:
            season_size = region.monthf + 13 - region.month0  # region.month0 != region.monthf

        self.data = self.data.assign_coords(year=(
            'time', [year for year in range(region.year0, region.yearf + 1) for _ in range(season_size)]))

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

    def _check_region(
        self: _T, region: Region
    ) -> _T:
        """Checks if the region selected (only time-related part),
        if provided, is valid for the given dataset.

        Args
        ----
            region : Region
                Region use for slicing (see `stypes.Region`)

        Raises
        ------
            ValueError
                if the dataset ha not been loaded
            VariableSelectionError
                if the variable selected is not valid
            TimeBoundsSelectionError
                if the time region is not valid
            SelectedYearError
                if the selected_year (if provided) is not valid

        See Also
        --------
        stypes.Region
        """
        if not hasattr(self, '_data'):
            raise ValueError(
                'The dataset has not been opened yet. Call .open()'
            )

        assert type(region.year0) == int, \
            f'Invalid type for initial_year: {type(region.year0)}'
        assert type(region.yearf) == int, \
            f"Invalid type for final_year: {type(region.yearf)}"
        assert type(region.monthf) == int or type(region.monthf) == Month, \
            "Invalid type for final_month: %s" % type(region.monthf)
        assert type(region.month0) == int or type(region.month0) == Month, \
            f"Invalid type for initial_month: {type(region.month0)}"
        if not 1 <= region.month0 <= 12:
            raise TimeBoundsSelectionError(
                'Initial month not valid, must be int from 0 to 11'
            )
        if not 1 <= region.monthf <= 12:
            raise TimeBoundsSelectionError(
                'Final month not valid, must be int from 0 to 11'
            )
        if region.year0 > self.timestampf.year:
            raise TimeBoundsSelectionError(
                f"Initial year not valid. Dataset finishes in "
                f"{self.timestampf.year}, got {region.year0} as "
                f"initial year"
            )
        if region.year0 < self.timestamp0.year:
            raise TimeBoundsSelectionError(
                f"Initial year not valid. Dataset starts in "
                f"{self.timestamp0.year}, got {region.year0}"
            )
        if region.yearf > self.timestampf.year:
            raise TimeBoundsSelectionError(
                f"Final Year out of bounds. Dataset finishes in "
                f"{self.timestampf.year}, got {region.yearf}"
            )
        if region.yearf < self.timestamp0.year:
            raise TimeBoundsSelectionError(
                f"Final year not valid. Dataset starts in "
                f"{self.timestamp0.year}, got {region.year0}"
            )
        if region.yearf >= self.timestampf.year and \
                region.monthf > self.timestampf.month:
            raise TimeBoundsSelectionError(
                f"Final Month out of bounds. Dataset finishes in "
                f"{mon2str(Month(self.timestampf.month))} "
                f"{self.timestampf.year}, got "
                f"{mon2str(Month(region.monthf))} {region.yearf}"
            )
        if region.year0 == self.timestamp0.year and \
                region.month0 < self.timestamp0.month:
            raise TimeBoundsSelectionError(
                f"Initial Month out of bounds. Dataset starts in "
                f"{mon2str(Month(self.timestamp0.month))} "
                f"{self.timestamp0.year}, got "
                f"{mon2str(Month(region.month0))} {region.year0}"
            )
        if region.year0 > region.yearf:
            raise TimeBoundsSelectionError(
                f"Initial year bigger than final year\n"
                f'NOTE: initial_year={region.year0}, '
                f'final_year={region.yearf}'
            )
        if region.month0 > region.monthf and \
                region.year0 - 1 < self.timestamp0.year:
            raise TimeBoundsSelectionError(
                f'Initial year not valid, remember that when selecting '
                f'month slice that combines years, the initial year '
                f'backtracks one unit\n'
                f'NOTE: dataset initial timestamp : {self.timestamp0}'
            )
        if region.sy is not None and region.sy != 0:
            if not region.year0 <= region.sy <= region.yearf:
                raise SelectedYearError(region.sy)
        return self

    def save_nc(self: _T, name: str, folder: str = '.') -> _T:
        """Saves the data as a netcdf4 file
        """

        fig_data_path = os.path.join(folder, name)

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
