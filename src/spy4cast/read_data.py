import datetime
import os
import sys
import traceback
from typing import ClassVar, Set, Optional, TypeVar, Tuple, Dict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from .stypes import TimeStamp, Month, Slise, ChunkType
from .errors import VariableSelectionError, TimeBoundsSelectionError,\
    SelectedYearError, DatasetNotFoundError, DatasetError
from ._functions import mon2str, slise2str, debugprint

__all__ = ['ReadData', 'NAN_VAL']

NAN_VAL = 1e10  # Absolute value from where a point is considered NaN

ReadDataType = TypeVar('ReadDataType', bound='ReadData')
"""Type returned by each ReadData function that enables concatenability
See Also
--------
    `ReadData`
"""


class ReadData:
    """This class enables you to load, slice and modify the data in a netcdf4 file confortably.

    It is concatenable, meaning that the output of most methods is the object itself so that you can
    concatenate methods easily.

    Args
    ----
        dataset_dir : str, default=''
            Directory where the dataset you want to use is located
        dataset_name : str, default='dataset.nc'
            Name of the dataset
        variable : str, optional
            Variable to evaluate. If it is not set, the program will try to recgnise it by discarding time and coordinate variables
        plot_dir : str, default=''
            Directory to store the plot if later created
        plot_name : str, default='plot.png'
            Name of the plot saved if later created
        plot_data_dir : str, default=''
            Directory of the data saved if later saved
        force_name : bool, default=False
            Indicates wether or not inforce the names set above. If false the name will be modified not to overwrite any existing files
        chunks : `stypes.ChunkType`, optional
            Argument passed when loading the datasets (see chunks in dask library)

    Example
    -------
        >>> ReadData('data/', 'example.nc', 'var').open_dataset().slice_dataset(Slise(-20, 20, -10, 0, Month.Jan, Month.FEB, 1870, 2000)).save_fig_data()
    """
    _silent: ClassVar[bool] = False

    _dataset_initial_timestamp: TimeStamp
    _dataset_final_timestamp: TimeStamp

    _dataset: xr.Dataset
    _data: xr.DataArray
    _slise: Slise
    # _dataset_min_value: float
    # _dataset_max_value: float

    def __init__(self, dataset_dir: Optional[str] = None, dataset_name: Optional[str] = None,
                 variable: Optional[str] = None, plot_dir: Optional[str] = None,
                 plot_name: Optional[str] = None, plot_data_dir: Optional[str] = None,
                 force_name: bool = False, chunks: Optional[ChunkType] = None):
        self._dataset_dir = dataset_dir if dataset_dir is not None else ''
        self._dataset_name = dataset_name if dataset_name is not None else 'dataset.nc'
        self._plot_name = plot_name if plot_name is not None else 'plot.png'
        self._plot_data_name = plot_name if plot_name is not None else 'data.nc'
        self._plot_dir = plot_dir if plot_dir is not None else ''
        self._plot_data_dir = plot_data_dir if plot_data_dir is not None else ''
        self._variable = variable if variable is not None else ''
        self._chunks = chunks

        self._loaded_without_times = False

        self._valid_variables: Set[str] = set()
        self._opened_dataset = False
        self._plot_data = ''
        self._plot_bounds = ''

        # Initialize variables
        self._lon_key: str = 'longitude'
        self._lat_key: str = 'latitude'
        self._time_key: str = 'time'

        if not force_name:
            fig_path = os.path.join(self._plot_dir, self._plot_name)
            self._plot_name, fig_extension = os.path.splitext(self._plot_name)
            i = 1
            while os.path.exists(fig_path):
                fig_path = os.path.join(self._plot_dir, self._plot_name + f'({i})' + fig_extension)
                i += 1
            self._plot_name = os.path.split(fig_path)[-1]

            self._plot_data_name = os.path.splitext(self._plot_name)[0] + '.nc'
            fig_data_path = os.path.join(self._plot_data_dir, self._plot_data_name)
            fig_data_name, fig_data_extension = os.path.splitext(self._plot_data_name)
            i = 1
            while os.path.exists(fig_data_path):
                fig_data_path = os.path.join(self._plot_data_dir, fig_data_name + f'({i})' + fig_data_extension)
                i += 1
            self._plot_data_name = os.path.split(fig_data_path)[-1]

        # debugprint(f"[INFO] <{self.__class__.__name__}> New Read Data object plot_name: {self._plot_name},"
        #            f" plot_data_name: {self._plot_data_name}")

    @property
    def time(self) -> xr.DataArray:
        """Returns the time variable of the data evaluated. They key used is recognised automatically"""
        return self._data[self._time_key]

    @property
    def lat(self) -> xr.DataArray:
        """Returns the latitude variable of the data evaluated. They key used is recognised automatically"""
        return self._data[self._lat_key]

    @property
    def lon(self) -> xr.DataArray:
        """Returns the latitude variable of the data evaluated. They key used is recognised automatically"""
        return self._data[self._lon_key]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape variable of the data evaluated."""
        return cast(Tuple[int, ...], self._data.shape)

    def load_dataset(self: ReadDataType) -> ReadDataType:
        """Loads dataset into memory

        .. deprecated:: 0.0.1
          `ReadData.load_dataset` will be removed in SpyCaster 0.0.2, and replaced by
          `ReadData.open_dataset` because it doesn't load the dataset into memory.

        Warning
        -------
            Deprectaed, use `ReadData.open_dataset`

        See Also
        --------
            `ReadData.open_dataset`
        """
        print('[WARNING] Support for `ReadData.load_dataset` is deprected. Use `ReadData.open_dataset` instead',
              file=sys.stderr)
        if self._opened_dataset:
            return self
        self.open_dataset()
        self._dataset = self._dataset.load()
        return self

    def open_dataset(self: ReadDataType) -> ReadDataType:
        """Opens dataset without loading it into memory

        Raises
        ------
            `errors.DatasetError`
                If there is an error while opening the dataset
            `errors.DatasetNotFoundError`
                If the dataset name or dir is not valid
            `errors.VariableSelectionError`
                If teh variable selected does not exist or can not be inferred
        """
        if self._opened_dataset:
            return self
        # debugprint(f"[INFO] <{self.__class__.__name__}> Loading dataset: {self._dataset_name} for {self._plot_name}")
        try:
            self._dataset = xr.open_dataset(os.path.join(self._dataset_dir, self._dataset_name), mask_and_scale=False, chunks=self._chunks)
        except ValueError:
            try:
                self._dataset = xr.open_dataset(os.path.join(self._dataset_dir, self._dataset_name),
                                                mask_and_scale=False, decode_times=False, chunks=self._chunks)

                initial_timestamp = datetime.datetime.strptime(self._dataset.time.attrs['units'].split()[2],
                                                               '%Y-%M-%d')
                final_timestamp = initial_timestamp + pd.DateOffset(months=len(self._dataset.time))
                self._dataset = self._dataset.assign_coords(
                    time=pd.date_range(initial_timestamp, final_timestamp, freq='M'))
            except Exception as e:
                traceback.print_exc()
                debugprint(f"[ERROR] <{self.__class__.__name__}> Could not load dataset {self._dataset_name}"
                           f" for {self._plot_name}")
                raise DatasetError() from e
        #     self._dataset = xr.load_dataset(os.path.join(self._dataset_dir, self._dataset_name), decode_times=False)
        #     self.loaded_without_times = True
        except FileNotFoundError as e:
            raise DatasetNotFoundError() from e

        # Keys for the DATASET may vary (example: lon, longitude, ...) depending on the DATASET
        self._lon_key = 'longitude' if 'longitude' in self._dataset.variables \
            else 'lon' if 'lon' in self._dataset.variables else 'nan'
        if self._lon_key == 'nan':
            raise DatasetError(f'Cant recognise dataset longitude variable key: {self._dataset_name}')
        self._lat_key = 'latitude' if 'latitude' in self._dataset.variables \
            else 'lat' if 'lat' in self._dataset.variables else 'nan'
        if self._lat_key == 'nan':
            raise DatasetError(f'Cant recognise dataset latitude variable key: {self._dataset_name}')

        if max(self._dataset[self._lon_key]) > 180:
            self._dataset = self._dataset.assign_coords(
                {self._lon_key: (((self._dataset[self._lon_key] + 180) % 360) - 180)})

        d_keys: Set[str] = set(str(e) for e in self._dataset.variables.keys())
        self._valid_variables.update(
            d_keys - {'lon', 'longitude', 'lon_bnds', 'lat', 'latitude', 'lat_bnds', 'time', 'time_bnds', 'average_DT'}
        )

        if self._variable == '':
            for key in self._valid_variables:
                self._variable = key
                break
        elif self._variable not in self._dataset.variables.keys():
            raise VariableSelectionError(f'{self._variable}', valid_variables=list(self._valid_variables))

        # Check if values are in Kelvin
        if self._dataset.variables[self._variable].attrs['units'] == 'K':  # values in Kelvin
            self._data = getattr(self._dataset, self._variable) - 273.15
        else:
            self._data = getattr(self._dataset, self._variable)

        # Fill nan
        if self._data.attrs.get('missing_value') is not None:
            self._data = self._data.where(lambda e: e != self._data.attrs['missing_value'])

        # Establish the initial year of the dataset
        self._dataset_initial_timestamp = self._dataset.indexes['time'][0]
        self._dataset_final_timestamp = self._dataset.indexes['time'][-1]

        self._slise = Slise.default(year0=self._dataset_initial_timestamp.year,
                                    yearf=self._dataset_final_timestamp.year)

        self._plot_data = f'Jan to {mon2str(Month(self._dataset_final_timestamp.month))} ' \
                         f'{self._dataset_initial_timestamp.year}-{self._dataset_final_timestamp.year} '

        self._opened_dataset = True

        return self

    def check_variables(self: ReadDataType, slise: Optional[Slise] = None) -> ReadDataType:
        """Checks if the variable selected and the slise (only time-related part), if provided, is valid for the given dataset.

        Args
        ----
            slise : Slise
                Slise use for slicing (see `stypes.Slise`)

        Raises
        ------
            `ValueError`
                if the dataset ha not been loaded
            `spy4cast.errors.VariableSelectionError`
                if the variable selected is not valid
            `spy4cast.errors.TimeBoundsSelectionError`
                if the time slise is not valid
            `spy4cast.errors.SelectedYearError`
                if the selected_year (if provided) is not valid

        See Also
        --------
            `stypes.Slise`
        """
        # debugprint(f"[INFO] <{self.__class__.__name__}> Checking variables for {self._plot_name}")
        if not self._opened_dataset:
            raise ValueError('The dataset has not been loaded yet. Call load_dataset()')
        if self._variable not in self._valid_variables and self._variable != '':
            raise VariableSelectionError(self._variable)
        if slise is not None:
            assert type(slise.year0) == int, f'Invalid type for initial_year: ' \
                                                    f'{type(slise.year0)}'
            if slise.year0 > self._dataset_final_timestamp.year:
                raise TimeBoundsSelectionError(f"Initial year not valid. Dataset finishes in "
                                               f"{self._dataset_final_timestamp.year}, got {slise.year0} as "
                                               f"initial year")
            if slise.year0 < self._dataset_initial_timestamp.year:
                raise TimeBoundsSelectionError(f"Initial year not valid. Dataset starts in "
                                               f"{self._dataset_initial_timestamp.year}, got {slise.year0}")
            assert type(slise.yearf) == int, f"Invalid type for final_year: {type(slise.yearf)}"
            if slise.yearf > self._dataset_final_timestamp.year:
                raise TimeBoundsSelectionError(f"Final Year out of bounds. Dataset finishes in "
                                               f"{self._dataset_final_timestamp.year}, got {slise.yearf}")
            if slise.yearf < self._dataset_initial_timestamp.year:
                raise TimeBoundsSelectionError(f"Final year not valid. Dataset starts in "
                                               f"{self._dataset_initial_timestamp.year}, got {slise.year0}")
            assert type(slise.yearf) == int, "Invalid type for final_year: %s" % type(slise.yearf)
            assert type(slise.monthf) == int or type(slise.monthf) == Month, \
                "Invalid type for final_month: %s" % type(slise.monthf)
            if slise.yearf >= self._dataset_final_timestamp.year and \
                    slise.monthf > self._dataset_final_timestamp.month:
                raise TimeBoundsSelectionError(f"Final Month out of bounds. Dataset finishes in "
                                               f"{mon2str(Month(self._dataset_final_timestamp.month))} "
                                               f"{self._dataset_final_timestamp.year}, got "
                                               f"{mon2str(Month(slise.monthf))} {slise.yearf}")
            assert type(slise.yearf) == int, "Invalid type for final_year: %s" % type(slise.yearf)
            assert type(slise.year0) == int, f"Invalid type for initial_year: {type(slise.year0)}"
            if slise.year0 > slise.yearf:
                raise TimeBoundsSelectionError(f"Initial year bigger than final year\n"
                                               f'NOTE: initial_year={slise.year0}, final_year={slise.yearf}')
            assert type(slise.month0) == int or type(slise.month0) == Month, \
                f"Invalid type for initial_month: {type(slise.month0)}"
            if not 1 <= slise.month0 <= 12:
                raise TimeBoundsSelectionError(f'Initial month not valid, must be int from 0 to 11')
            assert type(slise.monthf) == int or type(slise.monthf) == Month, \
                "Invalid type for final_month: %s" % type(slise.monthf)
            if not 1 <= slise.monthf <= 12:
                raise TimeBoundsSelectionError(f'Final month not valid, must be int from 0 to 11')
            if slise.month0 > slise.monthf and slise.year0 - 1 < self._dataset_initial_timestamp.year:
                raise TimeBoundsSelectionError(f'Initial year not valid, remember that when selecting month slice that '
                                               f'combines years, the initial year backtracks one unit\n'
                                               f'NOTE: dataset_initial_timestamp={self._dataset_initial_timestamp}')
            if slise.sy is not None and slise.sy != 0:
                if not slise.year0 <= slise.sy <= slise.yearf:
                    raise SelectedYearError(slise.sy)
        # debugprint(params, self._dataset_initial_timestamp.year, self._dataset_final_timestamp.year)
        # elif (cmap := params.get('cmap')) not in valid_cmaps:
        #     raise CmapSelectionError(cmap)
        return self

    def slice_dataset(self: ReadDataType, slise: Slise, skip: int = 0) -> ReadDataType:
        """Method that slices the dataset accorging to a slise.

        Args
        ----
            slise : Slise
                Slise to use
            skip : int
                Amount of points to skip in the matrix

        Note
        ----
            It first calls `check_variables` method

        Note
        ----
            If the season contains months from different years (NOV-DEC-JAN-FEB for example)
            the initial year is applied to the month which comes at last (FEB). In this example, the
            data that will be used for NOV is on year before the initial year so keep this in mind
            if your dataset doesn't contain that specific year.

        See Also
        --------
            `stypes.Slise`
        """

        self.check_variables(slise)
        # debugprint(f"[INFO] <{self.__class__.__name__}> Slicing dataset: {self._dataset_name} for {self._plot_name}")
        self._slise = slise

        # Time slise
        fro = pd.to_datetime(self.time.values[0])
        to = fro + pd.DateOffset(months=len(self.time))
        time = pd.date_range(start=fro, end=to, freq='M')
        if len(time) == len(self.time) + 1:
            time = time[:-1]

        if slise.month0 <= slise.monthf:
            timemask = (time.month >= slise.month0) & (time.month <= slise.monthf) & \
                       (time.year >= slise.year0) & (time.year <= slise.yearf)
        else:
            timemask = ((time.month >= slise.month0) & (time.year >= (slise.year0 - 1)) &
                        (time.year <= (slise.yearf - 1))) | \
                       ((time.month <= slise.monthf) & (time.year >= slise.year0) &
                        (time.year <= slise.yearf))

        # Space slise
        latmask = (self.lat >= slise.lat0) & (self.lat <= slise.latf)
        lonmask = (self.lon >= slise.lon0) & (self.lon <= slise.lonf)

        self._data = self._data[{
            self._time_key: timemask,
            self._lat_key: latmask,
            self._lon_key: lonmask,
        }]

        latskipmask: npt.NDArray[np.bool_] = np.zeros(len(self.lat)).astype(bool)
        latskipmask[::skip + 1] = True

        lonskipmask: npt.NDArray[np.bool_] = np.zeros(len(self.lon)).astype(bool)
        lonskipmask[::skip + 1] = True

        self._data = self._data[{
            self._lat_key: latskipmask,
            self._lon_key: lonskipmask,
        }]

        self._plot_data = f'{mon2str(Month(slise.month0))} to {mon2str(Month(slise.monthf))} ' \
                         f'{slise.year0}-{slise.yearf}'
        self._plot_bounds = f'({slise2str(slise)})'

        # return var, lon, lat, time
        return self

    # def slice_dataset_old(self: T, slise: Slise, trust_slise: bool = False) -> T:
    #     """If the initial month is bigger than final month, th slise strats from the year before"""
    #     assert 1 <= slise.month0 <= 12 and 1 <= slise.monthf <= 12, \
    #         f'Initial and final month must be from 1 to 12, got {slise.month0}, {slise.monthf}'
    #     # Check if the years and months parameters are valid
    #     if not trust_slise:
    #         self.check_slise(slise)
    #     else:
    #         debugprint('[WARNING] Trusting slise')
    #     # debugprint(f"[INFO] <{self.__class__.__name__}> Slicing dataset: {self._dataset_name} for {self._plot_name}")
    #     self._slise = slise
    #
    #     if slise.lat0 == slise.latf and slise.lon0 == slise.lonf:
    #         lat_mask = (abs(self.lat - slise.lat0) < 0.6)
    #         # long way of saying self.latitude == latitude_min
    #         lon_mask = (abs(self.lon - slise.lat0) < 0.6)
    #     else:
    #         # Make an array of variable values sliced according to the minimum and maximum values set above
    #         lat_mask = (self.lat >= slise.lat0) \
    #                    & (self.lat <= slise.latf)
    #         lon_mask = (self.lon >= slise.lon0) \
    #                    & (self._data[self._lon_key] <= slise.lonf)
    #
    #     if not lat_mask.all() or not lon_mask.all():
    #         self._data = self._data[{
    #             self._lat_key: lat_mask,
    #             self._lon_key: lon_mask
    #         }]
    #     if self._time_key == '':
    #         assert False, "Loading dataset without time not implemented yet"
    #     # debugprint('[WARNING] <mon2str()> Changing month indexes!!!!!!!!!!!!!')
    #     if slise.month0 == Month.JAN and slise.monthf == Month.DEC:
    #         var = self._data
    #     elif slise.month0 <= slise.monthf:
    #         var = self._data.loc[{
    #             'time': (self._data['time.month'] >= slise.month0) &
    #                     (self._data['time.month'] <= slise.monthf)
    #         }]
    #     else:
    #         var_1 = (self._data['time.month'] >= slise.month0) & (
    #                     self._data['time.year'] < slise.yearf)
    #         var_2 = (self._data['time.month'] <= slise.monthf) & (
    #                     self._data['time.year'] > slise.year0 - 1)
    #         var = self._data.loc[{'time': var_1 + var_2}]
    #         slise.year0 -= 1
    #     try:
    #         typ = type(self.time.data[0])
    #     except KeyError:
    #         raise
    #     if typ == np.datetime64:
    #         sl = slice(datetime.datetime(slise.year0, 1, 1).strftime(T_FORMAT),
    #                    datetime.datetime(slise.yearf + 1, 1, 1).strftime(T_FORMAT))
    #     elif typ == np.int64 or typ == int:
    #         sl = slice(slise.year0, slise.yearf + 1)
    #     else:
    #         sl = slice(typ(slise.year0, 1, 1), typ(slise.yearf + 1, 1, 1))
    #     self._data = var.sel({self._time_key: sl})
    #     self._plot_data = f'{mon2str(Month(slise.month0))} to {mon2str(Month(slise.monthf))} ' \
    #                      f'{slise.year0}-{slise.yearf}'
    #     self._plot_bounds = f'({slise2str(slise)})'
    #
    #     return self

    def save_fig_data(self: ReadDataType) -> ReadDataType:
        """Saves the data as a netcdf4 file in the path specified in __init__
        """

        fig_data_path = os.path.join(self._plot_data_dir, self._plot_data_name)
        # debugprint(f"[INFO] <{self.__class__.__name__}> Saving plot data for {self._plot_name} as "
        #            f"{self._plot_data_name} in path: {fig_data_path}")
        # REMOVES NAN VALUES TO PREVENT ERRORS
        if self._lat_key in self._data.dims and self._lon_key in self._data.dims:
            to_save = self._data[{self._lat_key: ~np.isnan(self._data[self._lat_key]),
                                     self._lon_key: ~np.isnan(self._data[self._lon_key])}]
        else:
            to_save = self._data[~np.isnan(self._data)]
        to_save = to_save.where(lambda a: abs(a) < NAN_VAL)
        self._dataset.assign_coords({self._variable: to_save}).to_netcdf(fig_data_path)

        return self

    def get_dataset_info(self) -> Tuple[str, Dict[str, str]]:
        """Returns a tuple where the first element is the dataset name and the second is a dict with keys:

        · **title**: dataset name without the extension

        · **from**: initial date of the dataset loaded

        · **to**: final date of the dataset

        · **variable**: variable used

        Example
        -------
            >>> (
            ...     'HadISST_sst.nc',
            ...     {
            ...         'title': 'HadISST_sst',
            ...         'from': 'Jan 1870',
            ...         'to': 'May 2020',
            ...         'variable': 'sst'
            ...     }
            ... )

        """
        return (
            self._dataset_name, {
                "title": f"{'.'.join(self._dataset_name.split('.')[:-1])}",
                "from": f"{mon2str(Month(self._dataset_initial_timestamp.month))} "
                        f"{self._dataset_initial_timestamp.year}",
                "to": f"{mon2str(Month(self._dataset_final_timestamp.month))} "
                      f"{self._dataset_final_timestamp.year}",
                "variable": f"{self._variable}",
                # "methodologies": f"All",
            }
        )
