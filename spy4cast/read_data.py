import datetime
import json
import os
import traceback
from typing import ClassVar, Set, Optional, TypeVar, Tuple, Dict
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

from .stypes import TimeStamp, T_FORMAT, Month, Slise, ChunkType
from .errors import *
from .functions import mon2str, str2mon, slise2str, debugprint

__all__ = ['ReadData', 'NAN_VAL']

DATASET_INFO_DIR = 'website/static/data/dataset_data.json'
DATASET_DIR = 'website/static/datasets/'
if os.path.exists(DATASET_INFO_DIR):
    with open(DATASET_INFO_DIR, 'r') as f_info:
        DATASET_DATA = json.load(f_info)
else:
    DATASET_DATA = {}


NAN_VAL = 1e10  # Absolute value from where a point is considered NaN

T = TypeVar('T', bound='ReadData')


class ReadData:
    """Concatenable ReadData object"""
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
        self._dataset_known = self._dataset_name in DATASET_DATA

        self._loaded_without_times = False

        self._valid_variables: Set[str] = set()
        self._loaded_dataset = False
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
    def time(self) -> xr.DataArray: return self._data[self._time_key]

    @property
    def lat(self) -> xr.DataArray: return self._data[self._lat_key]

    @property
    def lon(self) -> xr.DataArray: return self._data[self._lon_key]

    @property
    def shape(self) -> Tuple[int, ...]: return self._data.shape

    def load_dataset_info(self: T) -> T:
        # debugprint(f"[INFO] <{self.__class__.__name__}> Loading dataset info: {self._dataset_name} "
        #            f"for {self._plot_name}")
        if not self._dataset_known:
            raise ValueError('Dataset is not known, add it to the data')
        data = DATASET_DATA[self._dataset_name]

        self._dataset_initial_timestamp = datetime.datetime(int(data['from'].split()[1]), 1, 1)
        self._dataset_final_timestamp = datetime.datetime(int(data['to'].split()[1]),
                                                          (str2mon(data['to'].split()[0])).value, 1)
        self._valid_variables = set(data['variables'].replace(' ', '').split(','))

        return self

    def check_variables(self: T, slise: Optional[Slise] = None) -> T:
        # debugprint(f"[INFO] <{self.__class__.__name__}> Checking variables for {self._plot_name}")
        if slise is not None:
            self.check_slise(slise)
        if self._dataset_known:
            self.load_dataset_info()
        else:
            if not self._loaded_dataset:
                self.load_dataset()
        # debugprint(params, self._dataset_initial_timestamp.year, self._dataset_final_timestamp.year)
        if self._variable not in self._valid_variables and self._variable != '':
            raise VariableSelectionError(self._variable)
        # elif (cmap := params.get('cmap')) not in valid_cmaps:
        #     raise CmapSelectionError(cmap)

        return self

    def check_slise(self: T, slise: Slise) -> T:
        if not self._loaded_dataset:
            if self._dataset_known:
                self.load_dataset_info()
            else:
                self.load_dataset()
        # debugprint(f"[INFO] <{self.__class__.__name__}> Checking slise for {self._plot_name}")
        assert type(slise.initial_year) == int, f'Invalid type for initial_year: ' \
                                                f'{type(slise.initial_year)}'
        if slise.initial_year > self._dataset_final_timestamp.year:
            raise TimeBoundsSelectionError(f"Initial year not valid. Dataset finishes in "
                                           f"{self._dataset_final_timestamp.year}, got {slise.initial_year} as "
                                           f"initial year")
        if slise.initial_year < self._dataset_initial_timestamp.year:
            raise TimeBoundsSelectionError(f"Initial year not valid. Dataset starts in "
                                           f"{self._dataset_initial_timestamp.year}, got {slise.initial_year}")
        assert type(slise.final_year) == int, f"Invalid type for final_year: {type(slise.final_year)}"
        if slise.final_year > self._dataset_final_timestamp.year:
            raise TimeBoundsSelectionError(f"Final Year out of bounds. Dataset finishes in "
                                           f"{self._dataset_final_timestamp.year}, got {slise.final_year}")
        if slise.final_year < self._dataset_initial_timestamp.year:
            raise TimeBoundsSelectionError(f"Final year not valid. Dataset starts in "
                                           f"{self._dataset_initial_timestamp.year}, got {slise.initial_year}")
        assert type(slise.final_year) == int, "Invalid type for final_year: %s" % type(slise.final_year)
        assert type(slise.final_month) == int or type(slise.final_month) == Month, \
            "Invalid type for final_month: %s" % type(slise.final_month)
        if slise.final_year >= self._dataset_final_timestamp.year and \
                slise.final_month > self._dataset_final_timestamp.month:
            raise TimeBoundsSelectionError(f"Final Month out of bounds. Dataset finishes in "
                                           f"{mon2str(Month(self._dataset_final_timestamp.month))} "
                                           f"{self._dataset_final_timestamp.year}, got "
                                           f"{mon2str(Month(slise.final_month))} {slise.final_year}")
        assert type(slise.final_year) == int, "Invalid type for final_year: %s" % type(slise.final_year)
        assert type(slise.initial_year) == int, f"Invalid type for initial_year: {type(slise.initial_year)}"
        if slise.initial_year > slise.final_year:
            raise TimeBoundsSelectionError(f"Initial year bigger than final year")
        assert type(slise.initial_month) == int or type(slise.initial_month) == Month, \
            f"Invalid type for initial_month: {type(slise.initial_month)}"
        if not 1 <= slise.initial_month <= 12:
            raise TimeBoundsSelectionError(f'Initial month not valid, must be int from 0 to 11')
        assert type(slise.final_month) == int or type(slise.final_month) == Month, \
            "Invalid type for final_month: %s" % type(slise.final_month)
        if not 1 <= slise.final_month <= 12:
            raise TimeBoundsSelectionError(f'Final month not valid, must be int from 0 to 11')
        if slise.initial_month > slise.final_month and slise.initial_year - 1 < self._dataset_initial_timestamp.year:
            raise TimeBoundsSelectionError(f'Initial year not valid, remember that when selecting month slice that '
                                           f'combines years, the initial year backtracks one unit')
        if slise.selected_year is not None and slise.selected_year != 0:
            if not slise.initial_year <= slise.selected_year <= slise.final_year:
                raise SelectedYearError(slise.selected_year)

        return self

    def load_dataset(self: T) -> T:
        if self._loaded_dataset:
            return self
        # debugprint(f"[INFO] <{self.__class__.__name__}> Loading dataset: {self._dataset_name} for {self._plot_name}")
        try:
            self._dataset = xr.load_dataset(os.path.join(self._dataset_dir, self._dataset_name), mask_and_scale=False, chunks=self._chunks)
        except ValueError:
            try:
                self._dataset = xr.load_dataset(os.path.join(self._dataset_dir, self._dataset_name),
                                                mask_and_scale=False, decode_times=False, chunks=self._chunks)

                initial_timestamp = datetime.datetime.strptime(self._dataset.time.attrs['units'].split()[2],
                                                               '%y-%M-%d')
                final_timestamp = initial_timestamp + relativedelta(months=len(self._dataset.time))
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
            raise VariableSelectionError(self._variable)

        # Check if values are in Kelvin
        if self._dataset.variables[self._variable].attrs['units'] == 'K':  # values in Kelvin
            self._data = getattr(self._dataset, self._variable) - 273.15
        else:
            self._data = getattr(self._dataset, self._variable)

        # Establish the initial year of the dataset
        self._dataset_initial_timestamp = self._dataset.indexes['time'][0]
        self._dataset_final_timestamp = self._dataset.indexes['time'][-1]

        self._slise = Slise.default(initial_year=self._dataset_initial_timestamp.year,
                                    final_year=self._dataset_final_timestamp.year)

        self._plot_data = f'Jan to {mon2str(Month(self._dataset_final_timestamp.month))} ' \
                         f'{self._dataset_initial_timestamp.year}-{self._dataset_final_timestamp.year} '

        self._loaded_dataset = True

        return self

    def load_dataset_from_preprocessed(self: T, slise: Slise) -> T:
        assert 1 <= slise.initial_month <= 12 and 1 <= slise.final_month <= 12, \
            'Initial and final month must be from 1 to 12'
        # debugprint('[WARNING] <mon2str()> Changing month indexes!!!!!!!!!!!!!')
        if self._loaded_dataset:
            return self
        # debugprint(f"[INFO] <{self.__class__.__name__}> Loading dataset from preprocessed: {self._dataset_name} "
        #            f"for {self._plot_name}")
        # Load the Dataset
        try:
            self._dataset = xr.load_dataset(os.path.join(self._dataset_dir, self._dataset_name), mask_and_scale=False)
        except FileNotFoundError as e:
            raise DatasetNotFoundError(os.path.join(self._dataset_dir, self._dataset_name)) from e

        # Keys for the DATASET may vary (example: lon, longitude, ...) depending on the DATASET
        self._lon_key = 'longitude' if 'longitude' in self._dataset.variables \
            else 'lon' if 'lon' in self._dataset.variables else ''
        if self._lon_key == '':
            raise DatasetError(f'Cant recognise dataset longitude variable key: {self._dataset_name}')
        self._lat_key = 'latitude' if 'latitude' in self._dataset.variables \
            else 'lat' if 'lat' in self._dataset.variables else ''
        if self._lat_key == '':
            raise DatasetError(f'Cant recognise dataset latitude variable key: {self._dataset_name}')
        # if self._time_key == '':
        #     raise DatasetError(f'Cant recognise dataset time variable key: {self._dataset_name}')

        if max(self._dataset[self._lon_key]) > 180:
            self._dataset = self._dataset.assign_coords(
                {self._lon_key: (((self._dataset[self._lon_key] + 180) % 360) - 180)})

        d_keys: Set[str] = set(str(e) for e in self._dataset.variables.keys())
        self._valid_variables.update(
            d_keys - {'lon', 'longitude', 'lon_bnds', 'lat', 'latitude', 'lat_bnds', 'time', 'year', 'time_bnds'}
        )

        if self._variable == '':
            for key in self._valid_variables:
                self._variable = key
                break
        elif self._variable not in self._dataset.variables.keys():
            raise VariableSelectionError(self._variable)

        self._time_key = 'time' if 'time' in self._dataset[self._variable].dims else \
            'year' if 'year' in self._dataset[self._variable].dims else ''
        # Check if values are in Kelvin
        if 'units' in (var := self._dataset.variables[self._variable].attrs) and \
                var['units'] == 'K':  # values in Kelvin
            self._data = getattr(self._dataset, self._variable) - 273.15
        else:
            self._data = getattr(self._dataset, self._variable)
        a = ~np.isnan(self._data)
        if len(a.shape) == 3:
            a = a[0, :, :]
        lat = self.lat.where([x.any() for x in a])
        lon = self.lon.where([x.any() for x in a.transpose()])
        self._data = self._data.assign_coords({self._lat_key: lat, self._lon_key: lon})

        self._dataset_initial_timestamp = datetime.datetime(year=slise.initial_year, month=1, day=1)
        self._dataset_final_timestamp = datetime.datetime(year=slise.final_year, month=1, day=1)

        self._slise = Slise.default(initial_year=self._dataset_initial_timestamp.year,
                                    final_year=self._dataset_final_timestamp.year)

        self._plot_data = f'{mon2str(Month(slise.initial_month))} to {mon2str(Month(slise.final_month))} ' \
                         f'{slise.initial_year}-{slise.final_year} '
        self._plot_bounds = f'({slise2str(slise)})'

        self._loaded_dataset = True

        return self

    def slice_dataset(self: T, slise: Slise, trust_slise: bool = False) -> T:
        """
        Note: If the season contains months from different years (NOV-DEC-JAN-FEB for example)
        the initial year is applied to the month which comes at last (FEB). In this example, the
        data that will be used for NOV is on year before the initial year so keep this in mind
        if your dataset doesn't contain that specific year.
        """


        if not trust_slise:
            self.check_slise(slise)
        else:
            debugprint('[WARNING] Trusting slise')
        # debugprint(f"[INFO] <{self.__class__.__name__}> Slicing dataset: {self._dataset_name} for {self._plot_name}")
        self._slise = slise

        # Time slise
        fro = pd.to_datetime(self.time.values[0])
        to = fro + relativedelta(months=len(self.time))
        time = pd.date_range(start=fro, end=to, freq='M')

        if slise.initial_month <= slise.final_month:
            timemask = (time.month >= slise.initial_month) & (time.month <= slise.final_month) & \
                       (time.year >= slise.initial_year) & (time.year <= slise.final_year)
        else:
            timemask = ((time.month >= slise.initial_month) & (time.year >= (slise.initial_year - 1)) &
                        (time.year <= (slise.final_year - 1))) | \
                       ((time.month <= slise.final_month) & (time.year >= slise.initial_year) &
                        (time.year <= slise.final_year))

        # Space slise
        latmask = (self.lat >= slise.latitude_min) & (self.lat <= slise.latitude_max)
        lonmask = (self.lon >= slise.longitude_min) & (self.lon <= slise.longitude_max)

        self._data = self._data[{
            self._time_key: timemask,
            self._lat_key: latmask,
            self._lon_key: lonmask,
        }]

        self._plot_data = f'{mon2str(Month(slise.initial_month))} to {mon2str(Month(slise.final_month))} ' \
                         f'{slise.initial_year}-{slise.final_year}'
        self._plot_bounds = f'({slise2str(slise)})'

        # return var, lon, lat, time
        return self

    def slice_dataset_old(self: T, slise: Slise, trust_slise: bool = False) -> T:
        """If the initial month is bigger than final month, th slise strats from the year before"""
        assert 1 <= slise.initial_month <= 12 and 1 <= slise.final_month <= 12, \
            f'Initial and final month must be from 1 to 12, got {slise.initial_month}, {slise.final_month}'
        # Check if the years and months parameters are valid
        if not trust_slise:
            self.check_slise(slise)
        else:
            debugprint('[WARNING] Trusting slise')
        # debugprint(f"[INFO] <{self.__class__.__name__}> Slicing dataset: {self._dataset_name} for {self._plot_name}")
        self._slise = slise

        if slise.latitude_min == slise.latitude_max and slise.longitude_min == slise.longitude_max:
            lat_mask = (abs(self.lat - slise.latitude_min) < 0.6)
            # long way of saying self.latitude == latitude_min
            lon_mask = (abs(self.lon - slise.latitude_min) < 0.6)
        else:
            # Make an array of variable values sliced according to the minimum and maximum values set above
            lat_mask = (self.lat >= slise.latitude_min) \
                       & (self.lat <= slise.latitude_max)
            lon_mask = (self.lon >= slise.longitude_min) \
                       & (self._data[self._lon_key] <= slise.longitude_max)

        if not lat_mask.all() or not lon_mask.all():
            self._data = self._data[{
                self._lat_key: lat_mask,
                self._lon_key: lon_mask
            }]
        if self._time_key == '':
            assert False, "Loading dataset without time not implemented yet"
        # debugprint('[WARNING] <mon2str()> Changing month indexes!!!!!!!!!!!!!')
        if slise.initial_month == Month.JAN and slise.final_month == Month.DEC:
            var = self._data
        elif slise.initial_month <= slise.final_month:
            var = self._data.loc[{
                'time': (self._data['time.month'] >= slise.initial_month) &
                        (self._data['time.month'] <= slise.final_month)
            }]
        else:
            var_1 = (self._data['time.month'] >= slise.initial_month) & (
                        self._data['time.year'] < slise.final_year)
            var_2 = (self._data['time.month'] <= slise.final_month) & (
                        self._data['time.year'] > slise.initial_year - 1)
            var = self._data.loc[{'time': var_1 + var_2}]
            slise.initial_year -= 1
        try:
            typ = type(self.time.data[0])
        except KeyError:
            raise
        if typ == np.datetime64:
            sl = slice(datetime.datetime(slise.initial_year, 1, 1).strftime(T_FORMAT),
                       datetime.datetime(slise.final_year + 1, 1, 1).strftime(T_FORMAT))
        elif typ == np.int64 or typ == int:
            sl = slice(slise.initial_year, slise.final_year + 1)
        else:
            sl = slice(typ(slise.initial_year, 1, 1), typ(slise.final_year + 1, 1, 1))
        self._data = var.sel({self._time_key: sl})
        self._plot_data = f'{mon2str(Month(slise.initial_month))} to {mon2str(Month(slise.final_month))} ' \
                         f'{slise.initial_year}-{slise.final_year}'
        self._plot_bounds = f'({slise2str(slise)})'

        return self

    def save_fig_data(self: T) -> T:
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
        return (
            self._dataset_name, {
                "title": f"{'.'.join(self._dataset_name.split('.')[:-1])}",
                "from": f"Jan {self._dataset_initial_timestamp.year}",
                "to": f"{mon2str(Month(self._dataset_final_timestamp.month))} "
                      f"{self._dataset_final_timestamp.year}",
                "variables": f"{self._variable}",
                "methodologies": f"All",
            }
        )

    @property
    def string_dataset_info(self) -> str:
        return f"""    "{self._dataset_name}": {{
        "title": "{'.'.join(self._dataset_name.split('.')[:-1])}",
        "from": "Jan {self._dataset_initial_timestamp.year}",
        "to": "{mon2str(Month(self._dataset_final_timestamp.month))} {self._dataset_final_timestamp.year}",
        "variables": "{self._variable}",
        "methodologies": "All"
    }},"""
