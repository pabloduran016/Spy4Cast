import datetime
import json
import os
import traceback
from abc import ABCMeta
from typing import ClassVar, Set, Optional, Callable, Type, Any, Tuple, Dict
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

from custom_types import TimeStamp, T_FORMAT, Month, Slise
from errors import *
from functions import mon2str, str2mon, slise2str

__all__ = ['ReadData']

DATASET_INFO_DIR = 'website/static/data/dataset_data.json'
DATASET_DIR = 'website/static/datasets/'
if os.path.exists(DATASET_INFO_DIR):
    with open(DATASET_INFO_DIR, 'r') as f_info:
        DATASET_DATA = json.load(f_info)
else:
    DATASET_DATA = {}


NAN_VAL = 1e3  # Absolute value from where a points is considered NaN


class Concatenable(ABCMeta):
    @staticmethod
    def wrapper(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def rv(self: Any, *args: Any, **kwargs: Any) -> Any:
            f(self, *args, **kwargs)
            return self
        return rv

    def __new__(cls, name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> type:
        for k, v in dct.items():
            if hasattr(v, '__call__') and not k.startswith('_'):
                dct[k] = cls.wrapper(v)
        x: type = super().__new__(cls, name, bases, dct)
        return x


class ReadData(metaclass=Concatenable):
    """Concatenable ReadData object"""
    _silent: ClassVar[bool] = False

    dataset_initial_timestamp: TimeStamp
    dataset_final_timestamp: TimeStamp

    dataset: xr.Dataset
    var_data: xr.DataArray
    slise: Slise
    slice_initial_year: int
    slice_final_year: int
    dataset_min_value: float
    dataset_max_value: float

    def __init__(self, dataset_dir: Optional[str] = None, dataset_name: Optional[str] = None,
                 variable: Optional[str] = None, plot_dir: Optional[str] = None,
                 plot_name: Optional[str] = None, plot_data_dir: Optional[str] = None,
                 force_name: bool = False):
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.dataset_name = dataset_name if dataset_name is not None else 'dataset.nc'
        self.plot_name = plot_name if plot_name is not None else 'plot.png'
        self.plot_data_name = plot_name if plot_name is not None else 'data.nc'
        self.plot_dir = plot_dir if plot_dir is not None else ''
        self.plot_data_dir = plot_data_dir if plot_data_dir is not None else ''
        self.variable = variable if variable is not None else ''
        self.dataset_known = self.dataset_name in DATASET_DATA

        self.loaded_without_times = False

        self.valid_variables: Set[str] = set()
        self.loaded_dataset = False
        self.plot_data = ''
        self.plot_bounds = ''

        # Initialize variables
        self.lon_key: str = 'longitude'
        self.lat_key: str = 'latitude'
        self.time_key: str = 'time'

        if not force_name:
            fig_path = os.path.join(self.plot_dir, self.plot_name)
            self.plot_name, fig_extension = os.path.splitext(self.plot_name)
            i = 1
            while os.path.exists(fig_path):
                fig_path = os.path.join(self.plot_dir, self.plot_name + f'({i})' + fig_extension)
                i += 1
            self.plot_name = os.path.split(fig_path)[-1]

            self.plot_data_name = os.path.splitext(self.plot_name)[0] + '.nc'
            fig_data_path = os.path.join(self.plot_data_dir, self.plot_data_name)
            fig_data_name, fig_data_extension = os.path.splitext(self.plot_data_name)
            i = 1
            while os.path.exists(fig_data_path):
                fig_data_path = os.path.join(self.plot_data_dir, fig_data_name + f'({i})' + fig_data_extension)
                i += 1
            self.plot_data_name = os.path.split(fig_data_path)[-1]
        self.print(f"[INFO] <{self.__class__.__name__}> New Read Data object plot_name: {self.plot_name},"
                   f" plot_data_name: {self.plot_data_name}")

    @classmethod
    def set_silence(cls, val: bool) -> None:
        cls._silent = val

    @classmethod
    def print(cls, *args: Any, **kwargs: Any) -> None:
        if not cls._silent:
            print(*args, **kwargs)

    def load_dataset_info(self) -> None:
        self.print(f"[INFO] <{self.__class__.__name__}> Loading dataset info: {self.dataset_name} for {self.plot_name}")
        if not self.dataset_known:
            raise ValueError('Dataset is not known, add it to the data')
        data = DATASET_DATA[self.dataset_name]

        self.dataset_initial_timestamp = datetime.datetime(int(data['from'].split()[1]), 1, 1)
        self.dataset_final_timestamp = datetime.datetime(int(data['to'].split()[1]),
                                                         (str2mon(data['to'].split()[0])).value, 1)
        self.valid_variables = set(data['variables'].replace(' ', '').split(','))

    def check_variables(self, slise: Optional[Slise] = None) -> None:
        self.print(f"[INFO] <{self.__class__.__name__}> Checking variables for {self.plot_name}")
        if slise is not None:
            self.check_slise(slise)
        if self.dataset_known:
            self.load_dataset_info()
        else:
            if not self.loaded_dataset:
                self.load_dataset()
        # self.print(params, self.dataset_initial_timestamp.year, self.dataset_final_timestamp.year)
        if self.variable not in self.valid_variables and self.variable != '':
            raise VariableSelectionError(self.variable)
        # elif (cmap := params.get('cmap')) not in valid_cmaps:
        #     raise CmapSelectionError(cmap)

    def check_slise(self, slise: Slise) -> None:
        if not self.loaded_dataset:
            if self.dataset_known:
                self.load_dataset_info()
            else:
                self.load_dataset()
        self.print(f"[INFO] <{self.__class__.__name__}> Checking slise for {self.plot_name}")
        assert type(slise.initial_year) == int, f'Invalid type for initial_year: ' \
                                                f'{type(slise.initial_year)}'
        if slise.initial_year > self.dataset_final_timestamp.year:
            raise TimeBoundsSelectionError(f"Initial year not valid. Dataset finishes in "
                                           f"{self.dataset_final_timestamp.year}, got {slise.initial_year} as "
                                           f"initial year")
        if slise.initial_year < self.dataset_initial_timestamp.year:
            raise TimeBoundsSelectionError(f"Initial year not valid. Dataset starts in "
                                           f"{self.dataset_initial_timestamp.year}, got {slise.initial_year}")
        assert type(slise.final_year) == int, f"Invalid type for final_year: {type(slise.final_year)}"
        if slise.final_year > self.dataset_final_timestamp.year:
            raise TimeBoundsSelectionError(f"Final Year out of bounds. Dataset finishes in "
                                           f"{self.dataset_final_timestamp.year}, got {slise.final_year}")
        if slise.final_year < self.dataset_initial_timestamp.year:
            raise TimeBoundsSelectionError(f"Final year not valid. Dataset starts in "
                                           f"{self.dataset_initial_timestamp.year}, got {slise.initial_year}")
        assert type(slise.final_year) == int, "Invalid type for final_year: %s" % type(slise.final_year)
        assert type(slise.final_month) == int or type(slise.final_month) == Month, "Invalid type for final_month: %s" % type(slise.final_month)
        if slise.final_year >= self.dataset_final_timestamp.year and \
                slise.final_month > self.dataset_final_timestamp.month:
            raise TimeBoundsSelectionError(f"Final Month out of bounds. Dataset finishes in "
                                           f"{mon2str(Month(self.dataset_final_timestamp.month))} "
                                           f"{self.dataset_final_timestamp.year}, got "
                                           f"{mon2str(Month(slise.final_month))} {slise.final_year}")
        assert type(slise.final_year) == int, "Invalid type for final_year: %s" % type(slise.final_year)
        assert type(slise.initial_year) == int, f"Invalid type for initial_year: {type(slise.initial_year)}"
        if slise.initial_year > slise.final_year:
            raise TimeBoundsSelectionError(f"Initial year bigger than final year")
        assert type(slise.initial_month) == int or type(slise.initial_month) == Month, \
            f"Invalid type for initial_month: {type(slise.initial_month)}"
        if not 1 <= slise.initial_month <= 12:
            raise TimeBoundsSelectionError(f'Initial month not valid, must be int from 0 to 11')
        assert type(slise.final_month) == int or type(slise.final_month) == Month, "Invalid type for final_month: %s" % type(slise.final_month)
        if not 1 <= slise.final_month <= 12:
            raise TimeBoundsSelectionError(f'Final month not valid, must be int from 0 to 11')
        if slise.initial_month > slise.final_month and slise.initial_year - 1 < self.dataset_initial_timestamp.year:
            raise TimeBoundsSelectionError(f'Initial year not valid, remember that when selecting month slice that '
                                           f'combines years, the initial year backtracks one unit')
        if slise.selected_year is not None and slise.selected_year != 0:
            if not slise.initial_year <= slise.selected_year <= slise.final_year:
                raise SelectedYearError(slise.selected_year)

    def load_dataset(self) -> None:
        if self.loaded_dataset:
            return
        self.print(f"[INFO] <{self.__class__.__name__}> Loading dataset: {self.dataset_name} for {self.plot_name}")
        try:
            self.dataset = xr.load_dataset(os.path.join(self.dataset_dir, self.dataset_name), mask_and_scale=False)
        except ValueError:
            try:
                self.dataset = xr.load_dataset(os.path.join(self.dataset_dir, self.dataset_name), mask_and_scale=False,
                                               decode_times=False)

                initial_timestamp = datetime.datetime.strptime(self.dataset.time.attrs['units'].split()[2],
                                                               '%y-%M-%d')
                final_timestamp = initial_timestamp + relativedelta(months=len(self.dataset.time))
                self.dataset = self.dataset.assign_coords(
                    time=pd.date_range(initial_timestamp, final_timestamp, freq='M'))
            except Exception as e:
                traceback.print_exc()
                self.print(f"[ERROR] <{self.__class__.__name__}> Could not load dataset {self.dataset_name} for {self.plot_name}")
                raise DatasetError() from e
        #     self.dataset = xr.load_dataset(os.path.join(self.dataset_dir, self.dataset_name), decode_times=False)
        #     self.loaded_without_times = True
        except FileNotFoundError as e:
            raise DatasetNotFoundError() from e

        # Keys for the DATASET may vary (example: lon, longitude, ...) depending on the DATASET
        self.lon_key = 'longitude' if 'longitude' in self.dataset.variables \
            else 'lon' if 'lon' in self.dataset.variables else 'nan'
        if self.lon_key == 'nan':
            raise DatasetError(f'Cant recognise dataset longitude variable key: {self.dataset_name}')
        self.lat_key = 'latitude' if 'latitude' in self.dataset.variables \
            else 'lat' if 'lat' in self.dataset.variables else 'nan'
        if self.lat_key == 'nan':
            raise DatasetError(f'Cant recognise dataset latitude variable key: {self.dataset_name}')

        if max(self.dataset[self.lon_key]) > 180:
            self.dataset = self.dataset.assign_coords(
                {self.lon_key: (((self.dataset[self.lon_key] + 180) % 360) - 180)})

        d_keys: Set[str] = set(str(e) for e in self.dataset.variables.keys())
        self.valid_variables.update(
            d_keys - {'lon', 'longitude', 'lon_bnds', 'lat', 'latitude', 'lat_bnds', 'time', 'time_bnds', 'average_DT'}
        )

        if self.variable == '':
            for key in self.valid_variables:
                self.variable = key
                break
        elif self.variable not in self.dataset.variables.keys():
            raise VariableSelectionError(self.variable)

        # Check if values are in Kelvin
        if self.dataset.variables[self.variable].attrs['units'] == 'K':  # values in Kelvin
            self.var_data = getattr(self.dataset, self.variable) - 273.15
        else:
            self.var_data = getattr(self.dataset, self.variable)

        # Establish the initial year of the dataset
        self.dataset_initial_timestamp = self.dataset.indexes['time'][0]
        self.dataset_final_timestamp = self.dataset.indexes['time'][-1]

        self.slice_initial_year = self.dataset_initial_timestamp.year
        self.slice_final_year = self.dataset_final_timestamp.year

        self.plot_data = f'Jan to {mon2str(Month(self.dataset_final_timestamp.month))} ' \
                         f'{self.dataset_initial_timestamp.year}-{self.dataset_final_timestamp.year} '

        self.loaded_dataset = True

    def load_dataset_from_preprocessed(self, slise: Slise) -> None:
        assert 1 <= slise.initial_month <= 12 and 1 <= slise.final_month <= 12, 'Initial and final month must be from 1 to 12'
        # self.print('[WARNING] <mon2str()> Changing month indexes!!!!!!!!!!!!!')
        if self.loaded_dataset:
            return
        self.print(f"[INFO] <{self.__class__.__name__}> Loading dataset from preprocessed: {self.dataset_name} for {self.plot_name}")
        # Load the Dataset
        try:
            self.dataset = xr.load_dataset(os.path.join(self.dataset_dir, self.dataset_name), mask_and_scale=False)
        except FileNotFoundError as e:
            raise DatasetNotFoundError(os.path.join(self.dataset_dir, self.dataset_name)) from e

        # Keys for the DATASET may vary (example: lon, longitude, ...) depending on the DATASET
        self.lon_key = 'longitude' if 'longitude' in self.dataset.variables \
            else 'lon' if 'lon' in self.dataset.variables else ''
        if self.lon_key == '':
            raise DatasetError(f'Cant recognise dataset longitude variable key: {self.dataset_name}')
        self.lat_key = 'latitude' if 'latitude' in self.dataset.variables \
            else 'lat' if 'lat' in self.dataset.variables else ''
        if self.lat_key == '':
            raise DatasetError(f'Cant recognise dataset latitude variable key: {self.dataset_name}')
        # if self.time_key == '':
        #     raise DatasetError(f'Cant recognise dataset time variable key: {self.dataset_name}')

        if max(self.dataset[self.lon_key]) > 180:
            self.dataset = self.dataset.assign_coords(
                {self.lon_key: (((self.dataset[self.lon_key] + 180) % 360) - 180)})

        d_keys: Set[str] = set(str(e) for e in self.dataset.variables.keys())
        self.valid_variables.update(
            d_keys - {'lon', 'longitude', 'lon_bnds', 'lat', 'latitude', 'lat_bnds', 'time', 'year', 'time_bnds'}
        )

        if self.variable == '':
            for key in self.valid_variables:
                self.variable = key
                break
        elif self.variable not in self.dataset.variables.keys():
            raise VariableSelectionError(self.variable)

        self.time_key = 'time' if 'time' in self.dataset[self.variable].dims else \
            'year' if 'year' in self.dataset[self.variable].dims else ''
        # Check if values are in Kelvin
        if 'units' in (var := self.dataset.variables[self.variable].attrs) and var['units'] == 'K':  # values in Kelvin
            self.var_data = getattr(self.dataset, self.variable) - 273.15
        else:
            self.var_data = getattr(self.dataset, self.variable)
        a = ~np.isnan(self.var_data)
        if len(a.shape) == 3:
            a = a[0, :, :]
        lat = self.var_data[self.lat_key].where([x.any() for x in a])
        lon = self.var_data[self.lon_key].where([x.any() for x in a.transpose()])
        self.var_data = self.var_data.assign_coords({self.lat_key: lat, self.lon_key: lon})

        self.dataset_initial_timestamp = datetime.datetime(year=slise.initial_year, month=1, day=1)
        self.dataset_final_timestamp = datetime.datetime(year=slise.final_year, month=1, day=1)
        self.slice_initial_year = slise.initial_year
        self.slice_final_year = slise.final_year

        self.plot_data = f'{mon2str(Month(slise.initial_month))} to {mon2str(Month(slise.final_month))} ' \
                         f'{slise.initial_year}-{slise.final_year} '
        self.plot_bounds = f'({slise2str(slise)})'

        self.loaded_dataset = True

    def slice_dataset(self, slise: Slise, trust_slise: bool = False) -> None:
        """If the initial month is bigger than final month, th slise strats from the year before"""
        assert 1 <= slise.initial_month <= 12 and 1 <= slise.final_month <= 12, \
            f'Initial and final month must be from 1 to 12, got {slise.initial_month}, {slise.final_month}'
        # Check if the years and months parameters are valid
        if not trust_slise:
            self.check_slise(slise)
        else:
            self.print('[WARNING] Trusting slise')
        self.print(f"[INFO] <{self.__class__.__name__}> Slicing dataset: {self.dataset_name} for {self.plot_name}")
        self.slise = slise

        if slise.latitude_min == slise.latitude_max and slise.longitude_min == slise.longitude_max:
            lat_mask = (abs(self.var_data[self.lat_key] - slise.latitude_min) < 0.6)
            # long way of saying self.latitude == latitude_min
            lon_mask = (abs(self.var_data[self.lon_key] - slise.latitude_min) < 0.6)
        else:
            # Make an array of variable values sliced according to the minimum and maximum values set above
            lat_mask = (self.var_data[self.lat_key] >= slise.latitude_min) \
                       & (self.var_data[self.lat_key] <= slise.latitude_max)
            lon_mask = (self.var_data[self.lon_key] >= slise.longitude_min) \
                       & (self.var_data[self.lon_key] <= slise.longitude_max)

        if not lat_mask.all() or not lon_mask.all():
            self.var_data = self.var_data[{
                self.lat_key: lat_mask,
                self.lon_key: lon_mask
            }]
        if self.time_key == '':
            assert False, "Loading dataset without time not implemented yet"
        # self.print('[WARNING] <mon2str()> Changing month indexes!!!!!!!!!!!!!')
        if slise.initial_month == Month.JAN and slise.final_month == Month.DEC:
            var = self.var_data
        elif slise.initial_month <= slise.final_month:
            var = self.var_data.loc[{
                'time': (self.var_data['time.month'] >= slise.initial_month) &
                        (self.var_data['time.month'] <= slise.final_month)
            }]
        else:
            var_1 = (self.var_data['time.month'] >= slise.initial_month) & (
                        self.var_data['time.year'] < slise.final_year)
            var_2 = (self.var_data['time.month'] <= slise.final_month) & (
                        self.var_data['time.year'] > slise.initial_year - 1)
            var = self.var_data.loc[{'time': var_1 + var_2}]
            slise.initial_year -= 1
        typ = type(self.var_data[self.time_key].data[0])
        if typ == np.datetime64:
            sl = slice(datetime.datetime(slise.initial_year, 1, 1).strftime(T_FORMAT),
                       datetime.datetime(slise.final_year + 1, 1, 1).strftime(T_FORMAT))
        elif typ == np.int64 or typ == int:
            sl = slice(slise.initial_year, slise.final_year + 1)
        else:
            sl = slice(typ(slise.initial_year, 1, 1), typ(slise.final_year + 1, 1, 1))
        self.var_data = var.sel(time=sl)
        self.slice_initial_year = slise.initial_year
        self.slice_final_year = slise.final_year
        self.plot_data = f'{mon2str(Month(slise.initial_month))} to {mon2str(Month(slise.final_month))} ' \
                         f'{slise.initial_year}-{slise.final_year}'
        self.plot_bounds = f'({slise2str(slise)})'

    def save_fig_data(self) -> None:
        fig_data_path = os.path.join(self.plot_data_dir, self.plot_data_name)
        self.print(f"[INFO] <{self.__class__.__name__}> Saving plot data for {self.plot_name} as {self.plot_data_name} in path: "
                   f"{fig_data_path}")
        # REMOVES NAN VALUES TO PREVENT ERRORS
        if self.lat_key in self.var_data.dims and self.lon_key in self.var_data.dims:
            to_save = self.var_data[{self.lat_key: ~np.isnan(self.var_data[self.lat_key]),
                                     self.lon_key: ~np.isnan(self.var_data[self.lon_key])}]
        else:
            to_save = self.var_data[~np.isnan(self.var_data)]
        to_save = to_save.where(lambda a: abs(a) < NAN_VAL)
        self.dataset.assign_coords({self.variable: to_save}).to_netcdf(fig_data_path)

    def get_dataset_info(self) -> tuple[str, dict[str, str]]:
        return (
            self.dataset_name, {
                "title": f"{'.'.join(self.dataset_name.split('.')[:-1])}",
                "from": f"Jan {self.dataset_initial_timestamp.year}",
                "to": f"{mon2str(Month(self.dataset_final_timestamp.month))} "
                      f"{self.dataset_final_timestamp.year}",
                "variables": f"{self.variable}",
                "methodologies": f"All",
            }
        )

    @property
    def string_dataset_info(self) -> str:
        return f"""    "{self.dataset_name}": {{
        "title": "{'.'.join(self.dataset_name.split('.')[:-1])}",
        "from": "Jan {self.dataset_initial_timestamp.year}",
        "to": "{mon2str(Month(self.dataset_final_timestamp.month))} {self.dataset_final_timestamp.year}",
        "variables": "{self.variable}",
        "methodologies": "All"
    }},"""
