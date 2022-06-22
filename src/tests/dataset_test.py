import datetime
import os

import numpy as np
import pandas as pd

from spy4cast import Slise, Month
from spy4cast.errors import TimeBoundsSelectionError, SelectedYearError
from . import BaseTestCase
from spy4cast.dataset import Dataset
import xarray as xr

DATASETS_DIR = '/Users/Shared/datasets'
DATA_DIR = 'src/tests/data'
HadISST_sst = 'HadISST_sst.nc'
chlos_bscr_1958_2016 = 'chlos_bscr_1958_2016.nc'
SST = 'sst'
CHLOS = 'chlos'

class DatasetTest(BaseTestCase):
    def test_from_xrarray(self) -> None:
        """Create a dataset from a xarray"""
        xr_ds = xr.open_dataset(os.path.join(DATASETS_DIR, HadISST_sst))
        spy_ds = Dataset.from_xrarray(xr_ds[SST])
        self.assertHasAttr(spy_ds, '_data')
        self.assertHasAttr(spy_ds, '_lat_key')
        self.assertHasAttr(spy_ds, '_lon_key')
        self.assertHasAttr(spy_ds, '_time_key')

        self.assertTrue(type(spy_ds.data) == xr.DataArray)
        self.assertTrue(type(spy_ds.lat) == xr.DataArray)
        self.assertTrue(type(spy_ds.lon) == xr.DataArray)
        self.assertTrue(type(spy_ds.time) == xr.DataArray)
        self.assertTrue(type(spy_ds.shape) == tuple)
        self.assertTrue(type(spy_ds.timestamp0) == pd.Timestamp or datetime.datetime)
        self.assertTrue(type(spy_ds.timestampf) == pd.Timestamp or datetime.datetime)
        self.assertTrue(type(spy_ds.slise) == Slise)

    def test_get_data(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR)
        self.assertNotHasAttr(ds, '_data')
        with self.assertRaises(ValueError):
            getattr(ds, 'data')
        ds.open(SST)
        self.assertHasAttr(ds, '_data')
        self.assertTrue(type(ds.data) == xr.DataArray)

    def test_set_data(self) -> None:
        ds0 = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        ds = Dataset(HadISST_sst, DATASETS_DIR)
        self.assertNotHasAttr(ds, '_data')
        ds.data = ds0.data
        self.assertHasAttr(ds, '_data')
        self.assertTrue(type(ds._data) == xr.DataArray)

    def test_get_time(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        self.assertTrue(type(ds.time) == xr.DataArray)

    def test_get_lat(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        self.assertTrue(type(ds.lat) == xr.DataArray)

    def test_get_lon(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        self.assertTrue(type(ds.lon) == xr.DataArray)

    def test_get_var(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR)
        self.assertNotHasAttr(ds, '_var')
        self.assertRaises(ValueError, lambda: ds.var)
        ds.open(SST)
        self.assertHasAttr(ds, '_var')
        self.assertTrue(type(ds.var) == str)

    def test_get_shape(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        self.assertTrue(type(ds.shape) == tuple)

    def test_get_timestamp0(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        self.assertTrue(type(ds.timestamp0) == pd.Timestamp)

        ds2 = Dataset(chlos_bscr_1958_2016, DATASETS_DIR).open(CHLOS)
        self.assertTrue(type(ds2.timestamp0) == pd.Timestamp)

    def test_get_timestampf(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        self.assertTrue(type(ds.timestampf) == pd.Timestamp)

        ds2 = Dataset(chlos_bscr_1958_2016, DATASETS_DIR).open(CHLOS)
        self.assertTrue(type(ds2.timestampf) == pd.Timestamp)

    def test_get_slise(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR)
        self.assertRaises(ValueError, lambda: ds.slise)
        ds.open(SST)
        self.assertTrue(type(ds.slise) == Slise)

    def test_open(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR)
        self.assertNotHasAttr(ds, '_ds')
        self.assertNotHasAttr(ds, '_data')
        self.assertNotHasAttr(ds, '_var')
        ds.open(SST)
        self.assertTrue(type(ds._ds) == xr.Dataset)
        self.assertHasAttr(ds, '_ds')
        self.assertHasAttr(ds, '_data')
        self.assertHasAttr(ds, '_var')

    def test__roll_lon(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR)
        ds.open(SST)
        self.assertTrue(ds.lon[0] < ds.lon[-1] and ds.lon.max() <= 180)

    def test__detect_vars(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR)
        self.assertNotHasAttr(ds, '_lat_key')
        self.assertNotHasAttr(ds, '_lon_key')
        self.assertNotHasAttr(ds, '_time_key')
        self.assertNotHasAttr(ds, '_var')
        ds.open(SST)
        self.assertHasAttr(ds, '_lat_key')
        self.assertHasAttr(ds, '_lon_key')
        self.assertHasAttr(ds, '_time_key')
        self.assertHasAttr(ds, '_var')

    def test_slice(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(
            Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, 1990)
        )
        self.assertTrue(abs(ds.lat.min().values - 19) <= 0.5)
        self.assertTrue(abs(ds.lat.max().values - 21) <= 0.5)
        self.assertTrue(abs(ds.lon.min().values - 30) <= 0.5)
        self.assertTrue(abs(ds.lon.max().values - 45) <= 0.5)
        self.assertEqual(ds.timestamp0.month, Month.JAN)
        self.assertEqual(ds.timestamp0.year, 1870)
        self.assertEqual(ds.timestampf.month, Month.MAR)
        self.assertEqual(ds.timestampf.year, 1990)
        self.assertHasAttr(ds, '_slise')

    def test__check_slise(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        # type(slise.year0) == int
        with self.assertRaises(AssertionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, '1870', 1990)  # type: ignore
            )
        # not slise.year0 > self.timestampf.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 2030, 2040)
            )
        # not slise.year0 < self.timestamp0.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 1000, 2020)
            )
        # type(slise.yearf) == int
        with self.assertRaises(AssertionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, '2040')  # type: ignore
            )
        # not slise.yearf > self.timestampf.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, 2040)
            )
        # not slise.yearf < self.timestamp0.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 1810, 1850)
            )
        # type(slise.monthf) == int or type(slise.monthf) == Month
        with self.assertRaises(AssertionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, 'Month.MAR', 1870, 2020)  # type: ignore
            )
        # not slise.yearf >= self.timestampf.year and \
        #         slise.monthf > self.timestampf.month
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, 2040)
            )
        # not slise.year0 > slise.yearf
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 1970, 1920)
            )
        # type(slise.month0) == int or type(slise.month0) == Month
        with self.assertRaises(AssertionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, 'Month.JAN', Month.MAR, 1870, 2020)  # type: ignore
            )
        # 1 <= slise.month0 <= 12
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, 0, Month.MAR, 1870, 2020)
            )
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, 13, Month.MAR, 1870, 2020)
            )
        # not not 1 <= slise.monthf <= 12
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.MAR, 0, 1870, 2020)
            )
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.MAR, 13, 1870, 2020)
            )
        # not slise.month0 > slise.monthf and \
        #         slise.year0 - 1 < self.timestamp0.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.DEC, Month.JAN, 1870, 2020)
            )
        # not slise.sy is not None and slise.sy != 0
        #     slise.year0 <= slise.sy <= slise.yearf
        with self.assertRaises(SelectedYearError):
            ds._check_slise(
                Slise(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, 2020, 2021)
            )

    def test_save_nc(self) -> None:
        name = 'test_save_nc.nc'
        sl = Slise(-20, 20, -10, 20, Month.JAN, Month.MAR, 1870, 1990)
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(sl)
        ds2 = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(sl)
        self.assertTrue(
            np.isclose(
                ds.data.values[~np.isnan(ds.data.values)],
                ds2.data.values[~np.isnan(ds2.data.values)]
            ).all()
        )
