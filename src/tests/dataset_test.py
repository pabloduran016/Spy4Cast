import datetime
import os

import numpy as np
import pandas as pd

from spy4cast import Region, Month
from spy4cast.errors import TimeBoundsSelectionError, SelectedYearError, DatasetError, DatasetNotFoundError
from . import BaseTestCase
from spy4cast.dataset import Dataset
import xarray as xr

DATASETS_DIR = '/Users/Shared/datasets'
DATA_DIR = 'src/tests/data'
HadISST_sst = 'HadISST_sst.nc'
oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'
Spain02_v5_0_MM_010reg_aa3d_pr = "Spain02_v5.0_MM_010reg_aa3d_pr.nc"
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
        self.assertTrue(type(spy_ds.region) == Region)

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

    def test_get_region(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR)
        self.assertRaises(ValueError, lambda: ds.region)
        ds.open(SST)
        self.assertTrue(type(ds.region) == Region)

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
        ds.open(SST)  # Behaviour when opening an already-opened Dataset

        _ = Dataset(oisst_v2_mean_monthly, DATASETS_DIR).open()  # Decode times.open(
        with self.assertRaises(DatasetError):
            corrupted_dataset = 'corrupted-dataset.nc'
            with open(corrupted_dataset, 'w') as f:
                f.write('nothing')
            _ = Dataset(corrupted_dataset).open()
            os.remove(corrupted_dataset)
        with self.assertRaises(DatasetNotFoundError):
            _ = Dataset('non_existing_dataset.nc', DATASETS_DIR).open()

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
        for region, year0, yearf, month0, monthf in (
            (Region(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, 1990), 1870, 1990, Month.JAN, Month.MAR),
            (Region(19, 21, 30, 45, Month.NOV, Month.MAR, 1871, 1990), 1870, 1990, Month.NOV, Month.MAR),
        ):
            print(region)
            ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(region)
            self.assertTrue(abs(ds.lat.min().values - 19) <= 0.5)
            self.assertTrue(abs(ds.lat.max().values - 21) <= 0.5)
            self.assertTrue(abs(ds.lon.min().values - 30) <= 0.5)
            self.assertTrue(abs(ds.lon.max().values - 45) <= 0.5)
            self.assertEqual(ds.timestamp0.month, month0)
            self.assertEqual(ds.timestamp0.year, year0)
            self.assertEqual(ds.timestampf.month, monthf)
            self.assertEqual(ds.timestampf.year, yearf)
            self.assertHasAttr(ds, '_region')

    def test__check_region(self) -> None:
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST)
        # type(region.year0) == int
        with self.assertRaises(AssertionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAR, '1870', 1990)  # type: ignore
            )
        # not region.year0 > self.timestampf.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAR, 2030, 2040)
            )
        # not region.year0 < self.timestamp0.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAR, 1000, 2020)
            )
        # type(region.yearf) == int
        with self.assertRaises(AssertionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, '2040')  # type: ignore
            )
        # not region.yearf > self.timestampf.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, 2040)
            )
        # not region.yearf < self.timestamp0.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAR, 1900, 1850)
            )
        # type(region.monthf) == int or type(region.monthf) == Month
        with self.assertRaises(AssertionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, 'Month.MAR', 1870, 2020)  # type: ignore
            )
        # not region.yearf >= self.timestampf.year and \
        #         region.monthf > self.timestampf.month
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.JUN, 1870, 2020)
            )
        # not region.year0 > region.yearf
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAR, 1970, 1920)
            )
        # type(region.month0) == int or type(region.month0) == Month
        with self.assertRaises(AssertionError):
            ds._check_region(
                Region(19, 21, 30, 45, 'Month.JAN', Month.MAR, 1870, 2020)  # type: ignore
            )
        # 1 <= region.month0 <= 12
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, 0, Month.MAR, 1870, 2020)
            )
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, 13, Month.MAR, 1870, 2020)
            )
        # not not 1 <= region.monthf <= 12
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.MAR, 0, 1870, 2020)
            )
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.MAR, 13, 1870, 2020)
            )
        # not region.month0 > region.monthf and \
        #         region.year0 - 1 < self.timestamp0.year
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.DEC, Month.JAN, 1870, 2020)
            )
        # not region.sy is not None and region.sy != 0
        #     region.year0 <= region.sy <= region.yearf
        with self.assertRaises(SelectedYearError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAR, 1870, 2020, 2021)
            )
        ds = ds.slice(Region(-90, 90, -180, 180, Month.MAR, Month.MAY, 1870, 2020))
        with self.assertRaises(TimeBoundsSelectionError):
            ds._check_region(
                Region(19, 21, 30, 45, Month.JAN, Month.MAY, 1870, 2020)
            )

    def test_save_nc(self) -> None:
        name = 'test_save_nc.nc'
        sl = Region(-20, 20, -10, 20, Month.JAN, Month.MAR, 1870, 1990)
        ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(sl)
        ds.save_nc(name)
        ds2 = Dataset(name).open(SST).slice(sl)
        self.assertTrue(
            np.isclose(
                ds.data.values[~np.isnan(ds.data.values)],
                ds2.data.values[~np.isnan(ds2.data.values)]
            ).all()
        )
        os.remove(name)
