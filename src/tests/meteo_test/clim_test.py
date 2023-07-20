import os
from typing import Any

import numpy as np
import xarray as xr

from spy4cast.meteo.clim import _clim
from .. import BaseTestCase
from spy4cast import Region, Dataset, Month
from spy4cast.meteo import Clim, PlotType

DATASETS_FOLDER = '/Users/Shared/datasets'
DATA_DIR = 'src/tests/data'
HadISST_sst = 'HadISST_sst.nc'
oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'
SST = 'sst'
CHLOS = 'chlos'


class ClimTest(BaseTestCase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ds = Dataset(HadISST_sst, DATASETS_FOLDER).open(SST).slice(
            Region(-45, 45, -25, 25, Month.JAN, Month.MAR, 1870, 1990)
        )
        self.ts_clim = Clim(self.ds, 'ts')
        self.map_clim = Clim(self.ds, 'map')

    def test___init__(self) -> None:
        with self.assertRaises(TypeError):
            _ = Clim(self.ds, 'idk')  # type: ignore
        self.assertEqual(len(self.map_clim.data.shape), 2)
        self.assertEqual(len(self.ts_clim.data.shape), 1)

    def test_get_type(self) -> None:
        self.assertEqual(self.ts_clim.type, PlotType.TS)
        self.assertEqual(self.map_clim.type, PlotType.MAP)

    def test_get_lat(self) -> None:
        with self.assertRaises(TypeError):
            _ = self.ts_clim.lat
        _ = self.map_clim.lat

    def test_set_lat(self) -> None:
        with self.assertRaises(TypeError):
            self.ts_clim.lat = self.ds.lat.values
        self.map_clim.lat = self.map_clim.lat.values
        with self.assertRaises(ValueError):
            self.map_clim.lat = self.map_clim.lat.values.tolist()
        with self.assertRaises(ValueError):
            self.map_clim.lat = self.map_clim.lat.values.astype(np.int32)
        with self.assertRaises(ValueError):
            self.map_clim.lat = self.map_clim.lat.values[2:]  # Shape mismatch
        arr = (self.map_clim.lat - 20).values
        self.map_clim.lat = arr
        self.assertTrue((arr == self.map_clim.lat).all())
        self.assertTrue((arr == self.map_clim.data[self.map_clim._lat_key]).all())
        self.map_clim.lat = arr + 20  # Reset the value

    def test_get_lon(self) -> None:
        with self.assertRaises(TypeError):
            self.ts_clim = Clim(self.ds, 'ts')
            _ = self.ts_clim.lon
        _ = self.map_clim.lon

    def test_set_lon(self) -> None:
        with self.assertRaises(TypeError):
            self.ts_clim.lon = self.ds.lon.values
        self.map_clim.lon = self.map_clim.lon.values
        with self.assertRaises(ValueError):
            self.map_clim.lon = self.map_clim.lon.values.tolist()
        with self.assertRaises(ValueError):
            self.map_clim.lon = self.map_clim.lon.values.astype(np.int32)
        with self.assertRaises(ValueError):
            self.map_clim.lon = self.map_clim.lon.values[2:]  # Shape mismatch
        arr = (self.map_clim.lon - 20).values
        self.map_clim.lon = arr
        self.assertTrue((arr == self.map_clim.lon).all())
        self.assertTrue((arr == self.map_clim.data[self.map_clim._lon_key]).all())
        self.map_clim.lon = arr + 20  # Reset the value

    def test_get_time(self) -> None:
        with self.assertRaises(TypeError):
            _ = self.map_clim.time

        ts_time = self.ts_clim.time
        self.assertEqual(type(ts_time), xr.DataArray)
        self.assertEqual(ts_time.dtype, np.uint)

    def test_set_time(self) -> None:
        self.ts_clim.time = self.ts_clim.time.values
        with self.assertRaises(ValueError):
            self.ts_clim.time = self.ts_clim.time.values.tolist()
        with self.assertRaises(TypeError):
            self.map_clim.time = ...
        with self.assertRaises(ValueError):
            self.ts_clim.time = self.ts_clim.time.values.astype(np.float32)
        with self.assertRaises(ValueError):
            self.ts_clim.time = self.ts_clim.time.values[2:]  # Shape mismatch
        arr = (self.ts_clim.time + 20).values
        self.ts_clim.time = arr
        self.assertTrue((arr == self.ts_clim.time).all())
        self.assertTrue((arr == self.ts_clim.data[self.ts_clim._time_key]).all())
        self.ts_clim.time = arr - 20  # Reset the value

    def test_get_region(self) -> None:
        _ = self.ts_clim.region

        xr_sst = xr.open_dataset(
            os.path.join(DATASETS_FOLDER, HadISST_sst)
        )[SST]
        xr_sst = xr_sst[xr_sst['time.month'] == 1]

        map_clim = Clim.__new__(Clim)
        map_clim._type = PlotType.MAP
        map_clim.data = xr_sst[0].values
        map_clim.lat = xr_sst['latitude'].values
        map_clim.lon = xr_sst['longitude'].values

        ts_clim = Clim.__new__(Clim)
        ts_clim._type = PlotType.TS
        ts_clim.data = xr_sst[:, 0, 0].values

        self.assertFalse(hasattr(map_clim, '_region'))
        self.assertFalse(hasattr(ts_clim, '_region'))
        _ = ts_clim.region
        _ = map_clim.region
        self.assertTrue(hasattr(map_clim, '_region'))
        self.assertTrue(hasattr(ts_clim, '_region'))

        ts_clim = Clim.__new__(Clim)
        ts_clim._ds = self.ts_clim._ds
        _ = ts_clim.region

    def test_get_var(self) -> None:
        obj = Clim.__new__(Clim)
        self.assertEqual(obj.var, '')
        self.assertEqual(self.ds.var, self.map_clim.var)
        self.assertEqual(self.ds.var, self.ts_clim.var)

    def test_set_data(self) -> None:
        self.map_clim.data = self.map_clim.data.values
        self.ts_clim.data = self.ts_clim.data.values
        with self.assertRaises(ValueError):
            self.ts_clim.data = self.map_clim.data.values.tolist()
        with self.assertRaises(ValueError):
            self.map_clim.data = self.map_clim.data.values.astype(np.int32)

        map_clim = Clim.__new__(Clim)
        self.assertFalse(hasattr(map_clim, '_type'))
        self.assertFalse(hasattr(map_clim, '_data'))
        self.assertFalse(hasattr(map_clim, '_lat'))
        self.assertFalse(hasattr(map_clim, '_lon'))

        map_clim.type = PlotType.MAP
        map_clim.data = np.empty((10, 20), dtype=np.float32)

        self.assertTrue(hasattr(map_clim, '_data'))
        self.assertEqual(map_clim.data.shape, (10, 20))
        self.assertEqual(map_clim._lat_key, 'lat')
        self.assertEqual(map_clim._lon_key, 'lon')
        self.assertEqual(map_clim.data['lat'].shape, (10,))
        self.assertEqual(map_clim.data['lon'].shape, (20,))

        ts_clim = Clim.__new__(Clim)
        self.assertFalse(hasattr(ts_clim, '_type'))
        self.assertFalse(hasattr(ts_clim, '_data'))
        ts_clim.type = PlotType.TS
        ts_clim.data = np.empty((10,), dtype=np.float32)
        self.assertTrue(hasattr(ts_clim, '_data'))
        self.assertEqual(ts_clim.data.shape, (10,))
        self.assertEqual(ts_clim._time_key, 'time')

        with self.assertRaises(ValueError):
            arr = np.empty((10, 10, 10), dtype=np.float32)
            clim = Clim.__new__(Clim)
            clim._type = PlotType.MAP
            clim.data = arr

    def test_plot(self) -> None:
        self.map_clim.plot()
        self.ts_clim.plot()

        with self.assertRaises(TypeError):
            self.ts_clim.plot(cmap='bwr')
        with self.assertRaises(TypeError):
            self.map_clim.plot(color=(1, 1, 2))

    def test_load(self) -> None:
        folder = 'clim-data'
        self.map_clim.save('clim_map_', folder)
        self.ts_clim.save('clim_ts_', folder)
        _ = Clim.load('clim_map_', folder, type='map')
        _ = Clim.load('clim_ts_', folder, type='ts')
        with self.assertRaises(TypeError):
            _ = Clim.load('clim_map_', folder, type='map', hello='hello')
        with self.assertRaises(TypeError):
            _ = Clim.load('clim_map_', folder)
        with self.assertRaises(ValueError):
            _ = Clim.load('clim_map_', folder, type='ts')
        with self.assertRaises(ValueError):
            _ = Clim.load('clim_ts_', folder, type='map')
        for x in os.listdir(folder):
            os.remove(os.path.join(folder, x))
        os.removedirs(folder)

    def test__clim(self) -> None:
        xr_sst = xr.open_dataset(
            os.path.join(DATASETS_FOLDER, HadISST_sst)
        )[SST]

        xr_sst = xr_sst[
            ((xr_sst['time.year'] >= 1910) & (xr_sst['time.year'] <= 2000)) &
            ((xr_sst['time.month'] > 10) | (xr_sst['time.month'] <= 1))
        ]
        xr_sst = xr_sst.assign_coords(year=('time', xr_sst['time.year'].values))

        xr_sst_m = xr_sst.mean('longitude').mean('latitude')
        month_clim0 = _clim(xr_sst_m, 'month').values
        month_clim1 = xr_sst_m.values.reshape((2000 - 1910 + 1, 3)).mean(0)
        self.assertTrue((np.abs(month_clim0 - month_clim1) < 1e-5).all())

        year_clim0 = _clim(xr_sst_m, 'year').values
        year_clim1 = xr_sst_m.values.reshape((2000 - 1910 + 1, 3)).mean(1)
        self.assertTrue((np.abs(year_clim0 - year_clim1) < 1e-5).all())

        time_clim0 = _clim(xr_sst, 'time').values
        time_clim0 = time_clim0[~np.isnan(time_clim0)]
        time_clim1 = xr_sst.values.mean(0)
        time_clim1 = time_clim1[~np.isnan(time_clim1)]
        self.assertTrue((np.abs(time_clim0 - time_clim1) < 1e-5).all())

        with self.assertRaises(TypeError):
            _clim(np.array([1, 2, 3]))
        # with self.assertRaises(ValueError):
        #     _clim(xr_sst_m[1:], 'year')
        with self.assertRaises(ValueError):
            _clim(xr_sst_m, 'idk')

        clim = Clim(Dataset(HadISST_sst, DATASETS_FOLDER).open('sst').slice(
            Region(-45, 45, -25, 25, Month.DEC, Month.FEB, 1871, 1990)
        ), 'map')
        # self.assertTrue(len(clim.time) == 1990 - 1871 + 1)
