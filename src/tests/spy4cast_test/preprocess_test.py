import os
from typing import Any

import numpy as np
import xarray as xr

from spy4cast._procedure import _plot_map
from spy4cast.spy4cast.preprocess import Preprocess
from spy4cast import Dataset, Month, Slise
from spy4cast.meteo import Clim
from .. import BaseTestCase


DATASETS_DIR = '/Users/Shared/datasets'
DATA_DIR = 'src/tests/data'
HadISST_sst = 'HadISST_sst.nc'
oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'
SST = 'sst'
CHLOS = 'chlos'


class PreprocessTest(BaseTestCase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(
            Slise(-45, 45, -25, 25, Month.JAN, Month.MAR, 1870, 1990)
        )
        self.preprocessed = Preprocess(self.ds)

    def test___init__(self) -> None:
        _ = Preprocess(self.ds, order=5, period=12)
        ppcessed = Preprocess(self.ds)

        self.assertEqual(len(ppcessed.lat) * len(ppcessed.lon), ppcessed.shape[0])
        self.assertEqual(len(ppcessed.time), ppcessed.shape[1])

        with self.assertRaises(TypeError):
            _ = Preprocess(self.ds, order=5)
        with self.assertRaises(TypeError):
            _ = Preprocess(self.ds, period=13)
        with self.assertRaises(AssertionError):
            _ = Preprocess(Clim(self.ds, 'ts'))  # type: ignore

    def test_var_names(self) -> None:
        var_names = (
            'time', 'lat',  'lon', 'data', 'meta'
        )

        self.assertEqual(var_names, self.preprocessed.var_names)

    def test_get_meta(self) -> None:
        self.assertTrue(
            (np.array([*self.ds.slise.as_numpy(), self.preprocessed.var]) ==
             self.preprocessed.meta).all()
        )

    def test_set_meta(self) -> None:
        ppcessed = Preprocess.__new__(Preprocess)
        ppcessed.meta = self.preprocessed.meta

        self.assertEqual(ppcessed.slise, self.preprocessed.slise)
        self.assertEqual(ppcessed.var, self.preprocessed.var)

    def test_get_time(self) -> None:
        self.assertTrue(
            (self.preprocessed.time == self.preprocessed._time).all())
        self.assertEqual(
            len(self.preprocessed.time),
            self.preprocessed._data.shape[1])

    def test_set_time(self) -> None:
        ppcessed = Preprocess(self.ds)
        time = ppcessed.time
        ppcessed._time = np.zeros(len(time))

        with self.assertRaises(TypeError):
            ppcessed.time = time
        with self.assertRaises(TypeError):
            ppcessed.time = time.values.astype(float)

        ppcessed.time = time.values
        self.assertTrue(
            (ppcessed.time == time).all())

    def test_get_lat(self) -> None:
        self.assertTrue(
            (self.preprocessed.lat == self.preprocessed._lat).all())
        self.assertTrue(
            len(self.preprocessed.lat),
            self.preprocessed._data.shape[0] // len(self.preprocessed._lon))

    def test_set_lat(self) -> None:
        ppcessed = Preprocess(self.ds)
        lat = ppcessed.lat
        ppcessed._lat = np.zeros(len(lat))

        with self.assertRaises(TypeError):
            ppcessed.lat = lat
        with self.assertRaises(TypeError):
            ppcessed.lat = lat.values.astype(str)

        ppcessed.lat = lat.values
        self.assertTrue(
            np.isclose(ppcessed.lat, lat).all()
        )

    def test_get_lon(self) -> None:
        self.assertTrue(
            (self.preprocessed.lon == self.preprocessed._lon).all())
        self.assertEqual(
            len(self.preprocessed.lon),
            self.preprocessed._data.shape[0] // len(self.preprocessed._lat))

    def test_set_lon(self) -> None:
        ppcessed = Preprocess(self.ds)
        lon = ppcessed.lon
        ppcessed._lon = np.zeros(len(lon))

        with self.assertRaises(TypeError):
            ppcessed.lon = lon
        with self.assertRaises(TypeError):
            ppcessed.lon = lon.values.astype(str)

        ppcessed.lon = lon.values
        self.assertTrue(
            np.isclose(ppcessed.lon, lon).all()
        )

    def test_get_shape(self) -> None:
        self.assertEqual(
            self.preprocessed.shape,
            self.preprocessed._data.shape
        )

    def test_get_data(self) -> None:
        self.assertTrue(
            (self.preprocessed.data.values == self.preprocessed._data.values).all())

    def test_set_data(self) -> None:
        ppcessed = Preprocess(self.ds)

        with self.assertRaises(TypeError):
            ppcessed.data.values = xr.DataArray(ppcessed.data.values)
        with self.assertRaises(TypeError):
            ppcessed.data.values = ppcessed.data.values.astype(str)
        with self.assertRaises(TypeError):
            ppcessed.data.values = ppcessed.data.values[np.newaxis, :, :]
        with self.assertRaises(TypeError):
            ppcessed.data.values = ppcessed.data.values[:, 1:]
        with self.assertRaises(TypeError):
            ppcessed.data.values = ppcessed.data.values[1:, :]

        ppcessed.data.values = ppcessed.data.values

    def test_get_var(self) -> None:
        ppcessed = Preprocess.__new__(Preprocess)
        self.assertEqual('', ppcessed.var)

        ppcessed._ds = self.ds
        self.assertEqual(self.ds.var, ppcessed.var)

        var = 'hello'
        ppcessed._var = var
        self.assertEqual(var, ppcessed.var)

    def test_set_var(self) -> None:
        ppcessed = Preprocess(self.ds)

        with self.assertRaises(TypeError):
            ppcessed.var = 'bar'

        del ppcessed._ds
        with self.assertRaises(TypeError):
            ppcessed.var = 1  # type: ignore
        ppcessed.var = 'bar'

    def test_slise(self) -> None:
        ppcessed = Preprocess.__new__(Preprocess)
        ppcessed._lat = self.preprocessed.lat
        ppcessed._lon = self.preprocessed.lon
        ppcessed._time = self.preprocessed.time
        self.assertEqual(ppcessed.slise, Slise(
            lat0=ppcessed.lat.values[0],
            latf=ppcessed.lat.values[-1],
            lon0=ppcessed.lon.values[0],
            lonf=ppcessed.lon.values[-1],
            month0=Month.JAN,
            monthf=Month.DEC,
            year0=ppcessed.time.values[0],
            yearf=ppcessed.time.values[-1],
        ))

        ppcessed = Preprocess.__new__(Preprocess)
        slise0 = Slise(0, 0, 0, 0, Month.JAN, Month.JAN, 0, 0)
        ppcessed._ds = Dataset.__new__(Dataset)
        ppcessed._ds._slise = slise0
        self.assertEqual(ppcessed.slise, slise0)

        ppcessed = Preprocess.__new__(Preprocess)
        slise1 = Slise(1, 1, 1, 1, Month.FEB, Month.FEB, 1, 1)
        ppcessed._slise = slise1
        self.assertEqual(ppcessed.slise, slise1)

    def test_plot(self) -> None:
        self.preprocessed.plot(name='name')
        self.preprocessed.plot(selected_year=self.ds.slise.year0)
