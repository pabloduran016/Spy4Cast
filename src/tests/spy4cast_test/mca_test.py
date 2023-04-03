import os
import shutil
from typing import Any

import numpy as np
import xarray as xr

from spy4cast.spy4cast.mca import MCA, Preprocess
from spy4cast import Dataset, Month, Slise
from spy4cast.meteo import Clim
from .. import BaseTestCase


DATASETS_DIR = '/Users/Shared/datasets'
DATA_DIR = 'src/tests/data'
HadISST_sst = 'HadISST_sst.nc'
oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'
chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'
SST = 'sst'
CHL = 'CHL'


class MCATest(BaseTestCase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.y_ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(
            Slise(-45, 45, -25, 25, Month.JAN, Month.MAR, 1998, 2002)
        )
        self.z_ds = Dataset(chl_1km_monthly_Sep1997_Dec2020, DATASETS_DIR).open(CHL).slice(
            Slise(36, 37, -5.3, -2, Month.JAN, Month.MAR, 1998, 2002), skip=4
        )
        self.y_preprocessed = Preprocess(self.y_ds)
        self.z_preprocessed = Preprocess(self.z_ds)
        self.mca = MCA(self.y_preprocessed, self.z_preprocessed, 3, .1)

    def test_var_names(self) -> None:
        var_names = (
            'RUY',
            'RUY_sig',
            'SUY',
            'SUY_sig',
            'RUZ',
            'RUZ_sig',
            'SUZ',
            'SUZ_sig',
            'pvalruz',
            'pvalruy',
            'Us',
            'Vs',
            'scf',
            'alpha',
        )
        self.assertEqual(var_names, self.mca.var_names)

    def test___init__(self) -> None:
        _ = MCA(self.y_preprocessed, self.z_preprocessed, 3, .1, sig="test-t")
        _ = MCA(self.y_preprocessed, self.z_preprocessed, 3, .1, sig="monte-carlo", montecarlo_iterations=3)

        y_ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(
            Slise(-45, 45, -25, 25, Month.JAN, Month.MAR, 1999, 2002)
        )
        z_ds = Dataset(chl_1km_monthly_Sep1997_Dec2020, DATASETS_DIR).open(CHL).slice(
            Slise(36, 37, -5.3, -2, Month.JAN, Month.MAR, 1998, 2002), skip=4
        )
        y_preprocessed = Preprocess(y_ds)
        z_preprocessed = Preprocess(z_ds)
        with self.assertRaises(ValueError):
            _ = MCA(y_preprocessed, z_preprocessed, 3, .1)

    def test_from_nparrays(self) -> None:
        _ = MCA.from_land_arrays(self.z_preprocessed.land_data, self.z_preprocessed.land_data, 3, .1)

    def test__mca(self) -> None:
        _ = MCA(self.y_preprocessed, self.z_preprocessed, 3, .1,
                dsy_index_regression=self.y_preprocessed, dsz_index_regression=self.z_preprocessed)

    def test_ydata(self) -> None:
        self.assertTrue(
            np.isclose(self.y_preprocessed.land_data.values, self.mca.ydata).all()
        )

    def test_yvar(self) -> None:
        self.assertEqual(self.y_preprocessed.var, self.mca.yvar)

    def test_yslise(self) -> None:
        self.assertEqual(self.y_preprocessed.slise, self.mca.yslise)

    def test_ytime(self) -> None:
        self.assertTrue(
            np.isclose(self.y_preprocessed.time, self.mca.ytime).all()
        )

    def test_ylat(self) -> None:
        self.assertTrue(
            np.isclose(self.y_preprocessed.lat, self.mca.ylat).all()
        )

    def test_ylon(self) -> None:
        self.assertTrue(
            np.isclose(self.y_preprocessed.lon, self.mca.ylon).all()
        )

    def test_zdata(self) -> None:
        self.assertTrue(
            np.isclose(self.y_preprocessed.land_data.values, self.mca.ydata).all()
        )

    def test_zvar(self) -> None:
        self.assertEqual(self.y_preprocessed.var, self.mca.yvar)

    def test_zslise(self) -> None:
        self.assertEqual(self.y_preprocessed.slise, self.mca.yslise)

    def test_ztime(self) -> None:
        self.assertTrue(
            np.isclose(self.y_preprocessed.time, self.mca.ytime).all()
        )

    def test_zlat(self) -> None:
        self.assertTrue(
            np.isclose(self.y_preprocessed.lat, self.mca.ylat).all()
        )

    def test_zlon(self) -> None:
        self.assertTrue(
            np.isclose(self.y_preprocessed.lon, self.mca.ylon).all()
        )

    def test_plot(self) -> None:
        with self.assertRaises(TypeError):
            self.mca.plot(signs=[True])
        with self.assertRaises(TypeError):
            self.mca.plot(signs=[1, 1, 1])  # type: ignore
        self.mca.plot(signs=[True, False, True])
        self.mca.plot(name='hello')

    def test_load(self) -> None:
        with self.assertRaises(TypeError):
            MCA.load('prefix', 'dir', hello='hello')
        with self.assertRaises(TypeError):
            MCA.load('prefix', 'dir')
        with self.assertRaises(TypeError):
            MCA.load('prefix', 'dir', dsy=self.y_ds, dsz=self.z_ds)  # type: ignore
        self.mca.save('mca_data_', 'mca-data')
        _ = MCA.load('mca_data_', 'mca-data', dsy=self.y_preprocessed, dsz=self.z_preprocessed)
        shutil.rmtree('mca-data')
