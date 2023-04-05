import os
import shutil
from typing import Any

import numpy as np
import xarray as xr

from spy4cast.spy4cast import Preprocess, MCA
from spy4cast.spy4cast.crossvalidation import Crossvalidation
from spy4cast import Dataset, Month, Region
from .. import BaseTestCase


DATASETS_DIR = '/Users/Shared/datasets'
DATA_DIR = 'src/tests/data'
HadISST_sst = 'HadISST_sst.nc'
oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'
chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'
SST = 'sst'
CHL = 'CHL'


class CrossvalidationTest(BaseTestCase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.y_ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(
            Region(-45, 45, -25, 25, Month.JAN, Month.MAR, 1998, 2002)
        )
        self.z_ds = Dataset(chl_1km_monthly_Sep1997_Dec2020, DATASETS_DIR).open(CHL).slice(
            Region(36, 37, -3, -2, Month.JAN, Month.MAR, 1998, 2002), skip=4
        )
        self.y_preprocessed = Preprocess(self.y_ds)
        self.z_preprocessed = Preprocess(self.z_ds)
        self.cross = Crossvalidation(self.y_preprocessed, self.z_preprocessed, 3, .1, True)

    def test_var_names(self) -> None:
        var_names = (
            'scf',
            'r_uv',
            'r_uv_sig',
            'p_uv',
            'us',
            'vs',

            'zhat_accumulated_modes',
            'zhat_separated_modes',

            'r_z_zhat_t_accumulated_modes',
            'p_z_zhat_t_accumulated_modes',
            'r_z_zhat_t_separated_modes',
            'p_z_zhat_t_separated_modes',

            'r_z_zhat_s_accumulated_modes',
            'p_z_zhat_s_accumulated_modes',
            'r_z_zhat_s_separated_modes',
            'p_z_zhat_s_separated_modes',

            'suy',
            'suz',
            'suy_sig',
            'suz_sig',

            'psi_accumulated_modes',
            'psi_separated_modes',
            'alpha',
        )
        self.assertEqual(var_names, self.cross.var_names)

    def test___init__(self) -> None:
        _ = Crossvalidation(self.y_preprocessed, self.z_preprocessed, 3, .1)

        y_ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(
            Region(-45, 45, -25, 25, Month.JAN, Month.MAR, 1999, 2002)
        )
        z_ds = Dataset(chl_1km_monthly_Sep1997_Dec2020, DATASETS_DIR).open(CHL).slice(
            Region(36, 37, -5.3, -2, Month.JAN, Month.MAR, 1998, 2002), skip=4
        )
        y_preprocessed = Preprocess(y_ds)
        z_preprocessed = Preprocess(z_ds)
        with self.assertRaises(ValueError):
            _ = Crossvalidation(y_preprocessed, z_preprocessed, 3, .1)

    def test_plot(self) -> None:
        mca = MCA(self.y_preprocessed, self.z_preprocessed, 3, .1)
        self.cross.plot(version='default', name='name')
        self.cross.plot(version='elena', mca=mca)
        with self.assertRaises(TypeError):
            self.cross.plot(version='default', mca=mca)
        with self.assertRaises(TypeError):
            self.cross.plot(version='elena')
        with self.assertRaises(TypeError):
            self.cross.plot(version='elena', mca=mca, map_ticks=[1, 2])
        with self.assertRaises(TypeError):
            self.cross.plot(version='elena', mca=mca, cmap='cmap')
        with self.assertRaises(ValueError):
            self.cross.plot(version='nothing')  # type: ignore

    def test_plot_zhat(self) -> None:
        self.cross.plot_zhat(year=1999)
        self.cross.plot_zhat(year=1999, name='name')

    def test_load(self) -> None:
        with self.assertRaises(TypeError):
            Crossvalidation.load('prefix', 'dir', hello='hello')
        with self.assertRaises(TypeError):
            Crossvalidation.load('prefix', 'dir')
        with self.assertRaises(TypeError):
            Crossvalidation.load('prefix', 'dir', dsy=self.y_ds, dsz=self.z_ds)  # type: ignore
        self.cross.save('cross_data_', 'cross-data')
        _ = Crossvalidation.load('cross_data_', 'cross-data', dsy=self.y_preprocessed, dsz=self.z_preprocessed)
        shutil.rmtree('cross-data')
