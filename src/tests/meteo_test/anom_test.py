import os
from typing import Any

import numpy as np
import xarray as xr

from spy4cast.meteo.anom import _anom, npanom
from .. import BaseTestCase
from spy4cast import Slise, Dataset, Month
from spy4cast.meteo import Anom, PlotType

DATASETS_DIR = '/Users/Shared/datasets'
DATA_DIR = 'src/tests/data'
HadISST_sst = 'HadISST_sst.nc'
oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'
SST = 'sst'
CHLOS = 'chlos'


class AnomTest(BaseTestCase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ds = Dataset(HadISST_sst, DATASETS_DIR).open(SST).slice(
            Slise(-45, 45, -25, 25, Month.JAN, Month.MAR, 1870, 1990)
        )
        self.ts_anom = Anom(self.ds, 'ts', st=True)
        self.map_anom = Anom(self.ds, 'map', st=True)

    def test___init__(self) -> None:
        with self.assertRaises(TypeError):
            _ = Anom(self.ds, 'idk')  # type: ignore
        self.assertEqual(len(self.map_anom.data.shape), 3)
        self.assertEqual(len(self.ts_anom.data.shape), 1)

    def test_get_type(self) -> None:
        self.assertEqual(self.ts_anom.type, PlotType.TS)
        self.assertEqual(self.map_anom.type, PlotType.MAP)

    def test_get_lat(self) -> None:
        with self.assertRaises(TypeError):
            self.ts_anom = Anom(self.ds, 'ts', st=True)
            _ = self.ts_anom.lat
        _ = self.map_anom.lat

    def test_set_lat(self) -> None:
        with self.assertRaises(TypeError):
            self.ts_anom.lat = self.ds.lat.values
        self.map_anom.lat = self.map_anom.lat.values
        with self.assertRaises(ValueError):
            self.map_anom.lat = self.map_anom.lat.values.tolist()
        with self.assertRaises(ValueError):
            self.map_anom.lat = self.map_anom.lat.values.astype(np.int32)
        with self.assertRaises(ValueError):
            self.map_anom.lat = self.map_anom.lat.values[2:]  # Shape mismatch
        arr = (self.map_anom.lat - 20).values
        self.map_anom.lat = arr
        self.assertTrue((arr == self.map_anom.lat).all())
        self.assertTrue((arr == self.map_anom.data[self.map_anom._lat_key]).all())
        self.map_anom.lat = arr + 20  # Reset the value

    def test_get_lon(self) -> None:
        with self.assertRaises(TypeError):
            self.ts_anom = Anom(self.ds, 'ts', st=True)
            _ = self.ts_anom.lon
        _ = self.map_anom.lon

    def test_set_lon(self) -> None:
        with self.assertRaises(TypeError):
            self.ts_anom.lon = self.ds.lon.values
        self.map_anom.lon = self.map_anom.lon.values
        with self.assertRaises(ValueError):
            self.map_anom.lon = self.map_anom.lon.values.tolist()
        with self.assertRaises(ValueError):
            self.map_anom.lon = self.map_anom.lon.values.astype(np.int32)
        with self.assertRaises(ValueError):
            self.map_anom.lon = self.map_anom.lon.values[2:]  # Shape mismatch
        arr = (self.map_anom.lon - 20).values
        self.map_anom.lon = arr
        self.assertTrue((arr == self.map_anom.lon).all())
        self.assertTrue((arr == self.map_anom.data[self.map_anom._lon_key]).all())
        self.map_anom.lon = arr + 20  # Reset the value

    def test_get_time(self) -> None:
        map_time = self.map_anom.time
        self.assertEqual(type(map_time), xr.DataArray)
        self.assertEqual(map_time.dtype, np.uint)

        ts_time = self.ts_anom.time
        self.assertEqual(type(ts_time), xr.DataArray)
        self.assertEqual(ts_time.dtype, np.uint)

    def test_set_time(self) -> None:
        for anom in (self.map_anom, self.ts_anom):
            anom.time = anom.time.values
            with self.assertRaises(ValueError):
                anom.time = anom.time.values.tolist()
            with self.assertRaises(ValueError):
                anom.time = anom.time.values.astype(np.float32)
            with self.assertRaises(ValueError):
                anom.time = anom.time.values[2:]  # Shape mismatch
            arr = (anom.time + 20).values
            anom.time = arr
            self.assertTrue((arr == anom.time).all())
            self.assertTrue((arr == anom.data[anom._time_key]).all())
            anom.time = arr - 20  # Reset the value

    def test_get_slise(self) -> None:
        xr_sst = xr.open_dataset(
            os.path.join(DATASETS_DIR, HadISST_sst)
        )[SST]
        xr_sst = xr_sst[xr_sst['time.month'] == 1]

        map_anom = Anom.from_xrarray(xr_sst)
        ts_anom = Anom.from_xrarray(xr_sst.mean('latitude').mean('longitude'))

        self.assertFalse(hasattr(map_anom, '_slise'))
        self.assertFalse(hasattr(ts_anom, '_slise'))
        _ = ts_anom.slise
        _ = map_anom.slise
        self.assertTrue(hasattr(map_anom, '_slise'))
        self.assertTrue(hasattr(ts_anom, '_slise'))

    def test_get_var(self) -> None:
        obj = Anom.__new__(Anom)
        self.assertEqual(obj.var, '')
        self.assertEqual(self.ds.var, self.map_anom.var)
        self.assertEqual(self.ds.var, self.ts_anom.var)

    def test_set_data(self) -> None:
        self.map_anom.data = self.map_anom.data.values
        self.ts_anom.data = self.ts_anom.data.values
        with self.assertRaises(ValueError):
            self.ts_anom.data = self.map_anom.data.values.tolist()
        with self.assertRaises(ValueError):
            self.map_anom.data = self.map_anom.data.values.astype(np.int32)

        map_anom = Anom.__new__(Anom)
        self.assertFalse(hasattr(map_anom, '_type'))
        self.assertFalse(hasattr(map_anom, '_data'))
        self.assertFalse(hasattr(map_anom, '_lat'))
        self.assertFalse(hasattr(map_anom, '_lon'))
        map_anom.data = np.empty((10, 20, 30), dtype=np.float32)
        self.assertTrue(hasattr(map_anom, '_data'))
        self.assertEqual(map_anom.data.shape, (10, 20, 30))
        self.assertEqual(map_anom.time.shape, (10,))
        self.assertEqual(map_anom.data['lat'].shape, (20,))
        self.assertEqual(map_anom.data['lon'].shape, (30,))
        self.assertEqual(map_anom.type, PlotType.MAP)

        ts_anom = Anom.__new__(Anom)
        self.assertFalse(hasattr(ts_anom, '_type'))
        self.assertFalse(hasattr(ts_anom, '_data'))
        ts_anom.data = np.empty((10,), dtype=np.float32)
        self.assertTrue(hasattr(ts_anom, '_data'))
        self.assertEqual(ts_anom.data.shape, (10,))
        self.assertEqual(ts_anom.time.shape, (10,))
        self.assertEqual(ts_anom.type, PlotType.TS)

        self.assertEqual(map_anom._time_key, 'year')
        self.assertEqual(ts_anom._time_key, 'year')

        with self.assertRaises(ValueError):
            arr = np.empty((10, 10), dtype=np.float32)
            anom = Anom.__new__(Anom)
            anom.data = arr

    def test_from_xrarray(self) -> None:
        xr_sst = xr.open_dataset(
            os.path.join(DATASETS_DIR, HadISST_sst)
        )[SST]
        xr_sst = xr_sst[xr_sst['time.month'] == 1]

        map_anom = Anom.from_xrarray(xr_sst)
        self.assertEqual(map_anom.type, PlotType.MAP)
        ts_anom = Anom.from_xrarray(xr_sst.mean('latitude').mean('longitude'))
        self.assertEqual(ts_anom.type, PlotType.TS)

        with self.assertRaises(TypeError):
            _ = Anom.from_xrarray(xr_sst.mean('time'))

    def test_plot(self) -> None:
        self.map_anom.plot(year=1980)
        self.ts_anom.plot()

        with self.assertRaises(TypeError):
            self.ts_anom.plot(year=2000)
        with self.assertRaises(TypeError):
            self.ts_anom.plot(cmap='bwr')
        with self.assertRaises(TypeError):
            self.map_anom.plot(color=(1, 1, 2))
        with self.assertRaises(TypeError):
            self.map_anom.plot()

    def test_load(self) -> None:
        dir = 'anom-data'
        self.map_anom.save('anom_map_', dir)
        self.map_anom.save('anom_ts_', dir)
        _ = Anom.load('anom_map_', dir, type='map')
        _ = Anom.load('anom_ts_', dir, type='ts')
        with self.assertRaises(TypeError):
            _ = Anom.load('anom_map_', dir, type='map', hello='hello')
        with self.assertRaises(TypeError):
            _ = Anom.load('anom_map_', dir)
        for x in os.listdir(dir):
            os.remove(os.path.join(dir, x))
        os.removedirs(dir)

    def test__anom(self) -> None:
        with self.assertRaises(TypeError):
            _anom(np.empty((10, 10)))  # type: ignore
        with self.assertRaises(AssertionError):
            _anom(xr.DataArray(np.empty((10, 10, 10)), dims=['year', 'lat', 'lon']))
        with self.assertRaises(KeyError):
            _anom(xr.DataArray(np.empty((10, 10)), dims=['time', 'lat']))
        with self.assertRaises(ValueError):
            _anom(self.ds.data[:, 0])
        with self.assertRaises(ValueError):
            _anom(self.ds.data[1:])

    def test_npanom(self) -> None:
        arr = np.array([[(x * y) / (1 + x + y) for x in range(10)] for y in range(20)])
        res = np.array([
            [0., -0.73546413, -1.34275602, -1.85942509, -2.30737516, -2.70106273 , -3.0507186 , -3.36392109, -3.64646997, -3.90291505],
            [0., -0.4021308 , -0.84275602, -1.25942509, -1.64070849, -1.98677702 , -2.3007186 , -2.58614331, -2.84646997, -3.08473324],
            [0., -0.23546413, -0.54275602, -0.85942509, -1.16451801, -1.45106273 , -1.71738526, -1.96392109, -2.19192452, -2.40291505],
            [0., -0.13546413, -0.34275602, -0.57371081, -0.80737516, -1.03439607 , -1.2507186 , -1.45483018, -1.64646997, -1.82599198],
            [0., -0.06879746, -0.19989888, -0.35942509, -0.52959738, -0.70106273 , -0.86890041, -1.03058776, -1.18493151, -1.33148648],
            [0., -0.02117842, -0.09275602, -0.19275843, -0.30737516, -0.42833546 , -0.5507186 , -0.6716134 , -0.78932712, -0.90291505],
            [0.,  0.01453587, -0.00942269, -0.05942509, -0.12555697, -0.20106273 , -0.28148783, -0.36392109, -0.44646997, -0.52791505],
            [0.,  0.04231365,  0.05724398,  0.04966582,  0.02595818, -0.00875504 , -0.0507186 , -0.09725442, -0.14646997, -0.1970327 ],
            [0.,  0.06453587,  0.11178943,  0.14057491,  0.15416331, 0.15608012  ,  0.1492814 ,  0.13607891,  0.11823591,  0.09708495],
            [0.,  0.08271769,  0.15724398,  0.21749798,  0.26405342, 0.29893727  ,  0.3242814 ,  0.34196126,  0.35353003,  0.36024284],
            [0.,  0.0978692 ,  0.19570551,  0.28343205,  0.35929151, 0.42393727  ,  0.47869317,  0.5249678 ,  0.56405634,  0.59708495],
            [0.,  0.11068972,  0.22867255,  0.34057491,  0.44262484, 0.53423138  ,  0.61594807,  0.68871049,  0.75353003,  0.81137066],
            [0.,  0.12167873,  0.25724398,  0.39057491,  0.51615426, 0.6322706   ,  0.73875509,  0.83607891,  0.9249586 ,  1.00617586],
            [0.,  0.13120254,  0.28224398,  0.43469255,  0.58151373, 0.7199899   ,  0.8492814 ,  0.96941224,  1.08080275,  1.18404147],
            [0.,  0.13953587,  0.3043028 ,  0.47390824,  0.63999327, 0.79893727  ,  0.9492814 ,  1.09062436,  1.22309524,  1.34708495],
            [0.,  0.14688881,  0.32391064,  0.50899596,  0.69262484, 0.87036584  ,  1.04019049,  1.2012963 ,  1.35353003,  1.49708495],
            [0.,  0.15342476,  0.3414545 ,  0.54057491,  0.74024389, 0.9353009   ,  1.12319445,  1.30274558,  1.47353003,  1.63554648],
            [0.,  0.15927271,  0.35724398,  0.56914634,  0.78353394, 0.99458944  ,  1.1992814 ,  1.39607891,  1.58429926,  1.76375161],
            [0.,  0.16453587,  0.37152969,  0.59512036,  0.82305963, 1.04893727  ,  1.2692814 ,  1.48223276,  1.68686336,  1.88279923],
            [0.,  0.16929778,  0.3845167 ,  0.61883578,  0.85929151, 1.09893727  ,  1.33389679,  1.56200484,  1.78210145,  1.99363667]
        ])
        st_res = np.array([
            [np.nan, -3.298777288700889,   -3.0105968581871876,  -2.8151227423678264,  -2.672774200961836,   -2.563803883826095,   -2.4772996474993048,   -2.4067186322611076 , -2.347881725599806,    -2.297983653544742  ],
            [np.nan, -1.803677275762068,   -1.8895455270887729,  -1.9067378596084368,  -1.9005333007341814,  -1.8858157475925585,  -1.8682710933442228,   -1.8502572234000436 , -1.8327793405893686,   -1.816249252992133  ],
            [np.nan, -1.0561272692926575,  -1.216914728429724,   -1.3011479377688433,  -1.3489326577144276,  -1.3773246454174064,  -1.3945822178902703,   -1.4050880963111925 , -1.4113319346717381,   -1.4148039191982922 ],
            [np.nan, -0.6075972654110114,  -0.768494195990358,   -0.8685837078834194,  -0.9352321754496125,  -0.9818315659478436,  -1.0156311175271082,   -1.0408588105112233 , -1.0601257630737126,   -1.075119405988119  ],
            [np.nan, -0.3085772628232474,  -0.44819381567652533, -0.5441605354693517,  -0.6134651336880897,  -0.6654371023721934,  -0.7055802172299758,   -0.7373344056779155 , -0.7629513101830755,   -0.7839612518079707 ],
            [np.nan, -0.09499154668912989, -0.20796853044115057, -0.29183140136952107, -0.35605150027887134, -0.40656890490120673, -0.44720446698236516,  -0.4805060631266552 , -0.5082303505625296,   -0.5316241848518423 ],
            [np.nan, 0.06519774041145798,  -0.02112664192474826, -0.08996809408965672, -0.14544034567132927, -0.1908454070087181,  -0.22857883215746405,  -0.2603674837970037 , -0.2874721855580563,   -0.3108292512652297 ],
            [np.nan, 0.18978940815635975,  0.12834686888837368,  0.07519279368477785,  0.03006894983495628,  -0.0083101395612274,  -0.041185430878977275, -0.06958071504463896, -0.09430879117914236,  -0.1160101922182186 ],
            [np.nan, 0.2894627423522813,   0.2506433777354738,   0.21282686683013985,  0.17857681526335142,  0.14814866110805006,  0.12122218356237803,   0.09735770761368027,  0.07612949797872294,   0.057162304712457784],
            [np.nan, 0.3710136521489443,   0.35255713510805686,  0.329286467183908,    0.30586927134483305,  0.28374628835475724,  0.26332884619856367,   0.24465631584160907,  0.22763019945238105,   0.21210611775569485 ],
            [np.nan, 0.43897274364616334,  0.4387918528848582,   0.4291089817728516,   0.4161893999487834,   0.4023942121956261,   0.38871707793637456,   0.3755884120442122,   0.36318345866565394,   0.35155554949460777 ],
            [np.nan, 0.49647659029765623,  0.5127073252649732,   0.5156218277499367,   0.5127195124772405,   0.5070835567610986,   0.5001732839255398,    0.49273818233075184,  0.4851813919576,       0.47772408297267216 ],
            [np.nan, 0.5457656017132215,   0.5767674013277401,   0.5913205679798856,   0.5978931411788202,   0.6001407519304076,   0.5998972577053192,    0.5981729755886379,   0.5955604744598363,    0.5924227497709124  ],
            [np.nan, 0.5884827449400454,   0.6328199678826606,   0.6581135740651346,   0.6736030333580018,   0.683402452871368,    0.6896488341071209,    0.69356635996482,     0.6959050949164153,    0.6971476194562624  ],
            [np.nan, 0.6258602452635158,   0.6822781148428846,   0.7174851350298012,   0.7413434632025327,   0.7583379837182326,   0.7708526413277985,    0.780287618488622,    0.7875240962028568,    0.7931454166678328  ],
            [np.nan, 0.6588403926077543,   0.726240912140862,    0.770607057998186,    0.802309850062611,    0.8261367973415864,   0.8446742842556871,    0.8594678980103547,   0.8715081807154279,    0.881463390102478   ],
            [np.nan, 0.6881560791359664,   0.7655760465653674,   0.818416788669733,    0.8574699143645862,   0.8877720824537257,   0.9120766538854986,    0.9320498209052764,   0.9487735384669936,    0.9629876732729192  ],
            [np.nan, 0.7143859039243671,   0.8009776675474227,   0.8616732116582752,   0.907615427366382,    0.9440477775561142,   0.9738621593794925,    0.9988251899686037,   1.0200954071607464,    1.038473120652958   ],
            [np.nan, 0.7379927462339274,   0.8330077055788059,   0.9009972325569504,   0.9534004609767175,   0.9956338313999703,   1.030704824433967,     1.060463992180906,    1.0861341744697766,    1.1085667503629935  ],
            [np.nan, 0.7593513178473391,   0.8621259219709727,   0.9369017733774795,   0.9953700751195246,   1.0430930009363177,   1.0831749767919434,    1.1175369571922973,   1.1474558869710196,    1.1738263366447506  ]
        ])
        self.assertTrue(np.isclose(npanom(arr), res).all())
        self.assertTrue(np.isclose(np.nan_to_num(npanom(arr, st=True)), np.nan_to_num(st_res)).all())
