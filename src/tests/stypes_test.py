import numpy as np

from spy4cast import Region, Month
from . import BaseTestCase


class RegionTest(BaseTestCase):
    def test_default(self) -> None:
        self.assertEqual(
            Region.default(),
            Region(-90, 90, -180, 180, Month.JAN, Month.DEC, 0, 2000, None)
        )

    def test_from_numpy(self) -> None:
        with self.assertRaises(TypeError):
            Region.from_numpy(np.array([-90, 90, -180, 180, 1, 12, 0, 2000, np.nan, 1]))
            Region.from_numpy(np.array([-90, 90, -180, 180, 1, 12, np.nan, 1]))

        self.assertEqual(
            Region(-90, 90, -180, 180, Month.JAN, Month.DEC, 0, 2000, None),
            Region.from_numpy(np.array([-90, 90, -180, 180, 1, 12, 0, 2000, np.nan]))
        )

    def test_as_numpy(self) -> None:
        numpy_sl = Region(-90, 90, -180, 180, Month.JAN, Month.DEC, 0, 2000, None).as_numpy()
        self.assertTrue(np.isclose(numpy_sl[:-1], np.array([-90, 90, -180, 180, 1, 12, 0, 2000])).all())
        self.assertTrue(np.isnan(numpy_sl[-1]))
