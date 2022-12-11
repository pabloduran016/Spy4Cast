import numpy as np

from spy4cast import Slise, Month
from . import BaseTestCase


class SliseTest(BaseTestCase):
    def test_default(self) -> None:
        self.assertEqual(
            Slise.default(),
            Slise(-90, 90, -180, 180, Month.JAN, Month.DEC, 0, 2000, None)
        )

    def test_from_numpy(self) -> None:
        with self.assertRaises(TypeError):
            Slise.from_numpy(np.array([-90, 90, -180, 180, 1, 12, 0, 2000, np.nan, 1]))
            Slise.from_numpy(np.array([-90, 90, -180, 180, 1, 12, np.nan, 1]))

        self.assertEqual(
            Slise(-90, 90, -180, 180, Month.JAN, Month.DEC, 0, 2000, None),
            Slise.from_numpy(np.array([-90, 90, -180, 180, 1, 12, 0, 2000, np.nan]))
        )

    def test_as_numpy(self) -> None:
        numpy_sl = Slise(-90, 90, -180, 180, Month.JAN, Month.DEC, 0, 2000, None).as_numpy()
        self.assertTrue(np.isclose(numpy_sl[:-1], np.array([-90, 90, -180, 180, 1, 12, 0, 2000])).all())
        self.assertTrue(np.isnan(numpy_sl[-1]))
