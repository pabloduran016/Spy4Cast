from time import sleep

from spy4cast import Region, Month, set_silence
from spy4cast._functions import time_from_here, time_to_here, region2str, mon2str, str2mon, debugprint, _warning, _error, \
    _debuginfo
from . import BaseTestCase

class FunctionsTest(BaseTestCase):
    def test_time_from_here_and_time_to_here(self) -> None:
        time_from_here()
        sleep(4)
        self.assertTrue(abs(time_to_here() - 4) < 0.01)

        with self.assertRaises(ValueError):
            time_to_here()

    def test_region2str(self) -> None:
        s = Region(-10, 10, -100, 80, Month.JAN, Month.FEB, 1870, 2020)
        self.assertEqual(region2str(s), 'JF (10.00ºS, 10.00ºN - 100.00ºW, 80.00ºE)')

        s2 = Region(-10, 10, -100, 80, Month.MAR, Month.FEB, 1870, 2020)
        self.assertEqual(region2str(s2), 'MAMJJASONDJF (10.00ºS, 10.00ºN - 100.00ºW, 80.00ºE)')

    def test_mon2str(self) -> None:
        with self.assertRaises(ValueError):
            mon2str(0)
            mon2str(13)
        self.assertEqual(mon2str(1), 'Jan')

    def test_str2mon(self) -> None:
        with self.assertRaises(ValueError):
            str2mon('0')
            str2mon('Jhsb')
        self.assertEqual(str2mon('July'), Month.JUL)
        self.assertEqual(str2mon('Janubar'), Month.JAN)
        self.assertEqual(str2mon('mArakjdo'), Month.MAR)

    def test_debugprint(self) -> None:
        set_silence(False)
        debugprint('this is a test. Should be printed')
        set_silence(True)
        debugprint('this is a test. Should not be printed')

    def test__warning(self) -> None:
        _warning('this is a wanrning')

    def test__error(self) -> None:
        _error('this is an error')

    def test__debuginfo(self) -> None:
        _debuginfo('this is an info')
