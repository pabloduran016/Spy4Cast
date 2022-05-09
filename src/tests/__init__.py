from typing import Optional
from unittest import TestCase

from spy4cast import set_silence, Settings

HAS_ATTR_MESSAGE = 'Expected object {obj} to have attribute {attrname}'
NOT_HAS_ATTR_MESSAGE = 'Expected object {obj} to not have attribute {attrname}'

class BaseTestCase(TestCase):
    def assertHasAttr(self,
        obj: object,
        attrname: str,
        message: Optional[str] = None
    ) -> None:
        if not hasattr(obj, attrname):
            if message is not None:
                self.fail(message)
            else:
                self.fail(HAS_ATTR_MESSAGE.format(obj=obj, attrname=attrname))

    def assertNotHasAttr(self,
        obj: object,
        attrname: str,
        message: Optional[str] = None
    ) -> None:
        if hasattr(obj, attrname):
            if message is not None:
                self.fail(message)
            else:
                self.fail(NOT_HAS_ATTR_MESSAGE.format(obj=obj, attrname=attrname))


class TestInit(BaseTestCase):
    def test_set_silence(self) -> None:
        set_silence(True)
        self.assertEqual(Settings.silence, True)
        set_silence(False)
        self.assertEqual(Settings.silence, False)
        with self.assertRaises(ValueError):
            set_silence('False')

from spy4cast import *
from spy4cast.meteo import *
from spy4cast.spy4cast import *
