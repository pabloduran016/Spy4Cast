from . import BaseTestCase
from spy4cast import set_silence, Settings


class TestInit(BaseTestCase):
    def test_set_silence(self) -> None:
        set_silence(True)
        self.assertEqual(Settings.silence, True)
        set_silence(False)
        self.assertEqual(Settings.silence, False)
        with self.assertRaises(TypeError):
            set_silence('False')  # type: ignore
