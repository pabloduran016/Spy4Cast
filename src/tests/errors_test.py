from . import BaseTestCase
from spy4cast.errors import *

class TestError(BaseTestCase):
    def test_Spy4CastError(self) -> None:
        Spy4CastError()

    def test_PlotCreationError(self) -> None:
        PlotCreationError()

    def test_VariableSelectionError(self) -> None:
        VariableSelectionError('this is a test for VariableSelectionError')

    def test_TimeBoundsSelectionError(self) -> None:
        TimeBoundsSelectionError('this is a test for TimeBoundsSelectionError')

    def test_PlotShowingError(self) -> None:
        PlotShowingError()

    def test_DataSavingError(self) -> None:
        DataSavingError()

    def test_SelectedYearError(self) -> None:
        SelectedYearError(0000)

    def test_DatasetNotFoundError(self) -> None:
        DatasetNotFoundError()

    def test_DatasetError(self) -> None:
        DatasetError()

    def test_PlotDataError(self) -> None:
        PlotDataError()

