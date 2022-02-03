from typing import Any, Optional, Generator

__all__ = [
    'CustomError',
    'PlotCreationError',
    'VariableSelectionError',
    'TimeBoundsSelectionError',
    'MethodologySelectionError',
    'PlotTypeSelectionError',
    'CmapSelectionError',
    'PlotSavingError',
    'PlotShowingError',
    'DataSavingError',
    'SelectedYearError',
    'DatasetNotFoundError',
    'DatasetUnknownError',
    'DatasetError',
    'PlotDataError',
    'FailedTestError',
]


def new_id_gen() -> Generator[int, None, None]:
    i = 0
    while True:
        i += 1
        yield i


new_id = new_id_gen()


class CustomError(Exception):
    """Base Class for all custom error of the project"""
    _flash = ''
    _category = ''
    _id = 0

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Error'
        super().__init__(msg, *args)

    @property
    def flash(self) -> str:
        """Return the flash message to show on the frontend"""
        return f'{self._flash} ({self._id:03})'
    
    @property
    def category(self) -> str:
        """Return the flash message to show on the frontend"""
        return self._category


class PlotCreationError(CustomError):
    """Error raised when there is an error during plot creation"""
    _flash = 'There was an error while creating the plot'
    _category = 'warning'
    _id = next(new_id)


class VariableSelectionError(CustomError, ValueError):
    """Error raised when there is an error when loading the dataset and the variable given is not valid"""
    _category = 'warning'
    _id = next(new_id)

    def __init__(self, variable: str, *args: Any):
        self._variable = variable
        self._flash = f'Variable selected ({self._variable}) is not valid'
        super().__init__(self._flash, *args)


class TimeBoundsSelectionError(CustomError, ValueError):
    """Error raised when there is an error when loading the dataset and the variable given is not valid"""
    _category = 'warning'
    _id = next(new_id)

    def __init__(self, msg: str, *args: Any):
        super().__init__(msg, *args)
        self._flash = msg


class MethodologySelectionError(CustomError, ValueError):
    """Error raised when there is an error when applying the selected methodology"""
    _category = 'warning'
    _id = next(new_id)

    def __init__(self, msg: str, *args: Any):
        super().__init__(msg, *args)
        self._flash = msg


class PlotTypeSelectionError(CustomError, ValueError):
    """Error raised when there is an error when selecting a plot type that it is not valid"""
    _category = 'warning'
    _flash = 'Unknown selected plot type'
    _id = next(new_id)

    def __init__(self, plt_type: str, *args: Any):
        self._flash += f', got "{plt_type}"'
        super().__init__(self._flash, *args)


class CmapSelectionError(CustomError, ValueError):
    """Error raised when there is an error when selecting a cmap that it is not valid"""
    _category = 'warning'
    _flash = 'Unknown selected cmap'
    _id = next(new_id)

    def __init__(self, cmap: str, *args: Any):
        self._flash += f', got "{cmap}"'
        super().__init__(self._flash, *args)


class PlotSavingError(CustomError):
    """Error raised when there is an error when selecting a cmap that it is not valid"""
    _category = 'warning'
    _flash = 'There was an error while saving the plot'
    _id = next(new_id)


class PlotShowingError(CustomError):
    """Error raised when there is an error when selecting a cmap that it is not valid"""
    _category = 'warning'
    _flash = 'There was an error while attempting to show the plot'
    _id = next(new_id)


class DataSavingError(CustomError):
    """Error raised when there is an error during the saving of the data"""
    _category = 'warning'
    _flash = 'There was an error while saving the data'
    _id = next(new_id)


class SelectedYearError(CustomError, ValueError):
    """Error raised when there is an error during the saving of the data"""
    _category = 'warning'
    _flash = 'Selected year not valid'
    _id = next(new_id)

    def __init__(self, s_year: int, *args: Any):
        self._flash += f', got "{s_year}"'
        super().__init__(self._flash, *args)


class DatasetNotFoundError(CustomError, ValueError):
    """Error raised when a dataset is not found"""
    _category = 'warning'
    _flash = "Error while locating the data, couldn't be found. Try again later"
    _id = next(new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else "Couldn't find dataset"
        super().__init__(msg, *args)


class DatasetUnknownError(CustomError, ValueError):
    """Error raised when there is a dataset is not known"""
    _category = 'warning'
    _flash = "Error while loading the dataset info. Try again later"
    _id = next(new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Unknown dataset'
        super().__init__(msg, *args)


class DatasetError(CustomError, ValueError):
    """Error raised when there is an error with the dataset which is supposed to be load"""
    _category = 'warning'
    _flash = "Error while loading the dataset. Try again later"
    _id = next(new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Dataset name not valid'
        super().__init__(msg, *args)


class PlotDataError(CustomError, ValueError):
    """Error raised when there is an error with the data used to create the plot"""
    _category = 'warning'
    _flash = "Error while creating the plot. Try changing the latitude and longitude params"
    _id = next(new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Plot Data not valid'
        super().__init__(msg, *args)


class FailedTestError(CustomError):
    """Error raised when there is an error with tests"""
    _category = 'none'
    _flash = "Unexpected error"
    _id = next(new_id)
