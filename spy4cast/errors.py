from typing import Any, Optional, Generator

__all__ = [
    'CustomError',
    'PlotCreationError',
    'VariableSelectionError',
    'TimeBoundsSelectionError',
    'PlotSavingError',
    'PlotShowingError',
    'DataSavingError',
    'SelectedYearError',
    'DatasetNotFoundError',
    'DatasetError',
    'PlotDataError',
]


def _new_id_gen() -> Generator[int, None, None]:
    i = 0
    while True:
        i += 1
        yield i


_new_id = _new_id_gen()


class CustomError(Exception):
    """Base Class for all custom error of the project"""
    _id = 0

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Error'
        super().__init__(msg, *args)


class PlotCreationError(CustomError):
    """Error raised when there is an error during plot creation"""
    _id = next(_new_id)


class VariableSelectionError(CustomError, ValueError):
    """Error raised when there is an error when loading the dataset and the variable given is not valid"""
    _id = next(_new_id)

    def __init__(self, variable: str, *args: Any):
        super().__init__(f'Variable selected ({variable}) is not valid', *args)


class TimeBoundsSelectionError(CustomError, ValueError):
    """Error raised when there is an error when loading the dataset and the variable given is not valid"""
    _id = next(_new_id)

    def __init__(self, msg: str, *args: Any):
        super().__init__(msg, *args)


# class MethodologySelectionError(CustomError, ValueError):
#     """Error raised when there is an error when applying the selected methodology"""
#     _id = next(_new_id)
#
#     def __init__(self, msg: str, *args: Any):
#         super().__init__(msg, *args)


# class PlotTypeSelectionError(CustomError, ValueError):
#     """Error raised when there is an error when selecting a plot type that it is not valid"""
#     _id = next(_new_id)
#
#     def __init__(self, plt_type: str, *args: Any):
#         super().__init__(self._flash, *args)


# class CmapSelectionError(CustomError, ValueError):
#     """Error raised when there is an error when selecting a cmap that it is not valid"""
#     _id = next(_new_id)
#
#     def __init__(self, cmap: str, *args: Any):
#         super().__init__(self._flash, *args)


class PlotSavingError(CustomError):
    """Error raised when there is an error when selecting a cmap that it is not valid"""
    _id = next(_new_id)


class PlotShowingError(CustomError):
    """Error raised when there is an error when selecting a cmap that it is not valid"""
    _id = next(_new_id)


class DataSavingError(CustomError):
    """Error raised when there is an error during the saving of the data"""
    _id = next(_new_id)


class SelectedYearError(CustomError, ValueError):
    """Error raised when there is an error during the saving of the data"""
    _id = next(_new_id)

    def __init__(self, s_year: int, *args: Any):
        super().__init__(f'Selected year not valid , got `{s_year}`', *args)


class DatasetNotFoundError(CustomError, ValueError):
    """Error raised when a dataset is not found"""
    _id = next(_new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else "Couldn't find dataset"
        super().__init__(msg, *args)


# class DatasetUnknownError(CustomError, ValueError):
#     """Error raised when there is a dataset is not known"""
#     _id = next(_new_id)
#
#     def __init__(self, e: Optional[str] = None, *args: Any):
#         msg = e if e is not None else 'Unknown dataset'
#         super().__init__(msg, *args)


class DatasetError(CustomError, ValueError):
    """Error raised when there is an error with the dataset which is supposed to be load"""
    _id = next(_new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Dataset name not valid'
        super().__init__(msg, *args)


class PlotDataError(CustomError, ValueError):
    """Error raised when there is an error with the data used to create the plot"""
    _id = next(_new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Plot Data not valid'
        super().__init__(msg, *args)


# class FailedTestError(CustomError):
#     """Error raised when there is an error with tests"""
#     _id = next(_new_id)
