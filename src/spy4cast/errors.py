"""
Collection of excpetions that are raised from functions in the api
"""

from typing import Any, Optional, Generator, Sequence

__all__ = [
    'Spy4CastError',
    'PlotCreationError',
    'VariableSelectionError',
    'TimeBoundsSelectionError',
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


class Spy4CastError(Exception):
    """Base Class for all custom excpetins of the project"""

    _id = 0

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Error'
        super().__init__(msg, *args)


class PlotCreationError(Spy4CastError):
    """Exception raised when there is an error during plot creation"""

    _id = next(_new_id)


class VariableSelectionError(Spy4CastError, ValueError):
    """Exception raised when there is an error when loading the dataset and
    the variable given is not valid"""

    _id = next(_new_id)

    def __init__(self, variable: str, *args: Any,
                 valid_vars: Optional[Sequence[str]] = None):
        super().__init__(
            f'Variable selected ({variable}) is not valid.'
            f'{(f" Valid: {valid_vars}" if valid_vars is not None else "")}',
            *args)


class TimeBoundsSelectionError(Spy4CastError, ValueError):
    """Exception raised when checking a slise that has non-valid time
    constraints"""
    _id = next(_new_id)

    def __init__(self, msg: str, *args: Any):
        super().__init__(msg, *args)


# class MethodologySelectionError(Spy4CastError, ValueError):
#     """Exception raised when there is an error when applying the selected
#     methodology"""
#     _id = next(_new_id)
#
#     def __init__(self, msg: str, *args: Any):
#         super().__init__(msg, *args)


# class PlotTypeSelectionError(Spy4CastError, ValueError):
#     """Exception raised when there is an error when selecting a plot type
#     that it is not valid"""

#     _id = next(_new_id)
#
#     def __init__(self, plt_type: str, *args: Any):
#         super().__init__(self._flash, *args)


# class CmapSelectionError(Spy4CastError, ValueError):
#     """Exception raised when there is an error when selecting a cmap that it
#     is not valid"""

#     _id = next(_new_id)
#
#     def __init__(self, cmap: str, *args: Any):
#         super().__init__(self._flash, *args)


# class PlotSavingError(Spy4CastError):
#     """Exception raised when there is an error while saving the plot"""
#     _id = next(_new_id)
#

class PlotShowingError(Spy4CastError):
    """Exception raised when there is an error while showing the plot"""
    _id = next(_new_id)


class DataSavingError(Spy4CastError):
    """Exception raised when there is an error while saving the data"""
    _id = next(_new_id)


class SelectedYearError(Spy4CastError, ValueError):
    """Exception raised when the selected year is not valid"""
    _id = next(_new_id)

    def __init__(self, s_year: int, *args: Any):
        super().__init__(f'Selected year not valid , got `{s_year}`', *args)


class DatasetNotFoundError(Spy4CastError, ValueError):
    """Exception raised when a dataset is not found"""
    _id = next(_new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else "Couldn't find dataset"
        super().__init__(msg, *args)


# class DatasetUnknownError(Spy4CastError, ValueError):
#     """Exception raised when there is a dataset is not known"""
#     _id = next(_new_id)
#
#     def __init__(self, e: Optional[str] = None, *args: Any):
#         msg = e if e is not None else 'Unknown dataset'
#         super().__init__(msg, *args)


class DatasetError(Spy4CastError, ValueError):
    """Exception raised when there is an error with the dataset which
    is supposed to be load
    """
    _id = next(_new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else \
            'There was an error while loading the dataset'
        super().__init__(msg, *args)


class PlotDataError(Spy4CastError, ValueError):
    """Exception raised when there is an error with the data used to
    create the plot
    """
    _id = next(_new_id)

    def __init__(self, e: Optional[str] = None, *args: Any):
        msg = e if e is not None else 'Plot Data not valid'
        super().__init__(msg, *args)


# class FailedTestError(Spy4CastError):
#     """Exception raised when there is an error with tests"""
#     _id = next(_new_id)
