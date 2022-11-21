import os
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Callable, \
    TypeVar, Any, Tuple, List, Type, cast

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import numpy.typing as npt
from matplotlib.ticker import MaxNLocator

from . import F
from ._functions import time_from_here, time_to_here, _warning, _error, _debuginfo
from .stypes import Color

T = TypeVar('T', bound='_Procedure')


__all__ = [
    '_Procedure',
    '_plot_map',
    '_plot_ts',
    '_apply_flags_to_fig',
    '_get_index_from_sy',
    '_calculate_figsize',
    'MAX_WIDTH',
    'MAX_HEIGHT',
]

MAX_WIDTH = 17
MAX_HEIGHT = 8


class _Procedure(ABC):
    plot: Callable[..., Tuple[plt.Figure, Sequence[plt.Axes]]] = abstractmethod(lambda: (plt.figure(), []))

    @property
    @abstractmethod
    def var_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def save(self, prefix: str, dir: str = '.') -> None:
        clsname = type(self).__name__
        prefixed = os.path.join(dir, prefix)
        _debuginfo(f'Saving {clsname} data in `{prefix}*.npy`')

        variables: List[Tuple[str, npt.NDArray[Any]]] = [
            (name, getattr(self, name))
            for name in self.var_names
        ]

        if not os.path.exists(dir):
            _warning(f'Creating path {dir} that did not exist')
            folders = dir.split('/')
            for i, folder in enumerate(folders):
                if os.path.exists('/'.join(folders[:i + 1])):
                    continue
                os.mkdir('/'.join(folders[:i + 1]))

        for name, arr in variables:
            path = prefixed + name
            if os.path.exists(path):
                _warning(f'Found already existing file with path {path}')
            np.save(path, arr)

    @classmethod
    def load(cls: Type[T], prefix: str, dir: str = '.', **attrs: Any) -> T:

        clsname = cls.__name__
        # print(clsname, cls)
        prefixed = os.path.join(dir, prefix)
        _debuginfo(f'Loading {clsname} data from `{prefixed}*`', end='')
        time_from_here()

        self = cls.__new__(cls)
        for k, v in attrs.items():
            setattr(self, k, v)

        for name in self.var_names:
            path = prefixed + name + '.npy'
            try:
                setattr(self, name, np.load(path))
            except AttributeError:
                _error(f'Could not set variable `{name}`')
                traceback.print_exc()
                exit(1)
            except FileNotFoundError:
                _error(f'Could not find file `{path}` to load {clsname} variable {name}')
                raise

        _debuginfo(f' took {time_to_here():.03f} seconds')
        return self

def _plot_map(
    arr: npt.NDArray[np.float32],
    lat: Union[npt.NDArray[np.float32], xr.DataArray],
    lon: Union[npt.NDArray[np.float32], xr.DataArray],
    fig: plt.Figure,
    ax: plt.Axes,
    title: Optional[str] = None,
    levels: Optional[npt.NDArray[np.float32]] = None,
    xlim: Optional[Sequence[int]] = None,
    ylim: Optional[Sequence[int]] = None,
    cmap: Optional[str] = None,
    ticks: Optional[
        Union[npt.NDArray[np.float32], Sequence[float]]
    ] = None
) -> None:
    if levels is None:
        n = 30
        _std = np.nanstd(arr)
        _m = np.nanmean(arr)

        bound = max(abs(_m - _std),  abs(_m + _std))

        levels = np.sort(np.array([
            round(x, 2)
            for x in np.linspace(-bound, bound, n)
        ]))

    levels = cast(npt.NDArray[np.float_], np.unique(levels))

    if ticks is None:
        nticks = 6
        n0 = round(levels[0] * 10) / 10
        nf = round(levels[-1] * 10) / 10
        if n0 == nf:
            n0 -= 0.05
            nf += 0.05
        step = abs((nf - n0) / nticks)
        assert step != 0
        i = 0
        while step * (10**i) < 1:
            i += 1
        ticks = np.arange(n0, nf + round(step, i) / 2, round(step, i))
        ticks = [t for t in ticks if levels[0] < t < levels[-1]]
        if len(ticks) <= 1:
            _warning('Could not autmatically create ticks. Customize it yourself')
            ticks = None

    cmap = 'bwr' if cmap is None else cmap
    xlim = sorted((lon[0], lon[-1])) if xlim is None else xlim
    ylim = sorted((lat[-1], lat[0])) if ylim is None else ylim

    im = ax.contourf(
        lon, lat, arr, cmap=cmap, levels=levels,
        extend='both', transform=ccrs.PlateCarree()
    )
    cb = fig.colorbar(
        im, ax=ax, orientation='horizontal', pad=0.02,
        ticks=ticks,
    )
    ax.coastlines()
    cb.ax.tick_params(labelsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # # axs.margins(0)
    if title is not None:
        ax.set_title(title)


def _plot_ts(
    time: npt.NDArray[np.int_],
    arr: npt.NDArray[np.float_],
    ax: plt.Axes,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: Union[Color, str, None] = None,
    xtickslabels: Optional[List[Union[str, int]]] = None,
    only_int_xlabels: bool = True,
    label: Optional[str] = None,
) -> None:
    assert len(time.shape) == 1
    assert len(arr.shape) == 1

    ax.plot(time, arr, linewidth=3, color=color, label=label)
    ax.set_xlim(time[0], time[-1])
    if xtickslabels is not None:
        ax.set_xticklabels(xtickslabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if only_int_xlabels:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _apply_flags_to_fig(
    fig: plt.Figure,
    path: str,
    flags: int,
    *,
    block: bool = True  # Only for testing purposes
) -> None:
    if type(flags) == int:
        flags = F(flags)
    assert type(flags) == F, f"{type(flags)=} {flags=}, {F=}, {type(flags) == F = }, {F.__module__=}, {id(F)=}, {type(flags).__module__=}, {id(type(flags))=}"
    if F.SAVE_FIG in flags:
        _debuginfo(f'Saving plot with path {path}')
        for i in range(2):
            try:
                fig.savefig(path)
                break
            except FileNotFoundError:
                os.makedirs(os.path.dirname(path))
    if F.SHOW_PLOT in flags:
        fig.show()
    if F.SHOW_PLOT in flags and F.NOT_HALT not in flags:
        plt.show(block=block)

def _get_index_from_sy(arr: Union[xr.DataArray, npt.NDArray[np.float32]], sy: int) -> int:
    index = 0
    while index < len(arr) and arr[index] != sy:
        index += 1

    if index > len(arr) - 1 or arr[index] > sy:
        raise ValueError(
            f'Selected Year {sy} is not valid\nNOTE: Valid years {arr}'
        )
    return index

def _calculate_figsize(ratio: Optional[float], maxwidth: float, maxheight: float) -> Tuple[float, float]:
    """Caluculate figsize from ratio and maxwidth and minwidth

    Parameters
    ----------
        ratio : float
            height / width
        maxwidth : float
            Maximum value for the width
        maxheight : float
            Maximum value for the height

    Returns
    -------
        tuple[float, float]
             Figsize ready to pass into matplotlib
    """
    if ratio is None or ratio == 0:
        return maxwidth, maxheight
    if maxheight / ratio <= maxwidth:
        w = maxheight / ratio
        h = maxheight
    else:
        h = maxwidth * ratio
        w = maxwidth

    assert abs((h / w) - ratio) < 0.00001, f'{h / w = }, {ratio = }'
    assert h <= maxheight, f"{h = }, {maxheight = }, {ratio = }"
    assert w <= maxwidth, f"{w = }, {maxwidth = }, {ratio = }"

    return w, h
