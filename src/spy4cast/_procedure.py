import os
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Callable, Dict, TypeVar, Any, Tuple, List, Type

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import numpy.typing as npt
from . import F
from ._functions import debugprint, time_from_here, time_to_here
from .stypes import Color

T = TypeVar('T', bound='_Procedure')


__all__ = [
    '_Procedure',
    '_plot_map',
    '_plot_ts',
    '_apply_flags_to_fig',
    '_get_index_from_sy',
]


class _Procedure(ABC):
    plot: Callable[..., None] = abstractmethod(lambda: None)

    @property
    @abstractmethod
    def var_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def save(self, prefix: str, dir: str = '.') -> None:
        clsname = type(self).__name__
        prefixed = os.path.join(dir, prefix)
        debugprint(f'[INFO] Saving {clsname} data in `{prefix}*.npy`')

        variables: List[Tuple[str, npt.NDArray[Any]]] = [
            (name, getattr(self, name))
            for name in self.var_names
        ]

        if not os.path.exists(dir):
            debugprint(f'[WARNING] Creating path {dir} that did not exist', file=sys.stderr)
            folders = dir.split('/')
            for i, folder in enumerate(folders):
                if os.path.exists('/'.join(folders[:i + 1])):
                    continue
                os.mkdir('/'.join(folders[:i + 1]))

        for name, arr in variables:
            path = prefixed + name
            if os.path.exists(path):
                debugprint(f'[WARNING] Found already existing file with path {path}', file=sys.stderr)
            np.save(path, arr)

    @classmethod
    def load(cls: Type[T], prefix: str, dir: str = '.', **kwargs) -> T:

        clsname = cls.__name__
        # print(clsname, cls)
        prefixed = os.path.join(dir, prefix)
        debugprint(
            f'[INFO] Loading {clsname} data from '
            f'`{prefixed}*`',
            end=''
        )
        time_from_here()

        self = cls.__new__(cls)
        for k, v in kwargs.items():
            setattr(self, k, v)

        for name in self.var_names:
            path = prefixed + name + '.npy'
            try:
                setattr(self, name, np.load(path))
            except AttributeError:
                print(
                    f'[ERROR] Could not set variable `{name}`',
                    file=sys.stderr
                )
                traceback.print_exc()
                exit(1)
            except FileNotFoundError:
                print(
                    f'[ERROR] Could not find file `{path}` to load {clsname} variable {name}',
                    file=sys.stderr
                )
                raise

        debugprint(f' took {time_to_here():.03f} seconds')
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
        levels = np.sort(np.array([
            round(x, 2)
            for x in np.linspace(_m - _std, _m + _std, n)
        ]))
    else:
        n = len(levels)

    levels = np.unique(levels)

    if ticks is None:
        ticks = np.sort(np.concatenate(
            (levels[::n // 4], levels[-1:len(levels)])
        ))

    cmap = 'bwr' if cmap is None else cmap
    xlim = sorted((lon[0], lon[-1])) if xlim is None else xlim
    ylim = sorted((lat[-1], lat[0])) if ylim is None else ylim

    im = ax.contourf(
        lon, lat, arr, cmap=cmap, levels=levels,
        extend='both', transform=ccrs.PlateCarree()
    )
    fig.colorbar(
        im, ax=ax, orientation='horizontal', pad=0.02,
        ticks=ticks
    )
    ax.coastlines()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # # axs.margins(0)
    if title is not None:
        ax.set_title(title)


def _plot_ts(
    time: npt.NDArray[np.float32],
    arr: npt.NDArray[np.float32],
    ax: plt.Axes,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: Optional[Color] = None,
    xtickslabels: Optional[List[Union[str, int]]] = None,
) -> None:
    assert len(time.shape) == 1
    assert len(arr.shape) == 1

    ax.plot(time, arr, linewidth=3, color=color)
    ax.set_xlim(time[0], time[-1])
    if xtickslabels is not None:
        ax.set_xticklabels(xtickslabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


def _apply_flags_to_fig(fig: plt.Figure, path: str,
                        flags: int) -> None:
    if type(flags) == int:
        flags = F(flags)
    assert type(flags) == F, f"{type(flags)=} {flags=}, {F=}, {type(flags) == F = }, {F.__module__=}, {id(F)=}, {type(flags).__module__=}, {id(type(flags))=}"
    if F.SAVE_FIG in flags:
        debugprint(f'[INFO] Saving plot with path {path}')
        for i in range(2):
            try:
                fig.savefig(path)
                break
            except FileNotFoundError:
                os.mkdir("/".join(path.split('/')[:i + 1]))
    if F.SHOW_PLOT in flags:
        fig.show()
    if F.SHOW_PLOT in flags and F.NOT_HALT not in flags:
        plt.show()

def _get_index_from_sy(arr: Union[xr.DataArray, npt.NDArray[np.float32]], sy: int) -> int:
    index = 0
    while index < len(arr) and arr[index] != sy:
        index += 1

    if index > len(arr) - 1 or arr[index] > sy:
        raise ValueError(
            f'Selected Year {sy} is not valid\nNOTE: Valid years {arr}'
        )
    return index
