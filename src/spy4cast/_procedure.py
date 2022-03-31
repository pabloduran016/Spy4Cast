import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Callable, Dict, TypeVar

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import numpy.typing as npt
from . import F


T = TypeVar('T')


class _Procedure(ABC):
    plot: Callable[..., None] = abstractmethod(lambda: None)

    save: Callable[..., None] = abstractmethod(lambda: None)
    # TODO: Abstract this method out of subclasses

    load: Callable[..., '_Procedure'] = abstractmethod(lambda s: s)
    # TODO: Abstract this method out of subclasses

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
        levels = np.array([
            round(x, 2)
            for x in np.linspace(_m - _std, _m + _std, n)
        ])
    else:
        n = len(levels)

    if ticks is None:
        ticks = np.concatenate(
            (levels[::n // 4], levels[-1:len(levels)])
        )

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

def _apply_flags_to_fig(fig: plt.Figure, path: str,
                        flags: int) -> None:
    if type(flags) == int:
        flags = F(flags)
    assert type(flags) == F
    if F.SAVE_FIG in flags:
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
