import os
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Callable, \
    TypeVar, Any, Tuple, List, Type, cast, Literal

import matplotlib.contour
from matplotlib import ticker
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import xarray as xr
import cartopy.crs as ccrs
import cartopy.util
import numpy.typing as npt
from matplotlib.ticker import MaxNLocator

from ._functions import time_from_here, time_to_here, _warning, _error, _debuginfo, debugprint
from .stypes import Color, Region

T = TypeVar('T', bound='_Procedure')


__all__ = [
    '_Procedure',
    '_plot_map',
    '_plot_ts',
    '_apply_flags_to_fig',
    '_get_index_from_sy',
    '_calculate_figsize',
    '_add_cyclic_point',
    '_get_central_longitude_from_region',
    '_get_xlim_from_region',
    'MAX_WIDTH',
    'MAX_HEIGHT',
]

MAX_WIDTH = 17
MAX_HEIGHT = 8


class _Procedure(ABC):
    plot: Callable[..., Tuple[Tuple[plt.Figure, ...], Tuple[plt.Axes, ...]]] = abstractmethod(lambda: ((), ()))

    @property
    @abstractmethod
    def var_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def save(self, prefix: str, folder: str = '.') -> None:
        clsname = type(self).__name__
        prefixed = os.path.join(folder, prefix)
        _debuginfo(f'Saving {clsname} data in `{os.path.join(folder, prefix)}*.npy`')

        variables: List[Tuple[str, npt.NDArray[Any]]] = [
            (name, getattr(self, name))
            for name in self.var_names
        ]

        if not os.path.exists(folder):
            _warning(f'Creating path {folder} that did not exist')
            os.makedirs(folder)

        for name, arr in variables:
            path = prefixed + name
            if os.path.exists(path):
                _warning(f'Found already existing file with path {path}')
            np.save(path, arr)

    @classmethod
    def load(cls: Type[T], prefix: str, folder: str = '.', **attrs: Any) -> T:
        clsname = cls.__name__
        # print(clsname, cls)
        prefixed = os.path.join(folder, prefix)
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

        debugprint(f' took {time_to_here():.03f} seconds')
        return self

def _plot_map(
    arr: npt.NDArray[np.float32],
    lat: Union[npt.NDArray[np.float32], xr.DataArray],
    lon: Union[npt.NDArray[np.float32], xr.DataArray],
    fig: plt.Figure,
    ax: plt.Axes,
    title: Optional[str] = None,
    levels: Optional[
        Union[npt.NDArray[np.float32], Sequence[float], bool]
    ] = None,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    cmap: Optional[str] = None,
    ticks: Optional[
        Union[npt.NDArray[np.float32], Sequence[float]]
    ] = None,
    colorbar: bool = True,
    cax: Optional[plt.Axes] = None,
    labels: bool = True,
    add_cyclic_point: bool = False,
    plot_type: Literal["contour", "pcolor"] = "contour",
) -> matplotlib.contour.QuadContourSet:
    if add_cyclic_point:
        arr, lon = _add_cyclic_point(arr, coord=lon)
    cmap = 'bwr' if cmap is None else cmap

    if plot_type == "contour":
        if levels is False or levels is True:
            levels = None
        im = ax.contourf(
            lon, lat, arr, cmap=cmap, levels=levels,
            extend='both', transform=ccrs.PlateCarree(),
        )
        levels = im.levels
    elif plot_type == "pcolor":
        if levels is False:
            norm = None
        else:
            if levels is None or levels is True:
                levels = MaxNLocator(nbins=30).tick_values(np.nanmin(arr), np.nanmax(arr))
            cmap_val = plt.colormaps[cmap]
            norm = BoundaryNorm(levels, ncolors=cmap_val.N, clip=True)
        latstep, lonstep = np.diff(lat[:2])[0], np.diff(lon[:2])[0]
        p_lat = np.append(lat - 0.5 * latstep, lat[-1] + 0.5 * latstep)
        p_lon = np.append(lon - 0.5 * lonstep, lon[-1] + 0.5 * lonstep)
        im = ax.pcolormesh(
            p_lon, p_lat, arr, cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )
    else:
        assert False, f"Unreachable: {plot_type}"
    ax.coastlines()
    xlim = xlim if xlim is not None else (lon.min(), lon.max())
    ylim = ylim if ylim is not None else (lat.min(), lat.max())
    extent = [*xlim, *ylim]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if labels:
        gl = ax.gridlines(alpha=0, draw_labels=True)
        gl.xlocator = ticker.MaxNLocator(6, min_n_ticks=3)
        gl.ylocator = ticker.MaxNLocator(6, min_n_ticks=3)
        gl.top_labels = False
        gl.right_labels = False
    if colorbar:
        cb = fig.colorbar(
            im, ax=ax if cax is None else None, cax=cax, orientation='horizontal', pad=0.02,
            ticks=ticks, 
        )
        if ticks is None:
            tick_locator = ticker.MaxNLocator(nbins=5, prune='both', steps=[2, 5])
            cb.ax.xaxis.set_major_locator(tick_locator)
        cb.ax.tick_params(labelsize=11, labelrotation=0)
    # # axs.margins(0)
    if title is not None:
        ax.set_title(title)
    return im


def _plot_ts(
    time: npt.NDArray[np.int_],
    arr: npt.NDArray[np.float_],
    ax: plt.Axes,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: Union[Color, str, None] = None,
    xtickslabels: Optional[List[Union[str, int]]] = None,
    xticks: Optional[List[Union[str, int, float]]] = None,
    only_int_xlabels: bool = True,
    label: Optional[str] = None,
) -> None:
    assert len(time.shape) == 1
    assert len(arr.shape) == 1

    ax.plot(time, arr, linewidth=3, color=color, label=label)
    ax.set_xlim(time[0], time[-1])
    if xticks is not None:
        ax.set_xticks(xticks)
    if xtickslabels is not None:
        ax.set_xticklabels(xtickslabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if only_int_xlabels and xticks is None and xtickslabels is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _apply_flags_to_fig(
    fig: plt.Figure,
    path: str,
    *,
    save_fig: bool = False,
    show_plot: bool = False,
    halt_program: bool = False,
    _block: bool = True  # Only for testing purposes
) -> None:
    if save_fig:
        _debuginfo(f'Saving plot with path {path}')
        for i in range(2):
            try:
                fig.savefig(path)
                break
            except FileNotFoundError:
                os.makedirs(os.path.dirname(path))
    if show_plot:
        fig.show()
    if show_plot and halt_program:
        plt.show(block=_block)

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


def _add_cyclic_point(
    data: npt.NDArray[np.float_], coord: npt.NDArray[np.float_], axis: int = -1
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    if coord.ndim != 1:
        raise ValueError('The coordinate must be 1-dimensional.')
    if len(coord) != data.shape[axis]:
        raise ValueError(f'The length of the coordinate does not match '
                         f'the size of the corresponding dimension of '
                         f'the data array: len(coord) = {len(coord)}, '
                         f'data.shape[{axis}] = {data.shape[axis]}.')
    delta_coord = np.diff(coord)
    #if not np.allclose(delta_coord, delta_coord[0]):
    #    raise ValueError('The coordinate must be equally spaced.')
    new_coord = np.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = np.concatenate((data, data[tuple(slicer)]), axis=axis)
    return new_data, new_coord


def _get_xlim_from_region(lon0: float, lonf: float, cm: float) -> Tuple[float, float]:
    if cm >= 0:
        if lon0 > lonf:
            a = lon0 - cm
            b = 360 - cm + lonf
        else:
            a = lon0 - cm 
            b = lonf - cm
    else:
        if lon0 > lonf:
            a = lon0 - cm  - 360
            b = lonf - cm
        else:
            a = lon0 - cm
            b = lonf - cm
    xlim = (a + cm, b + cm)
    return xlim

    
def _get_central_longitude_from_region(lon0: float, lonf: float) -> float:
    central_longitude: float
    if lon0 < lonf:
        central_longitude = np.mean([lonf, lon0])
    else:
        central_longitude = np.mean([lonf + 360, lon0])
        central_longitude = central_longitude - 360 if central_longitude > 180 else central_longitude
    
    return central_longitude
    
