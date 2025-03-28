import os
from cartopy.util import add_cyclic_point

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from . import BaseTestCase
from spy4cast._procedure import (
    _calculate_figsize,
    _get_index_from_sy,
    plot_map,
    plot_ts,
    _apply_flags_to_fig,
    get_xlim_from_region,
    get_central_longitude_from_region,
)


class ProcedureTest(BaseTestCase):
    def test__plot_map(self) -> None:
        nlat = 20
        nlon = 30
        lat = np.linspace(-20, 20, nlat)
        lon = np.linspace(-25, 25, nlon)
        arr = np.array([[x for x in range(nlon)] for _ in range(nlat)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        plot_map(
            arr=arr,
            lat=lat,
            lon=lon,
            fig=fig,
            ax=ax,
            title='This is a title',
            levels=None,
            xlim=None,
            ylim=None,
            cmap=None,
            ticks=None,
        )
        plt.close(fig)

        nlat = 20
        nlon = 30
        lat = np.linspace(-20, 20, nlat)
        _lon = np.arange(-179.5, 179.5, 10) 
        lon = np.concatenate([_lon[:nlon//2], _lon[-nlon//2:]])
        arr = np.array([[x for x in range(nlon)] for _ in range(nlat)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        plot_map(
            arr=arr,
            lat=lat,
            lon=lon,
            fig=fig,
            ax=ax,
            title='This is a title',
            levels=None,
            xlim=None,
            ylim=None,
            cmap=None,
            ticks=None,
            add_cyclic_point=True,
        )
        plt.close(fig)

        nlat = 20
        nlon = 30
        lat = np.linspace(-20, 20, nlat)
        _lon = np.arange(-179.5, 179.5, 10) 
        lon = np.concatenate([_lon[:nlon//2], _lon[-nlon//2:]])
        arr = np.array([[x for x in range(nlon)] for _ in range(nlat)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        plot_map(
            arr=arr,
            lat=lat,
            lon=lon,
            fig=fig,
            ax=ax,
            title='This is a title',
            levels=None,
            xlim=None,
            ylim=None,
            cmap=None,
            ticks=None,
            add_cyclic_point=False,
            plot_type="pcolor",
        )
        plt.close(fig)

    def test__plot_ts(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        time = np.arange(1990, 2020)
        arr = np.empty(30)
        plot_ts(
            time=time,
            arr=arr,
            ax=ax,
            title='This is a title',
            ylabel='This is a ylabel',
            xlabel='This ia a xlabel',
            color='blue',
            xtickslabels=[1990, 2000, 2010, 2019],
            only_int_xlabels=True,
            label='This is a label',
        )
        plt.close(fig)

    def test__apply_flags_to_fig(self) -> None:
        fig = plt.figure()
        path = "./plot__.png"
        if os.path.exists(path):
            os.remove(path)
        self.assertFalse(os.path.exists(path))
        _apply_flags_to_fig(fig, path, halt_program=False)
        self.assertFalse(os.path.exists(path))
        plt.close(fig)

        fig = plt.figure()
        path = "./plot__.png"
        self.assertFalse(os.path.exists(path))
        _apply_flags_to_fig(fig, path, save_fig=True, show_plot=True, halt_program=False)
        self.assertTrue(os.path.exists(path))
        os.remove(path)
        plt.close(fig)

        fig = plt.figure()
        path = "./plots__/plot__.png"
        self.assertFalse(os.path.exists(path))
        _apply_flags_to_fig(fig, path, save_fig=True, show_plot=True, halt_program=True, _block=False)
        self.assertTrue(os.path.exists(path))
        os.remove(path)
        os.removedirs(os.path.dirname(path))
        plt.close(fig)

        # def _apply_flags_to_fig(fig: plt.Figure, path: str,
        #                         flags: int) -> None:
        #     if type(flags) == int:
        #         flags = F(flags)
        #     assert type(
        #         flags) == F, f"{type(flags)=} {flags=}, {F=}, {type(flags) == F = }, {F.__module__=}, {id(F)=}, {type(flags).__module__=}, {id(type(flags))=}"
        #     if F.SAVE_FIG in flags:
        #         _debuginfo(f'Saving plot with path {path}')
        #         for i in range(2):
        #             try:
        #                 fig.savefig(path)
        #                 break
        #             except FileNotFoundError:
        #                 os.mkdir("/".join(path.split('/')[:i + 1]))
        #     if F.SHOW_PLOT in flags:
        #         fig.show()
        #     if F.SHOW_PLOT in flags and F.NOT_HALT not in flags:
        #         plt.show()

    def test__get_index_from_sy(self) -> None:
        arr = np.arange(1990, 2020)
        i = _get_index_from_sy(arr, 1998)
        self.assertEqual(i, 8)

        with self.assertRaises(ValueError):
            _ = _get_index_from_sy(arr, 1980)

        with self.assertRaises(ValueError):
            _ = _get_index_from_sy(arr, 2030)

    def test__calculate_figsize(self) -> None:
        w, h = _calculate_figsize(16 / 9, 17, 8)
        self.assertEqual(w, 4.5)
        self.assertEqual(h, 8)

        w, h = _calculate_figsize(3 / 10, 17, 8)
        self.assertEqual(w, 17)
        self.assertEqual(h, 5.1)

        w, h = _calculate_figsize(None, 17, 8)
        self.assertEqual(w, 17)
        self.assertEqual(h, 8)

        w, h = _calculate_figsize(0, 17, 8)
        self.assertEqual(w, 17)
        self.assertEqual(h, 8)

    def test__get_central_longitude(self) -> None:
        self.assertEqual(get_central_longitude_from_region(100, 120), 110)
        self.assertEqual(get_central_longitude_from_region(-120, -100), -110)
        self.assertEqual(get_central_longitude_from_region(-100, -120), 70)
        self.assertEqual(get_central_longitude_from_region(100, -120), 170)
        self.assertEqual(get_central_longitude_from_region(100, -10), -135)

    def test__get_xlim_from_region(self) -> None:
        self.assertEqual(get_xlim_from_region(60, -60, 180), (-120 + 180, 120 + 180))
        self.assertEqual(get_xlim_from_region(100, -20, -140), (-120 - 140, 120 - 140))
        self.assertEqual(get_xlim_from_region(-160, 20, -70), (-90 - 70, 90 - 70))

