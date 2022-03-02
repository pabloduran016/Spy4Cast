"""
Collection of plotters
"""

import os
import traceback
from abc import ABC, abstractmethod
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
# import matplotlib
import numpy as np

from .stypes import Color, Slise, F
from .errors import Spy4CastError, PlotCreationError, DataSavingError, \
    PlotShowingError, PlotDataError, SelectedYearError
from .meteo import Meteo
from .read_data import ReadData, NAN_VAL


__all__ = ['PlotterTS', 'PlotterMap', 'Proker', 'ClimerTS', 'ClimerMap', 'AnomerTS', 'AnomerMap']


class Plotter(ReadData, ABC):
    @abstractmethod
    def create_plot(self, flags: F, **kwargs: Any) -> 'Plotter':
        raise NotImplementedError

    def run(self, flags: F = F(0), **kwargs: Any) -> 'Plotter':
        # Save the data if needed
        if F.SAVE_DATA in flags:
            try:
                self.save_fig_data()
            except Exception as e:
                traceback.print_exc()
                if F.SILENT_ERRORS not in flags:
                    raise DataSavingError(str(e)) from e

        # Create the plot
        if F.SHOW_PLOT in flags or F.SAVE_FIG in flags:
            try:
                self.create_plot(flags & ~F.SHOW_PLOT, **kwargs)
            except Spy4CastError:
                raise
            except Exception as e:
                traceback.print_exc()
                if F.SILENT_ERRORS not in flags:
                    raise PlotCreationError(str(e)) from e

        # Show the plot if needed
        try:
            if F.SHOW_PLOT in flags:
                plt.show()
        except Spy4CastError:
            raise
        except Exception as e:
            traceback.print_exc()
            if F.SILENT_ERRORS not in flags:
                raise PlotShowingError(str(e)) from e

        return self


class PlotterTS(Plotter):
    def create_plot(self, flags: F = F(0), **kws: Any) -> 'PlotterTS':
        fig = plt.figure()
        color: Color = (.43, .92, .20) if 'color' not in kws else kws.pop('color')
        if len(kws) != 0:
            raise TypeError('`create_plot` only accepts one keyword argument: `color`')
        print(f"[INFO] Creating plot for {self._plot_name}")
        # Create figure
        if len(self.shape) == 1:
            raise PlotCreationError('Time series arrays must be unidimensional')
        to_plot = self._data.where(lambda a: abs(a) < NAN_VAL)
        ax = fig.add_subplot()
        ax.plot(to_plot[self._time_key], to_plot, linewidth=3, color=color)
        ax.set_xlim(to_plot[self._time_key][0], to_plot[self._time_key][-1])
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{self._variable} ({self._dataset[self._variable].units})')
        # ax.set_ylim(np.min(self._data), np.max(self._data))
        if F.SAVE_FIG in flags:
            fig.savefig(os.path.join(self._plot_dir, self._plot_name))
        if F.SHOW_PLOT in flags:
            fig.show()

        return self


class PlotterMap(Plotter):
    _n_values = 50

    def create_plot(self, flags: F = F(0), **kws: Any) -> 'PlotterMap':
        fig = plt.figure()
        if 'slise' not in kws:
            raise TypeError("`create_plot` missing 1 required keyword argument: 'slise'")
        slise: Slise = kws.pop('slise')
        cmap: str = kws.pop('cmap') if 'cmap' in kws else 'jet'
        if len(kws) != 0:
            raise TypeError("`create_plot` only accepts two keyword arguments: `cmap` and `slise`")
        # self.print('Preparing dataset for plotting...')
        if len(self.shape) == 3:
            assert slise is not None and slise.sy is not None, 'Expected selected year inside of a slise'
            if not self._slise.year0 <= slise.sy <= self._slise.yearf:
                raise SelectedYearError(slise.sy)
            self._plot_data += f' ({slise.sy})'
            self._time_key = self._time_key if self._time_key in self._data.dims else 'year'
            # .reset_index()?
            to_plot = self._data.sel({self._time_key: slise.sy})
            # to_plot = self._data.sel(year=selected_year)
        else:
            to_plot = self._data

        # Convert to NaN any number above |NAN_VAL|
        to_plot = to_plot.where(lambda a: abs(a) < NAN_VAL).sortby(self._lon_key)

        if to_plot.shape[0] < 2 or to_plot.shape[1] < 2:
            raise PlotDataError(f'Slicing Error, got {to_plot.shape} as slice shapes. Might bee too small')

        # self.print(self._plot_data, selected_year)
        ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=0))

        lat_mask = (to_plot[self._lat_key] < 50) & (to_plot[self._lat_key] > -50)
        # Consider any points
        mean = to_plot.where(lat_mask).mean()
        std = to_plot.where(lat_mask).std()
        from_value = mean - std
        to_value = mean + std
        step = (to_value - from_value) / self._n_values
        # cmap = 'Blues'
        # self.print('Plotting...')
        # self.print(to_plot.dims)
        # to_plot = to_plot.dropna('latitude').dropna('longitude')
        im = ax.contourf(to_plot[self._lon_key], to_plot[self._lat_key], to_plot,
                         cmap=cmap,
                         levels=np.arange(from_value, to_value, step),
                         extend='both',
                         transform=ccrs.PlateCarree(central_longitude=0))
        ax.coastlines()
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.02)

        box = ax.get_tightbbox(fig.canvas.get_renderer()).bounds
        ratio = box[2] / box[3]
        fig.set_size_inches(8, 8 / ratio)
        # self.print(self._plot_data, selected_year)
        plt.suptitle(f'{self._variable.upper()}: {self._plot_data}', fontsize=15, weight='bold')
        ax.set_title(self._plot_bounds, fontsize=10)
        plt.tight_layout(pad=1)
        plt.margins(1, 1)

        if F.SAVE_FIG in flags:
            fig.savefig(os.path.join(self._plot_dir, self._plot_name))
        if F.SHOW_PLOT in flags:
            fig.show()

        return self


class Proker(ABC):
    @abstractmethod
    def apply(self, **kwargs: Any) -> 'Proker':
        raise NotImplementedError


class ClimerTS(PlotterTS):
    def apply(self, **_: Any) -> 'ClimerTS':
        self._data = self._data.mean(dim=self._lon_key).mean(dim=self._lat_key)
        self._data = Meteo.clim(self._data, dim='month')
        self._time_key = 'year'
        self._plot_data = 'CLIM ' + self._plot_data
        return self


class ClimerMap(PlotterMap):
    def apply(self, **_: Any) -> 'ClimerMap':
        self._data = Meteo.clim(self._data)
        self._plot_data = 'CLIM ' + self._plot_data
        return self


class AnomerTS(PlotterTS):
    def apply(self, **kwargs: Any) -> 'AnomerTS':
        st = kwargs.pop('st') if 'st' in kwargs else False
        if len(kwargs) != 0:
            raise TypeError('`apply` only accepts one keyword argument: `st`')
        # index 2 becomes 1 after doinf mean on index 1
        self._data = self._data.mean(dim=self._lon_key).mean(dim=self._lat_key)
        self._data = Meteo.anom(self._data, st)
        self._time_key = 'year'
        if st:
            self._plot_data += ' (Standarized)'
        self._plot_data = 'ANOM ' + self._plot_data
        return self


class AnomerMap(PlotterMap):
    def apply(self, **kwargs: Any) -> 'AnomerMap':
        st = kwargs.pop('st') if 'st' in kwargs else False
        if len(kwargs) != 0:
            raise TypeError('`apply` only accepts one keyword argument: `st`')
        # print(f"[INFO] <apply()> {plt_type=} {methodology=}, {kwargs})")
        self._data = Meteo.anom(self._data, st)
        self._time_key = 'year'
        self._slise.year0 = int(self.time[0])
        self._slise.yearf = int(self.time[-1])
        if st:
            self._plot_data += ' (Standarized)'
        self._plot_data = 'ANOM ' + self._plot_data
        return self

# if __name__ == '__main__':
#     matplotlib.use('TkAgg')
#     DATASET_DIR = '../../gbg'
#     PLOTS_DIR = '../../gbg/plots'
#     PREDICTAND_DSET = 'ecoocean_gfdl-reanalysis_hist_w-diaz_fishing_no-oa_b10cm_global_monthly_1971_2005.nc'
#     PREDICTAND_NAME = 'predictand.png'
#     PREDICTOR_DSET = 'gfdl_reanalysis_to_zs_monthly_1959-2004.nc4'
#     PREDICTOR_NAME = 'predictor.png'
#
#     spy = Spy4Caster(
#         RDArgs(dataset_dir= DATASET_DIR, dataset_name= PREDICTOR_DSET, variable= 'to'),
#         RDArgs(dataset_dir= DATASET_DIR, dataset_name= PREDICTAND_DSET, variable= 'b10cm'),
#         plot_dir= PLOTS_DIR, plot_name= PREDICTOR_NAME, plot_data_dir= DATASET_DIR, force_name= False,
#     )
#     yslis = Slise(-30, 30, -60, 0, 6, 8, 1959, 2004)
#     zslis = Slise(-30, 30, -60, 0, 6, 8, 1972, 2005)
#     spy.load_datasets()
#     spy.rdy._data = spy.rdy._data.squeeze('zt_ocean')
#     spy.slice_datasets(yslis, zslis)
#     spy.apply(order=8, period=5.5, nm=3, alpha=.1)
#     spy.run(F.SHOW_PLOT)
