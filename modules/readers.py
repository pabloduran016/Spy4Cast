import os
import traceback
from abc import ABC, abstractmethod
from typing import Type, Optional, TypedDict, Any, Union
from enum import IntFlag, auto

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
import numpy as np

from custom_types import Methodology, PltType, Color, Slise
from errors import CustomError, PlotCreationError, DataSavingError, PlotSavingError, \
    PlotShowingError, PlotDataError, SelectedYearError
from meteo import Meteo, MCAOut
from read_data import ReadData, NAN_VAL



class F(IntFlag):
    SAVE_DATA = auto()
    SAVE_FIG = auto()
    TESTING = auto()
    SILENT_ERRORS = auto()
    SHOW_PLOT = auto()

    @staticmethod
    def checkf(f: 'F', other: Union[int, 'F']) -> bool:
        return (other & f) == f


# matplotlib.use('Agg')

'''
~~ INHERITANCE TREE ~~

    ReadData     Plotter (run, create)
        |        |-> Proker (apply) ------------------|--|--|--> 
         (-run, -create_plot, -apply)
        |--------|-> RDPlotter (-run) ----------------|  |--|--> ClimerTS, AnomerTS (-apply)
                     |-> PlotterTS (-create_plot) ----|--|  |--> ClimerMap, AnomerMap (-apply)
                     |-> PlotterMap (-create_plot) ---|-----|
'''


class Plotter(ABC):
    @abstractmethod
    def create_plot(self, fig: plt.figure, **__: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def run(self, flags: int = 0, **kwargs: Any) -> None:
        raise NotImplementedError()


class RDPlotter(ReadData, Plotter):
    @abstractmethod
    def create_plot(self, fig: plt.figure, **__: Any) -> None:
        raise NotImplementedError()

    def run(self, flags: int = 0, **kwargs: Any) -> None:
        # Save the data if needed
        if F.checkf(F.SAVE_DATA, flags):
            try:
                self.save_fig_data()
            except Exception as e:
                traceback.print_exc()
                if not F.checkf(F.SILENT_ERRORS, flags):
                    raise DataSavingError(str(e)) from e

        # Create the plot
        fig = plt.figure()
        if F.checkf(F.SHOW_PLOT, flags) or F.checkf(F.SAVE_FIG, flags) or F.checkf(F.TESTING, flags):
            try:
                self.create_plot(fig, **kwargs)
            except CustomError:
                raise
            except Exception as e:
                traceback.print_exc()
                if not F.checkf(F.SILENT_ERRORS, flags):
                    raise PlotCreationError(str(e)) from e

        # Save the fig0 if needed
        if F.checkf(F.SAVE_FIG, flags):
            try:
                self.print(f"[INFO] <{self.__class__.__name__}> Saving plot as {self.plot_name} in {self.plot_dir}")
                # Generate a PNG of the figure
                plt.savefig(os.path.join(self.plot_dir, self.plot_name))
            except CustomError:
                raise
            except Exception as e:
                traceback.print_exc()
                if not F.checkf(F.SILENT_ERRORS, flags):
                    raise PlotSavingError(str(e)) from e

        # Show the plot if needed
        try:
            if F.checkf(F.SHOW_PLOT, flags):
                plt.show()
        except CustomError:
            raise
        except Exception as e:
            traceback.print_exc()
            if not F.checkf(F.SILENT_ERRORS, flags):
                raise PlotShowingError(str(e)) from e
        plt.close()


class Proker(Plotter):
    """Read + Plot + Process data"""
    @abstractmethod
    def apply(self, **kwargs: Any) -> None:
        raise NotImplementedError()


class RDProker(RDPlotter, Proker):
    pass


def plotter_factory(m: Optional[Methodology], pt: PltType) -> Type[RDPlotter]:
    if m == Methodology.CLIM:
        if pt == PltType.TS:
            return ClimerTS
        if pt == PltType.MAP:
            return ClimerMap
    elif m == Methodology.ANOM:
        if pt == PltType.TS:
            return AnomerTS
        if pt == PltType.MAP:
            return AnomerMap
    # elif m == Methodology.SPY4CAST and pt is None:
    #     return Spy4Caster
    elif m is None:
        if pt == PltType.TS:
            return PlotterTS
        if pt == PltType.MAP:
            return PlotterMap
    raise ValueError('Unknown methodology and/or plt type')


class PlotterTS(RDPlotter):
    def create_plot(self, fig: plt.figure, **kws: Any) -> None:
        color: Color = (.43, .92, .20) if 'color' not in kws else kws['color']
        self.print(f"[INFO] <{self.__class__.__name__}> Creating plot for {self.plot_name}")
        # Create figure
        assert len(self.var_data.shape) == 1, 'Time series arrays must be unidimensional'
        to_plot = self.var_data.where(lambda a: abs(a) < NAN_VAL)
        ax = fig.add_subplot()
        ax.plot(to_plot[self.time_key], to_plot, linewidth=3, color=color)
        ax.set_xlim(to_plot[self.time_key][0], to_plot[self.time_key][-1])
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{self.variable} ({self.dataset[self.variable].units})')
        # ax.set_ylim(np.min(self.var_data), np.max(self.var_data))


class PlotterMap(RDPlotter):
    n_values = 50

    def create_plot(self, fig: plt.figure, **kws: Any) -> None:
        if 'slise' not in kws:
            raise TypeError("create_plot() missing 1 required positional argument: 'slise'")
        slise: Slise = kws['slise']
        cmap: str = kws['cmap'] if 'cmap' in kws else 'jet'
        # self.print('Preparing dataset for plotting...')
        if len(self.var_data.shape) == 3:
            assert slise is not None and slise.selected_year is not None, 'Expected selected year inside of a slise'
            if not self.slice_initial_year <= slise.selected_year <= self.slice_final_year:
                raise SelectedYearError(slise.selected_year)
            self.plot_data += f' ({slise.selected_year})'
            self.time_key = self.time_key if self.time_key in self.var_data.dims else 'year'
            # .reset_index()?
            to_plot = self.var_data.sel({self.time_key: slise.selected_year})
            # to_plot = self.var_data.sel(year=selected_year)
        else:
            to_plot = self.var_data

        # Convert to NaN any number above |NAN_VAL|
        to_plot = to_plot.where(lambda a: abs(a) < NAN_VAL).sortby(self.lon_key)

        if to_plot.shape[0] < 2 or to_plot.shape[1] < 2:
            raise PlotDataError(f'Slicing Error, got {to_plot.shape} as slice shapes')

        # self.print(self.plot_data, selected_year)
        ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=0))

        lat_mask = (to_plot[self.lat_key] < 50) & (to_plot[self.lat_key] > -50)
        # Consider any points
        mean = to_plot.where(lat_mask).mean()
        std = to_plot.where(lat_mask).std()
        from_value = mean - std
        to_value = mean + std
        step = (to_value - from_value) / self.n_values
        # cmap = 'Blues'
        # self.print('Plotting...')
        # self.print(to_plot.dims)
        # to_plot = to_plot.dropna('latitude').dropna('longitude')
        im = ax.contourf(to_plot[self.lon_key], to_plot[self.lat_key], to_plot,
                         cmap=cmap,
                         levels=np.arange(from_value, to_value, step),
                         extend='both',
                         transform=ccrs.PlateCarree(central_longitude=0))
        ax.coastlines()
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.02)

        box = ax.get_tightbbox(fig.canvas.get_renderer()).bounds
        ratio = box[2] / box[3]
        fig.set_size_inches(8, 8 / ratio)
        # self.print(self.plot_data, selected_year)
        plt.suptitle(f'{self.variable.upper()}: {self.plot_data}', fontsize=15, weight='bold')
        ax.set_title(self.plot_bounds, fontsize=10)
        plt.tight_layout(pad=1)
        plt.margins(1, 1)


class ClimerTS(Proker, PlotterTS):
    def apply(self, **_: Any) -> None:
        self.var_data = self.var_data.mean(dim=self.lon_key).mean(dim=self.lat_key)
        self.var_data = Meteo.clim(self.var_data, dim='month')
        self.time_key = 'year'
        self.plot_data = 'CLIM ' + self.plot_data


class ClimerMap(Proker, PlotterMap):
    def apply(self, **_: Any) -> None:
        self.var_data = Meteo.clim(self.var_data)
        self.plot_data = 'CLIM ' + self.plot_data


class AnomerTS(Proker, PlotterTS):
    def apply(self, **kwargs: Any) -> None:
        st = kwargs['st'] if 'st' in kwargs else False
        # index 2 becomes 1 after doinf mean on index 1
        self.var_data = self.var_data.mean(dim=self.lon_key).mean(dim=self.lat_key)
        self.var_data = Meteo.anom(self.var_data, st)
        self.time_key = 'year'
        if st:
            self.plot_data += ' (Standarized)'
        self.plot_data = 'ANOM ' + self.plot_data


class AnomerMap(Proker, PlotterMap):
    def apply(self, **kwargs: Any) -> None:
        st = kwargs['st'] if 'st' in kwargs else False
        # print(f"[INFO] <apply()> {plt_type=} {methodology=}, {kwargs})")
        self.var_data = Meteo.anom(self.var_data, st)
        self.time_key = 'year'
        self.slise.initial_year = int(self.var_data[self.time_key][0])
        self.slise.final_year = int(self.var_data[self.time_key][-1])
        if st:
            self.plot_data += ' (Standarized)'
        self.plot_data = 'ANOM ' + self.plot_data


RDArgs = TypedDict('RDArgs', {'dataset_dir': str, 'dataset_name': str, 'variable': str, 'plot_dir': str,
                              'plot_name': str, 'plot_data_dir': str, 'force_name': bool})


class Spy4Caster(Proker):
    def __init__(self, yargs: RDArgs, zargs: RDArgs):
        self.rdy = ReadData(**yargs)  # Predictor
        self.rdz = ReadData(**zargs)  # Predictand
        self.mca_out: Optional[MCAOut] = None

    def load_datasets(self) -> None:
        self.rdy.load_dataset()
        self.rdz.load_dataset()
        self.rdy.var_data = self.rdy.var_data.where(lambda a: abs(a) < NAN_VAL).sortby(self.rdy.lon_key)
        self.rdz.var_data = self.rdz.var_data.where(lambda a: abs(a) < NAN_VAL).sortby(self.rdz.lon_key)

    def slice_datasets(self, yslise: Slise, zslise: Slise) -> None:
        self.rdy.slice_dataset(yslise)
        self.rdz.slice_dataset(zslise)

    def apply(self, **kws: Union[float, int]) -> None:
        for i in ['order', 'period', 'nm', 'alpha']:
            if i not in kws:
                raise TypeError(f'missing argument {i} for apply()')
        assert type(kws['order']) == int
        assert type(kws['float']) == float
        assert type(kws['nm']) == int
        assert type(kws['alpha']) == float
        order: int = kws['order']
        period: float = kws['period']
        nm: int = kws['nm']
        alpha: float = kws['alpha']
        self.rdy.var_data = Meteo.anom(self.rdy.var_data)
        self.rdz.var_data = Meteo.anom(self.rdz.var_data)

        y0 = max(self.rdy.slise.initial_year, self.rdz.slise.initial_year)
        yf = min(self.rdy.slise.final_year, self.rdz.slise.final_year)

        self.rdy.slice_dataset(Slise.default(initial_year=y0, final_year=yf))
        self.rdz.slice_dataset(Slise.default(initial_year=y0, final_year=yf))

        z = self.rdz.var_data.values
        # zlon = self.rdz.var_data.lon.values
        # zlat = self.rdz.var_data.lat.values
        ztrans = np.reshape(z, (z.shape[0], z.shape[1] * z.shape[2])).transpose()

        y = self.rdy.var_data.values
        # ylon = self.rdy.var_data.longitude.values
        # ylat = self.rdy.var_data.latitude.values
        ytrans = np.reshape(y, (y.shape[0], y.shape[1] * y.shape[2])).transpose()

        b, a = signal.butter(order, 1/period, btype='high', analog=False, output='ba', fs=None)

        # Filtro la señal ampliada y me quedo con la parte central:
        zmask = np.ma.empty(ztrans.shape)
        for index in range(ztrans.shape[0]):
            zmask[index, :] = signal.filtfilt(b, a, ztrans[i, :])

        # zmask, zlon, zlat; ytrans, ylon, ylat
        self.mca_out = Meteo.mca(zmask[:, 1:], ytrans[:, :-1], 1, nm, alpha)
        ...

    def create_plot(self, fig: plt.figure, **__: Any) -> None:
        if self.mca_out is None:
            raise ValueError("The methodology hasn't been applied yet")

        _, (ax10, ax11, ax12), (ax20, ax21, ax22) = fig.subplots(nrows=3, ncols=3,
                                                                 subplot_kw={'projection': ccrs.PlateCarree()})
        ax00 = plt.subplot(331)
        ax01 = plt.subplot(332)
        ax02 = plt.subplot(333)

        ax00.margins(0)
        ax00.plot(np.arange(1972, 2004), self.mca_out.Us[0, :], color='green', label=f'Us 0')
        ax00.plot(np.arange(1972, 2004), self.mca_out.Vs[0, :], color='blue', label=f'Vs 0')
        ax00.legend(loc='upper left')
        ax00.grid(True)
        ax00.set_title('Us Vs mode 0')

        ax01.margins(0)
        ax01.plot(np.arange(1972, 2004), self.mca_out.Us[1, :], color='green', label=f'Us 1')
        ax01.plot(np.arange(1972, 2004), self.mca_out.Vs[1, :], color='blue', label=f'Vs 1')
        ax01.legend(loc='upper left')
        ax01.grid(True)
        ax01.set_title('Us Vs mode 1')

        ax02.margins(0)
        ax02.plot(np.arange(1972, 2004), self.mca_out.Us[2, :], color='green', label=f'Us 2')
        ax02.plot(np.arange(1972, 2004), self.mca_out.Vs[2, :], color='blue', label=f'Vs 2')
        ax02.legend(loc='upper left')
        ax02.grid(True)
        ax02.set_title('Us Vs mode 2')

        # --------------

        t10 = self.mca_out.RUY[:, 0].transpose().reshape((len(self.rdy.var_data[self.rdy.lat_key]),
                                                             len(self.rdy.var_data[self.rdy.lon_key])))
        im10 = ax10.contourf(self.rdy.var_data[self.rdy.lon_key], self.rdy.var_data[self.rdy.lat_key], t10,
                             cmap='jet', transform=ccrs.PlateCarree())
        fig.colorbar(im10, ax=ax10, orientation='horizontal')
        ax10.margins(0)
        ax10.coastlines()
        ax10.set_title('RUY mode 0')

        t11 = self.mca_out.RUY[:, 1].transpose().reshape((len(self.rdy.var_data[self.rdy.lat_key]),
                                                             len(self.rdy.var_data[self.rdy.lon_key])))
        im11 = ax11.contourf(self.rdy.var_data[self.rdy.lon_key], self.rdy.var_data[self.rdy.lat_key], t11,
                             cmap='jet', transform=ccrs.PlateCarree())
        fig.colorbar(im11, ax=ax11, orientation='horizontal')
        ax11.margins(0)
        ax11.coastlines()
        ax11.set_title('RUY mode 1')

        t12 = self.mca_out.RUY[:, 2].transpose().reshape((len(self.rdy.var_data[self.rdy.lat_key]),
                                                             len(self.rdy.var_data[self.rdy.lon_key])))
        im12 = ax12.contourf(self.rdy.var_data[self.rdy.lon_key], self.rdy.var_data[self.rdy.lat_key], t12,
                             cmap='jet', transform=ccrs.PlateCarree())
        fig.colorbar(im12, ax=ax12, orientation='horizontal')
        ax12.margins(0)
        ax12.coastlines()
        ax12.set_title('RUY mode 2')

        # ----------

        t20 = self.mca_out.RUZ[:, 0].transpose().reshape((len(self.rdz.var_data[self.rdz.lat_key]),
                                                             len(self.rdz.var_data[self.rdz.lon_key])))
        im20 = ax20.contourf(self.rdz.var_data[self.rdz.lon_key], self.rdz.var_data[self.rdz.lat_key], t20,
                             cmap='jet', transform=ccrs.PlateCarree())
        fig.colorbar(im20, ax=ax20, orientation='horizontal')
        ax20.margins(0)
        ax20.coastlines()
        ax20.set_title('RUZ mode 0')

        t21 = self.mca_out.RUZ[:, 1].transpose().reshape((len(self.rdz.var_data[self.rdz.lat_key]),
                                                             len(self.rdz.var_data[self.rdz.lon_key])))
        im21 = ax21.contourf(self.rdz.var_data[self.rdz.lon_key], self.rdz.var_data[self.rdz.lat_key], t21,
                             cmap='jet', transform=ccrs.PlateCarree())
        fig.colorbar(im21, ax=ax21, orientation='horizontal')
        ax21.margins(0)
        ax21.coastlines()
        ax21.set_title('RUZ mode 1')

        t22 = self.mca_out.RUZ[:, 2].transpose().reshape((len(self.rdz.var_data[self.rdz.lat_key]),
                                                             len(self.rdz.var_data[self.rdz.lon_key])))
        im22 = ax22.contourf(self.rdz.var_data[self.rdz.lon_key], self.rdz.var_data[self.rdz.lat_key], t22,
                             cmap='jet', transform=ccrs.PlateCarree())
        fig.colorbar(im22, ax=ax22, orientation='horizontal')
        ax22.margins(0)
        ax22.coastlines()
        ax22.set_title('RUZ mode 2')

        plt.tight_layout()

    def save_fig_data(self) -> None:
        raise NotImplementedError()

    def run(self, flags: int = 0, **kwargs: Any) -> None:
        # Save the data if needed
        if F.checkf(F.SAVE_DATA, flags):
            try:
                self.save_fig_data()
            except Exception as e:
                traceback.print_exc()
                if F.checkf(F.SILENT_ERRORS, flags):
                    raise DataSavingError(str(e)) from e

        # Create the plot
        fig = plt.figure()
        if F.checkf(F.SHOW_PLOT, flags) or F.checkf(F.SAVE_FIG, flags) or F.checkf(F.TESTING, flags):
            try:
                self.create_plot(fig, **kwargs)
            except CustomError:
                raise
            except Exception as e:
                traceback.print_exc()
                if F.checkf(F.SILENT_ERRORS, flags):
                    raise PlotCreationError(str(e)) from e

        # Save the fig0 if needed
        if F.checkf(F.SAVE_FIG, flags):
            try:
                self.rdz.print(f"[INFO] <{self.rdz.__class__.__name__}> Saving plot as {self.rdz.plot_name} "
                               f"in {self.rdz.plot_dir}")
                # Generate a PNG of the figure
                plt.savefig(os.path.join(self.rdz.plot_dir, self.rdz.plot_name))
            except CustomError:
                raise
            except Exception as e:
                traceback.print_exc()
                if F.checkf(F.SILENT_ERRORS, flags):
                    raise PlotSavingError(str(e)) from e

        # Show the plot if needed
        try:
            if F.checkf(F.SHOW_PLOT, flags):
                plt.show()
        except CustomError:
            raise
        except Exception as e:
            traceback.print_exc()
            if not F.checkf(F.SILENT_ERRORS, flags):
                raise PlotShowingError(str(e)) from e
        plt.close()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    DATASET_DIR = '../../gbg'
    PLOTS_DIR = '../../gbg/plots'
    PREDICTAND_DSET = 'ecoocean_gfdl-reanalysis_hist_w-diaz_fishing_no-oa_b10cm_global_monthly_1971_2005.nc'
    PREDICTAND_NAME = 'predictand.png'
    PREDICTOR_DSET = 'gfdl_reanalysis_to_zs_monthly_1959-2004.nc4'
    PREDICTOR_NAME = 'predictor.png'

    spy = Spy4Caster(
        {'dataset_dir': DATASET_DIR, 'dataset_name': PREDICTOR_DSET, 'variable': 'to',
         'plot_dir': PLOTS_DIR, 'plot_name': PREDICTOR_NAME, 'plot_data_dir': DATASET_DIR, 'force_name': False},
        {'dataset_dir': DATASET_DIR, 'dataset_name': PREDICTAND_DSET, 'variable': 'b10cm',
         'plot_dir': PLOTS_DIR, 'plot_name': PREDICTAND_NAME, 'plot_data_dir': DATASET_DIR, 'force_name': False},
    )
    yslis = Slise(-30, 30, -60, 0, 6, 8, 1959, 2004)
    zslis = Slise(-30, 30, -60, 0, 6, 8, 1972, 2005)
    spy.load_datasets()
    spy.rdy.var_data = spy.rdy.var_data.squeeze('zt_ocean')
    spy.slice_datasets(yslis, zslis)
    spy.apply(order=8, period=5.5, nm=3, alpha=.1)
    spy.run(F.SHOW_PLOT)
