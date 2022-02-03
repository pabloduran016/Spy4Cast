import os
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Type, Optional, Any, Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy import signal
# import matplotlib
import numpy as np

from .stypes import Methodology, PltType, Color, Slise, F, RDArgs, RDArgsDict
from .errors import CustomError, PlotCreationError, DataSavingError, PlotSavingError, \
    PlotShowingError, PlotDataError, SelectedYearError
from .meteo import Meteo, MCAOut, CrossvalidationOut
from .read_data import ReadData, NAN_VAL


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
    def create_plot(self: 'Plotter', fig: plt.figure, **__: Any) -> 'Plotter':
        raise NotImplementedError()

    @abstractmethod
    def run(self: 'Plotter', flags: int = 0, **kwargs: Any) -> 'Plotter':
        raise NotImplementedError()


class RDPlotter(ReadData, Plotter):
    @abstractmethod
    def create_plot(self, fig: plt.figure, **__: Any) -> 'RDPlotter':
        raise NotImplementedError()

    def run(self, flags: int = 0, **kwargs: Any) -> 'RDPlotter':
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
                print(f"[INFO] <{self.__class__.__name__}> Saving plot as {self._plot_name} in {self._plot_dir}")
                # Generate a PNG of the figure
                plt.savefig(os.path.join(self._plot_dir, self._plot_name))
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

        return self


class Proker(Plotter):
    """Read + Plot + Process data"""
    @abstractmethod
    def apply(self: 'Proker', **kwargs: Any) -> 'Proker':
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
    def create_plot(self, fig: plt.figure, **kws: Any) -> 'PlotterTS':
        color: Color = (.43, .92, .20) if 'color' not in kws else kws['color']
        print(f"[INFO] <{self.__class__.__name__}> Creating plot for {self._plot_name}")
        # Create figure
        assert len(self.shape) == 1, 'Time series arrays must be unidimensional'
        to_plot = self._data.where(lambda a: abs(a) < NAN_VAL)
        ax = fig.add_subplot()
        ax.plot(to_plot[self._time_key], to_plot, linewidth=3, color=color)
        ax.set_xlim(to_plot[self._time_key][0], to_plot[self._time_key][-1])
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{self._variable} ({self._dataset[self._variable].units})')
        # ax.set_ylim(np.min(self._data), np.max(self._data))
        return self


class PlotterMap(RDPlotter):
    _n_values = 50

    def create_plot(self, fig: plt.figure, **kws: Any) -> 'PlotterMap':
        if 'slise' not in kws:
            raise TypeError("create_plot() missing 1 required positional argument: 'slise'")
        slise: Slise = kws['slise']
        cmap: str = kws['cmap'] if 'cmap' in kws else 'jet'
        # self.print('Preparing dataset for plotting...')
        if len(self.shape) == 3:
            assert slise is not None and slise.selected_year is not None, 'Expected selected year inside of a slise'
            if not self._slise.initial_year <= slise.selected_year <= self._slise.final_year:
                raise SelectedYearError(slise.selected_year)
            self._plot_data += f' ({slise.selected_year})'
            self._time_key = self._time_key if self._time_key in self._data.dims else 'year'
            # .reset_index()?
            to_plot = self._data.sel({self._time_key: slise.selected_year})
            # to_plot = self._data.sel(year=selected_year)
        else:
            to_plot = self._data

        # Convert to NaN any number above |NAN_VAL|
        to_plot = to_plot.where(lambda a: abs(a) < NAN_VAL).sortby(self._lon_key)

        if to_plot.shape[0] < 2 or to_plot.shape[1] < 2:
            raise PlotDataError(f'Slicing Error, got {to_plot.shape} as slice shapes')

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

        return self


class ClimerTS(Proker, PlotterTS):
    def apply(self, **_: Any) -> 'ClimerTS':
        self._data = self._data.mean(dim=self._lon_key).mean(dim=self._lat_key)
        self._data = Meteo.clim(self._data, dim='month')
        self._time_key = 'year'
        self._plot_data = 'CLIM ' + self._plot_data
        return self


class ClimerMap(Proker, PlotterMap):
    def apply(self, **_: Any) -> 'ClimerMap':
        self._data = Meteo.clim(self._data)
        self._plot_data = 'CLIM ' + self._plot_data
        return self


class AnomerTS(Proker, PlotterTS):
    def apply(self, **kwargs: Any) -> 'AnomerTS':
        st = kwargs['st'] if 'st' in kwargs else False
        # index 2 becomes 1 after doinf mean on index 1
        self._data = self._data.mean(dim=self._lon_key).mean(dim=self._lat_key)
        self._data = Meteo.anom(self._data, st)
        self._time_key = 'year'
        if st:
            self._plot_data += ' (Standarized)'
        self._plot_data = 'ANOM ' + self._plot_data
        return self


class AnomerMap(Proker, PlotterMap):
    def apply(self, **kwargs: Any) -> 'AnomerMap':
        st = kwargs['st'] if 'st' in kwargs else False
        # print(f"[INFO] <apply()> {plt_type=} {methodology=}, {kwargs})")
        self._data = Meteo.anom(self._data, st)
        self._time_key = 'year'
        self._slise.initial_year = int(self.time[0])
        self._slise.final_year = int(self.time[-1])
        if st:
            self._plot_data += ' (Standarized)'
        self._plot_data = 'ANOM ' + self._plot_data
        return self


class Spy4Caster(Proker):
    def __init__(self, yargs: Union[RDArgs, RDArgsDict], zargs: Union[RDArgs, RDArgsDict],
                 plot_dir: str = '', plot_name: str = 'plot.png', plot_data_dir: str = '',
                 force_name: bool = False):
        if type(yargs) == RDArgs:
            yargs = yargs.as_dict()
        if type(zargs) == RDArgs:
            zargs = zargs.as_dict()
        assert type(yargs) == dict
        assert type(zargs) == dict
        self.rdy = ReadData(**yargs)  # Predictor
        self.rdz = ReadData(**zargs)  # Predictand
        self.mca_out: Optional[MCAOut] = None
        self.crossvalidation_out: Optional[CrossvalidationOut] = None
        self._plot_dir = plot_dir
        self._plot_name = plot_name
        self._plot_data_dir = plot_data_dir
        self.force_name = force_name

    def load_datasets(self) -> 'Spy4Caster':
        self.rdy.load_dataset()
        self.rdz.load_dataset()
        self.rdy._data = self.rdy._data.where(lambda a: abs(a) < NAN_VAL).sortby(self.rdy._lon_key)
        self.rdz._data = self.rdz._data.where(lambda a: abs(a) < NAN_VAL).sortby(self.rdy._lon_key)
        return self

    def slice_datasets(self, yslise: Slise, zslise: Slise) -> 'Spy4Caster':
        self.rdy.slice_dataset(yslise)
        self.rdz.slice_dataset(zslise)
        return self

    def apply(self, **kws: Union[float, int]) -> 'Spy4Caster':
        for _i in ['order', 'period', 'nm', 'alpha']:
            if _i not in kws:
                raise TypeError(f'missing argument {_i} for apply()')
        del _i
        if type(kws['order']) != int:
            raise TypeError('Type of `order` should be `int`')
        if type(kws['period']) != float and type(kws['period']) != int:
            raise TypeError('Type of `period` should be `int`')
        if type(kws['nm']) != int:
            raise TypeError('Type of `nm` should be `int`')
        if type(kws['alpha']) != float:
            raise TypeError('Type of `alpha` should be `float`')
        order: int = kws['order']
        period: float = kws['period']
        nm: int = kws['nm']
        alpha: float = kws['alpha']
        print(f"[INFO] <{self.__class__.__name__}> Applying Spy4Cast: Y: {self.rdy._dataset_name}, "
              f"Z: {self.rdz._dataset_name}")
        self.rdy._data = Meteo.anom(self.rdy._data)
        self.rdy._time_key = 'year'
        self.rdz._data = Meteo.anom(self.rdz._data)
        self.rdz._time_key = 'year'

        y0 = max(self.rdy._slise.initial_year, self.rdz._slise.initial_year)
        yf = min(self.rdy._slise.final_year, self.rdz._slise.final_year)

        self.rdy.slice_dataset(Slise.default(initial_year=y0, final_year=yf))
        self.rdz.slice_dataset(Slise.default(initial_year=y0, final_year=yf))

        z = self.rdz._data.values
        # zlon = self.rdz._data.lon.values
        # zlat = self.rdz._data.lat.values
        ztrans = np.reshape(z, (z.shape[0], z.shape[1] * z.shape[2])).transpose()

        y = self.rdy._data.values
        # ylon = self.rdy._data.longitude.values
        # ylat = self.rdy._data.latitude.values
        ytrans = np.reshape(y, (y.shape[0], y.shape[1] * y.shape[2])).transpose()

        b, a = signal.butter(order, 1/period, btype='high', analog=False, output='ba', fs=None)

        # Filtro la señal ampliada y me quedo con la parte central:
        zmask = np.ma.empty(ztrans.shape)
        for index in range(ztrans.shape[0]):
            zmask[index, :] = signal.filtfilt(b, a, ztrans[index, :])

        # zmask, zlon, zlat; ytrans, ylon, ylat
        ynotnan = np.nan_to_num(ytrans)  # fill nan with 0
        znotnan = np.nan_to_num(zmask)  # fill nan with 0
        self.mca_out = Meteo.mca(znotnan[:, 1:], ynotnan[:, :-1], 1, nm, alpha)

        # self.crossvalidation_out = Meteo.crossvalidation(y, z, 1, nm, alpha)

        return self

    def plot_matrices(self):
        # sst = self.rdy._data.values
        # sst_lon = self.rdy.lon
        # sst_lat = self.rdy.lat
        # sst_time = self.rdy.time
        # slp = self.rdz._data.values
        # slp_lat = self.rdz.lat
        # slp_lon = self.rdz.lon
        # slp_time = self.rdz.time
        #
        # def plot_matrices(y, ytime, ylats, ylons, z, ztime, zlats, zlons):
        #     fig = plt.figure()
        #     ax0, ax1 = fig.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
        #     t0 = z[0, :, :]
        #     im0 = ax0.contourf(zlons, zlats, t0, cmap='viridis', transform=ccrs.PlateCarree())
        #     fig.colorbar(im0, ax=ax0, orientation='horizontal', pad=0.02)
        #     ax0.coastlines()
        #     t1 = y[10, :, :]
        #     im1 = ax1.contourf(ylons, ylats, t1, cmap='viridis', transform=ccrs.PlateCarree())
        #     fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.02)
        #     ax1.coastlines()
        #     plt.show()
        #
        # plot_matrices(sst, sst_time, sst_lat, sst_lon, slp, slp_time, slp_lat, slp_lon)

        fig = plt.figure()
        ax0, ax1 = fig.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
        print(self.rdy._data.values)
        t0 = self.rdz._data.values[0, :, :]
        im0 = ax0.contourf(self.rdz.lon, self.rdz.lat,
                           t0, cmap='viridis', transform=ccrs.PlateCarree())
        fig.colorbar(im0, ax=ax0, orientation='horizontal', pad=0.02)
        ax0.coastlines()

        t1 = self.rdy._data.values[0, :, :]
        im1 = ax1.contourf(self.rdy.lon, self.rdy.lat,
                           t1, cmap='viridis', transform=ccrs.PlateCarree())
        fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.02)
        ax1.coastlines()
        plt.show()

        return self

    def plot_mca(self, fig: plt.Figure) -> 'Spy4Caster':
        if self.mca_out is None:
            print('[WARNING] Can not plot mca. Methodology was not applied yet')
            return self

        time = self.rdy.time
        ylats = self.rdy.lat
        ylons = self.rdy.lon
        zlats = self.rdz.lat
        zlons = self.rdz.lon

        nrows = 3
        ncols = 3
        axs = [plt.subplot(nrows*100 + ncols*10 + i, projection=(ccrs.PlateCarree() if i > 3 else None))
               for i in range(1, ncols * nrows + 1)]

        # Plot timeseries
        for i, ax in enumerate(axs[:3]):
            ax.margins(0)
            ax.plot(time[:-1], self.mca_out.Us[i, :], color='green', label=f'Us')
            ax.plot(time[:-1], self.mca_out.Vs[i, :], color='blue', label=f'Vs')
            ax.grid(True)
            ax.set_title(f'Us Vs mode {i}')
        axs[0].legend(loc='upper left')

        n = 20
        for i, (name, su, ru, lats, lons) in enumerate((
                ('SUY', self.mca_out.SUY, self.mca_out.RUY_sig, ylats, ylons),
                ('SUZ', self.mca_out.SUZ, self.mca_out.RUZ_sig, zlats, zlons)
        )):
            if i == 0:
                su[su == 0.0] = np.nan
            _std = np.nanstd(su)
            levels = np.linspace(-_std, _std, n)
            xlim = lons[0], lons[-1]
            ylim = lats[-1], lats[0]
            for j, ax in enumerate(axs[3*(i + 1):3*(i + 1)+3]):
                t = su[:, j].transpose().reshape((len(lats), len(lons)))
                th = ru[:, j].transpose().reshape((len(lats), len(lons)))
                im = ax.contourf(lons, lats, t, cmap='bwr', levels=levels,
                                 extend='both', transform=ccrs.PlateCarree())
                _imh = ax.contourf(lons, lats, th, colors='none', hatches='..', extend='both',
                                   transform=ccrs.PlateCarree())
                _cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                                   ticks=np.concatenate((levels[::n // 4], levels[-1:len(levels)])))
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.margins(0)
                ax.coastlines()
                ax.set_title(f'{name} mode {j}')

        plt.tight_layout()
        return self

    def plot_crossvalidation(self, _fig: plt.Figure) -> 'Spy4Caster':
        print('[WARNING] Crossvalidation Plot is not implemented yet', file=sys.stderr)
        return self

    def create_plot(self, fig: plt.figure, **__: Any) -> 'Spy4Caster':
        if self.mca_out is None:  # or self.crossvalidation_out is None:
            raise ValueError("The methodology hasn't been applied yet")
        self.plot_mca(fig)
        # self.plot_crossvalidation(fig)
        return self

    @staticmethod
    def save_output(name: str, variables: Union[MCAOut, CrossvalidationOut]):
        for k, v in variables.__dict__.items():
            if type(v) == np.ma.MaskedArray:
                v = v.data
            try:
                np.save(name + '_' + k, v)
            except Exception:
                traceback.print_exc()

    def save_fig_data(self) -> 'Spy4Caster':
        if self.mca_out is not None:
            self.save_output('saved/save_mca', self.mca_out)
        else:
            print('[WARNING] No MCA data to save', file=sys.stderr)

        if self.crossvalidation_out is not None:
            self.save_output('saved/save_cross', self.crossvalidation_out)
        else:
            print('[WARNING] No Crossvalidation data to save', file=sys.stderr)

        return self

    def run(self, flags: int = 0, **kwargs: Any) -> 'Spy4Caster':
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
                print(f"[INFO] <{self.__class__.__name__}> Saving plot as {self._plot_name} "
                      f"in {self._plot_dir}")
                # Generate a PNG of the figure
                plt.savefig(os.path.join(self._plot_dir, self._plot_name))
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
