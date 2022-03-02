import spy4cast as spy
from spy4cast import set_silence
from spy4cast.stypes import F, Slise, Month, RDArgs

DATASET_DIR = '/Users/Shared/datasets/'
PLOTS_DIR = 'plots'
PLOTS_DATA_DIR = 'saved-anom-2006/'
PLOT_NAME = 'anom_chl_2006.png'
chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'  # Clorofila perdictando
oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'  # Clorofila perdictando con años bien
CHL = 'CHL'
oisst_v2_mean_monthly_Jan1996_Dec2020 = 'oisst_v2_mean_monthly_Jan1996_Dec2020.nc'  # SST predictor
SST = 'sst'


def anom() -> None:
    a = spy.AnomerMap(DATASET_DIR, chl_1km_monthly_Sep1997_Dec2020, CHL, PLOTS_DIR, PLOT_NAME)
    a.open_dataset()
    slise = Slise(
        lat0=36, latf=37,
        lon0=-5.3, lonf=-2,
        month0=Month.MAR, monthf=Month.APR,
        year0=1998, yearf=2020, sy=2006,
    )
    a.slice_dataset(slise)
    a.apply()
    a.run(F.SHOW_PLOT, slise=slise, cmap='viridis')


def spy4cast() -> None:
    s = spy.Spy4Caster(
        yargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=oisst_v2_mean_monthly, variable=SST, chunks=100),
        zargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=chl_1km_monthly_Sep1997_Dec2020, variable=CHL, chunks=100),
        plot_dir=PLOTS_DIR, plot_data_dir=PLOTS_DATA_DIR, plot_data_sufix='_spy')
    oisst_slise = Slise(
        lat0=-20, latf=25,
        lon0=-210, lonf=-60,
        month0=Month.OCT, monthf=Month.DEC,
        year0=1997, yearf=2019,
    )
    chl_slise = Slise(
        lat0=36, latf=37,
        lon0=-5.3, lonf=-2,
        month0=Month.MAR, monthf=Month.APR,
        year0=1998, yearf=2020, sy=2006,
    )
    # s.open_datasets()
    # s.slice_datasets(yslise=oisst_slise, zslise=chl_slise, yskip=0, zskip=0)
    s.load_preprocessed(PLOTS_DATA_DIR, 'save_preprocessed_spy_', '.npy')
    # s.preprocess()  # Primero sin filtro y luego con filtro de 8 años
    # s.save_fig_data()
    s.plot_preprocessed(F.SHOW_PLOT | F.NOT_HALT, selected_year=2006, cmap='viridis')


if __name__ == '__main__':
    set_silence(False)
    spy4cast()
    anom()