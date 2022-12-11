import spy4cast as spy
from spy4cast.stypes import F, Month, Slise, RDArgs


DATASET_DIR = '/Users/Shared/datasets/'
PLOTS_DIR = 'plots'
PLOTS_DATA_DIR = 'saved_pac/'
MATS_PLOT_NAME = 'mats_spy4cast_chlorMED_pac.png'
MCA_PLOT_NAME = 'mca_spy4cast_chlorMED_pac.png'
CROSS_PLOT_NAME = 'cross_spy4cast_chlorMED_pac.png'
ZHAT_PLOT_NAME = 'zhat_spy4cast_chlorMED_pac.png'
chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'
# Clorofila perdictando

oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'  # Clorofila perdictando con años bien
CHL = 'CHL'
oisst_v2_mean_monthly_Jan1996_Dec2020 = 'oisst_v2_mean_monthly_Jan1996_Dec2020.nc'  # SST predictor
SST = 'sst'

def main() -> None:
    nm=3
    alpha=.1

    oisst_slise = Slise(
        lat0=-20, latf=25,
        lon0=-210, lonf=-60,
        month0=Month.OCT, monthf=Month.DEC,
        year0=1997, yearf=2019,
    )  # PREDICTOR: Y

    chl_slise = Slise(
        lat0=36, latf=37,
        lon0=-5.3, lonf=-2,
        month0=Month.MAR, monthf=Month.APR,
        year0=1998, yearf=2020,
    )  # PRECITAND: Z
    selected_year = 2006

    s = spy.Spy4Caster(
            yargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=oisst_v2_mean_monthly, variable=SST, chunks=100),
            zargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=chl_1km_monthly_Sep1997_Dec2020, variable=CHL, chunks=100),
            plot_dir=PLOTS_DIR, mats_plot_name=MATS_PLOT_NAME, mca_plot_name=MCA_PLOT_NAME, cross_plot_name=CROSS_PLOT_NAME, zhat_plot_name=ZHAT_PLOT_NAME,
            plot_data_dir=PLOTS_DATA_DIR)
    # TODO: Implement `ray` for multiprocessing in crossvalidation
    load = True
    if not load:
        s.open_datasets()
        s.slice_datasets(yslise=oisst_slise, zslise=chl_slise, yskip=0, zskip=0)
        s.preprocess()  # Primero sin filtro y luego con filtro de 8 años
        s.mca(nm=nm, alpha=alpha)
        s.crossvalidation(nm=nm, alpha=alpha, multiprocessing=False)
        s.run(show_plot=True, save_fig=True | F.SAVE_DATA, sy=selected_year, cmap='viridis')
    else:
        s.load_preprocessed(PLOTS_DATA_DIR, 'save_preprocessed_', '.npy')
        s.load_mca(PLOTS_DATA_DIR, 'save_mca_', '.npy')
        s.load_crossvalidation(PLOTS_DATA_DIR, 'save_cross_', '.npy')
        s.run(show_plot=True, save_fig=True, sy=selected_year, cmap='viridis')
    # s.plot_preprocessed()
    # s.plot_mca(show_plot=True, save_fig=True)
    # s.plot_mca(show_plot=True, save_fig=True)
    # s.plot_crossvalidation(show_plot=True, save_fig=True)
    # s.plot_zhat(show_plot=True, save_fig=True, sy=selected_year)


if __name__ == '__main__':
    spy.set_silence(False)
    main()