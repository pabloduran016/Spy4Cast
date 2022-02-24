import spy4cast as spy
from spy4cast.stypes import F, Month, Slise, RDArgs


DATASET_DIR = '/Users/Shared/datasets/'
PLOTS_DIR = ''
PLOTS_DATA_DIR = 'saved_2/'
MATS_PLOT_NAME = 'mats_spy4cast_example_2.png'
MCA_PLOT_NAME = 'mca_spy4cast_example_2.png'
CROSS_PLOT_NAME = 'cross_spy4cast_example_2.png'
ZHAT_PLOT_NAME = 'zhat_spy4cast_example_2.png'
chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'  # Clorofila perdictando
CHL = 'CHL'
oisst_v2_mean_monthly_Jan1996_Dec2020 = 'oisst_v2_mean_monthly_Jan1996_Dec2020.nc'  # SST predictor
SST = 'sst'

def main():
    nm=3
    alpha=.1

    oisst_slise = Slise(
        lat0=5, latf=30,
        lon0=-90, lonf=-10,
        month0=Month.APR, monthf=Month.JUL,
        year0=1997, yearf=2005,
    )  # PREDICTOR: Y

    chl_slise = Slise(
        lat0=36, latf=37,
        lon0=-5.3, lonf=-2,
        month0=Month.MAR, monthf=Month.APR,
        year0=1998, yearf=2006,
    )  # PRECITAND: Z
    selected_year = 1999

    s = spy.Spy4Caster(
            yargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=oisst_v2_mean_monthly_Jan1996_Dec2020, variable=SST, chunks=100),
            zargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=chl_1km_monthly_Sep1997_Dec2020, variable=CHL, chunks=100),
            plot_dir=PLOTS_DIR, mats_plot_name=MATS_PLOT_NAME, mca_plot_name=MCA_PLOT_NAME, cross_plot_name=CROSS_PLOT_NAME, zhat_plot_name=ZHAT_PLOT_NAME,
            force_name=True, plot_data_dir=PLOTS_DATA_DIR)
    # s.open_datasets()
    # s.slice_datasets(yslise=oisst_slise, zslise=chl_slise, yskip=0, zskip=0)
    # s.preprocess()
    s.load_preprocessed('./saved_2', 'save_ppcessed_', '.npy')
    # s.plot_preprocessed()
    # s.mca(nm=nm, alpha=alpha)
    s.load_mca('./saved_2', 'save_mca_', '.npy')
    # s.plot_mca(F.SHOW_PLOT | F.SAVE_FIG)
    # s.crossvalidation(nm=nm, alpha=alpha, multiprocessing=True)
    s.load_crossvalidation('./saved_2', 'save_cross_', '.npy')
    # s.plot_mca(F.SHOW_PLOT | F.SAVE_FIG)
    # s.plot_crossvalidation(F.SHOW_PLOT | F.SAVE_FIG)
    # s.plot_zhat(F.SHOW_PLOT | F.SAVE_FIG, sy=selected_year)
    s.run(F.SHOW_PLOT | F.SAVE_FIG | F.SAVE_DATA, sy=selected_year)


if __name__ == '__main__':
    spy.set_silence(False)
    main()