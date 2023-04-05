import spy4cast as spy
from spy4cast.stypes import F, Month, Slise, RDArgs


DATASET_DIR = '/datasets/'
PLOTS_DIR = ''
PLOTS_DATA_DIR = 'data/'
MCA_PLOT_NAME = 'mca_spy4cast_example.png'
CROSS_PLOT_NAME = 'cross_spy4cast_example.png'
ZHAT_PLOT_NAME = 'zhat_spy4cast_example.png'
HadISST_sst = 'HadISST_sst.nc'
slp_ERA20_1900_2010 = 'slp_ERA20_1900-2010.nc'
SST = 'sst'
MSL = 'msl'


def main() -> None:
    order = 8
    period = 5.5

    sst_slise = Slise(
        lat0=20, latf=50,
        lon0=0, lonf=60,
        month0=Month.JAN, monthf=Month.APR,
        year0=1980, yearf=2010,
    )  # PREDICTOR: Y

    slp_slise = Slise(
        lat0=20, latf=45,
        lon0=-50, lonf=40,
        month0=Month.JAN, monthf=Month.APR,
        year0=1980, yearf=2010,
    )  # PRECITAND: Z

    s = spy.Spy4Caster(
            yargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=HadISST_sst, variable=SST, chunks=100),
            zargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=slp_ERA20_1900_2010, variable=MSL, chunks=100),
            plot_dir=PLOTS_DIR, mca_plot_name=MCA_PLOT_NAME, cross_plot_name=CROSS_PLOT_NAME, zhat_plot_name=ZHAT_PLOT_NAME,
            plot_data_dir=PLOTS_DATA_DIR)
    s.open_datasets()
    s.slice_datasets(yslise=sst_slise, zslise=slp_slise)
    s.preprocess(order=order, period=period)
    # s.mca(nm=3, alpha=.1)
    # s.plot_mca(show_plot=True, save_fig=True)
    # s.crossvalidation(nm=3, alpha=.1, multiprocessing=True)
    # s.load_ppcessed('./saved', 'save_ppcessed_', '.npy')
    s.load_mca('./saved', 'save_mca_', '.npy')
    s.load_crossvalidation('./saved', 'save_cross_', '.npy')
    # s.plot_mca(show_plot=True, save_fig=True)
    # s.plot_crossvalidation(show_plot=True, save_fig=True)
    selected_year = 1986
    # s.plot_zhat(show_plot=True, save_fig=True, sy=selected_year)
    s.run(show_plot=True, save_fig=True, sy=selected_year)


if __name__ == '__main__':
    main()