import spy4cast as spy
from spy4cast.stypes import F, Month, Slise, RDArgs


DATASET_DIR = '/datasets/'
PLOTS_DIR = ''
PLOTS_DATA_DIR = 'data/'
MCA_PLOT_NAME = 'mca_spy4cast_example.png'
CROSS_PLOT_NAME = 'cross_spy4cast_example.png'
HadISST_sst = 'HadISST_sst.nc'
slp_ERA20_1900_2010 = 'slp_ERA20_1900-2010.nc'
SST = 'sst'
MSL = 'msl'


def main():
    order=8
    period=5.5
    nm=3
    alpha=.1

    sst_slise = Slise(
        latitude_min=20,
        latitude_max=50,
        longitude_min=0,
        longitude_max=60,
        initial_month=Month.NOV,
        final_month=Month.FEB,
        initial_year=1901,
        final_year=2010,
    )  # PREDICTOR: Y

    slp_slise = Slise(
        latitude_min=20,
        latitude_max=45,
        longitude_min=-50,
        longitude_max=40,
        initial_month=Month.JAN,
        final_month=Month.APR,
        initial_year=1901,
        final_year=2010,
    )  # PRECITAND: Z

    # TODO: year0 and yearf dont have to be the same, there just needs to be same nuumber of times
    s = spy.Spy4Caster(
            yargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=HadISST_sst, variable=SST, chunks=20),
            zargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=slp_ERA20_1900_2010, variable=MSL, chunks=20),
            plot_dir=PLOTS_DIR, mca_plot_name=MCA_PLOT_NAME, cross_plot_name=CROSS_PLOT_NAME,
            force_name=True, plot_data_dir=PLOTS_DATA_DIR)
    s.load_datasets()
    s.slice_datasets(yslise=sst_slise, zslise=slp_slise)
    s.preprocess(order=order, period=period)
    s.mca(nm=nm, alpha=alpha)
    s.plot_mca(F.SHOW_PLOT | F.SAVE_FIG)
    s.crossvalidation(nm=nm, alpha=alpha, multiprocessing=False)
    s.plot_crossvalidation(F.SHOW_PLOT | F.SAVE_FIG)
    selected_year = 1990
    s.plot_zhat(F.SHOW_PLOT | F.SAVE_FIG, sy=selected_year)
    s.run(F.SHOW_PLOT | F.SAVE_FIG | F.SAVE_DATA, sy=selected_year)


if __name__ == '__main__':
    spy.Settings.silence = False
    main()