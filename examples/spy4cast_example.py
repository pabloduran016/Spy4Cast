import spy4cast as spy
from spy4cast.stypes import F, Month, Slise, RDArgs


DATASET_DIR = '/datasets/'
PLOTS_DIR = ''
PLOTS_DATA_DIR = 'data/'
PLOT_NAME = 'spy4cast_example.png'
HadISST_sst = 'HadISST_sst.nc'
slp_ERA20_1900_2010 = 'slp_ERA20_1900-2010.nc'
SST = 'sst'
MSL = 'msl'


def main():
    year0 = 1901
    yearf = 2010

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
        initial_year=year0,
        final_year=yearf,
    )  # PREDICTOR: Y

    slp_slise = Slise(
        latitude_min=20,
        latitude_max=45,
        longitude_min=-50,
        longitude_max=40,
        initial_month=Month.NOV,
        final_month=Month.FEB,
        initial_year=year0,
        final_year=yearf,
    )  # PRECITAND: Z

    s = spy.Spy4Caster(
            yargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=HadISST_sst, variable=SST),
            zargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=slp_ERA20_1900_2010, variable=MSL),
            plot_dir=PLOTS_DIR, plot_name=PLOT_NAME, force_name=True, plot_data_dir=PLOTS_DATA_DIR).load_datasets()
    s.slice_datasets(yslise=sst_slise, zslise=slp_slise)
    s.apply(order=order, period=period, nm=nm, alpha=alpha)
    s.run(F.SHOW_PLOT | F.SAVE_FIG)


if __name__ == '__main__':
    main()