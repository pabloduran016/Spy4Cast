import spy4cast as spy
from spy4cast.stypes import Slise, Month, F


DATASETS_DIR = '/datasets/'
HadISST_sst = 'HadISST_sst.nc'


def main():
    sl = Slise(
        latitude_min=-45,
        latitude_max=45,
        longitude_min=-100,
        longitude_max=100,
        initial_month=Month.JAN,
        final_month=Month.MAR,
        initial_year=1871,
        final_year=2020,
        selected_year=1990,
    )
    spy.AnomerMap(dataset_dir=DATASETS_DIR, dataset_name=HadISST_sst) \
        .load_dataset() \
        .slice_dataset(sl) \
        .apply() \
        .run(F.SHOW_PLOT | F.SAVE_FIG, slise=sl)


if __name__ == '__main__':
    main()
