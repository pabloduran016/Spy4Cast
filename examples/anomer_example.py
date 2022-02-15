import spy4cast as spy
from spy4cast.stypes import Slise, Month, F


DATASETS_DIR = '/datasets/'
HadISST_sst = 'HadISST_sst.nc'


def main():
    sl = Slise(
        lat0=-45, latf=45,
        lon0=-100, lonf=100,
        month0=Month.JAN, monthf=Month.MAR,
        year0=1871, yearf=2020,
        sy=1990,
    )
    spy.AnomerMap(dataset_dir=DATASETS_DIR, dataset_name=HadISST_sst) \
        .open_dataset() \
        .slice_dataset(sl) \
        .apply() \
        .run(F.SHOW_PLOT | F.SAVE_FIG, slise=sl)


if __name__ == '__main__':
    main()
