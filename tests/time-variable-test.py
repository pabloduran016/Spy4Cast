from spy4cast import Dataset, Slise, month


DATASETS_DIR = '/Users/Shared/datasets'
DATASET_NAME = 'ts_ctl_hadgem.nc'

a = Dataset(DATASET_NAME, DATASETS_DIR).open()
slise = Slise(-90, 90, -180, 180, Month.JAN, Month.DEC, 2100, 2119)


