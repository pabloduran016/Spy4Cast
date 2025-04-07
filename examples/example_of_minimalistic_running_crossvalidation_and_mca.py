from spy4cast.dataset import Dataset
from spy4cast import spy4cast, Month, Region
from matplotlib import pyplot as plt

DATASET_DIR = '/Users/Shared/datasets/'
predictand_dataset = 'chl_1km_monthly_Sep1997_Dec2020.nc'
predictor_dataset = 'oisst_v2_mean_monthly.nc'

y = Dataset(name=predictor_dataset, folder=DATASET_DIR).open(var='sst')
y.slice(Region(5, 25, -75, -20, Month.AUG, Month.SEP, 1997, 2019), skip=3)

z = Dataset(name=predictand_dataset, folder=DATASET_DIR).open(var="CHL")
z.slice(Region(36, 37, -5.3, -2, Month.MAR, Month.APR, 1998, 2020), skip=2)

y_ppcessed = spy4cast.Preprocess(y)
z_ppcessed = spy4cast.Preprocess(z)

SAVED_DATA_DIR = 'saved_tna'
y_ppcessed.save('save_preprocessed_y_', folder=SAVED_DATA_DIR)
z_ppcessed.save('save_preprocessed_z_', folder=SAVED_DATA_DIR)

nm = 3
alpha = 0.1
mca = spy4cast.MCA(y_ppcessed, z_ppcessed, nm, alpha)
mca.save('save_mca_', folder=SAVED_DATA_DIR)

cross = spy4cast.Crossvalidation(y_ppcessed, z_ppcessed, nm, alpha)
cross.save('save_cross_', folder=SAVED_DATA_DIR)
