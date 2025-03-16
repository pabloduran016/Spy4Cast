from spy4cast import Dataset, Region, Month
from spy4cast.spy4cast import Preprocess, MCA, Crossvalidation


DATASETS_FOLDER = '/Users/Shared/datasets'


PREDICTOR_NAME = "oisst_v2_mean_monthly.nc"
PREDICTOR_VAR = "sst"

PREDICTAND_NAME = "chl_1km_monthly_Sep1997_Dec2020.nc"
PREDICTAND_VAR = "CHL"

predictor = Dataset(PREDICTOR_NAME, DATASETS_FOLDER).open(PREDICTOR_VAR)
oisst_region = Region(
    lat0=5, latf=25,
    lon0=-75, lonf=-20,
    month0=Month.AUG, monthf=Month.SEP,
    year0=1997, yearf=2019,
)  # PREDICTOR: Y
predictor.slice(oisst_region, skip=3)

predictand = Dataset(PREDICTAND_NAME, DATASETS_FOLDER).open(PREDICTAND_VAR)
chl_region = Region(
    lat0=36, latf=37,
    lon0=-5.3, lonf=-2,
    month0=Month.MAR, monthf=Month.APR,
    year0=1998, yearf=2020,
)  # PRECITAND: Z
predictand.slice(chl_region, skip=0)

DATA_FOLDER = 'data-03122022'
PLOTS_FOLDER = 'plots-03122022'

PREDICTOR_PREPROCESSED_PREFIX = 'predictor_'
PREDICTAND_PREPROCESSED_PREFIX = 'predictand_'
MCA_PREFIX = 'mca_'
MCA_PLOT_NAME = 'mca.png'
CROSS_PREFIX = 'cross_'
CROSS_PLOT_NAME = 'cross.png'

LOAD_PREPROCESSED = False
LOAD_MCA = False
LOAD_CROSS = False

if LOAD_PREPROCESSED:
    predictor_preprocessed = Preprocess.load(PREDICTOR_PREPROCESSED_PREFIX, DATA_FOLDER)
    predictand_preprocessed = Preprocess.load(PREDICTAND_PREPROCESSED_PREFIX, DATA_FOLDER)
else:
    predictor_preprocessed = Preprocess(predictor)
    predictand_preprocessed = Preprocess(predictand)
    predictor_preprocessed.save(PREDICTOR_PREPROCESSED_PREFIX, DATA_FOLDER)
    predictand_preprocessed.save(PREDICTAND_PREPROCESSED_PREFIX, DATA_FOLDER)

nm = 3
alpha = .1

if LOAD_MCA:
    mca = MCA.load(MCA_PREFIX, DATA_FOLDER, dsy=predictor_preprocessed, dsz=predictand_preprocessed)
else:
    mca = MCA(predictor_preprocessed, predictand_preprocessed, nm, alpha)
    mca.save(MCA_PREFIX, DATA_FOLDER)

mca.plot(save_fig=True, cmap='viridis', name=MCA_PLOT_NAME, 
         figsize=(14, 8),
         width_ratios=[1, 1, 1],
         folder=PLOTS_FOLDER,)

if LOAD_CROSS:
    cross = Crossvalidation.load(CROSS_PREFIX, DATA_FOLDER, dsy=predictor_preprocessed, dsz=predictand_preprocessed)
else:
    cross = Crossvalidation(predictor_preprocessed, predictand_preprocessed, nm, alpha)
    cross.save(CROSS_PREFIX, DATA_FOLDER)

# cross.plot(save_fig=True, folder=PLOTS_FOLDER, name=CROSS_PLOT_NAME, version='default', mca=mca)
cross.plot(save_fig=True, folder=PLOTS_FOLDER, name=CROSS_PLOT_NAME, version=2, mca=mca)


import matplotlib.pyplot as plt

plt.show()
