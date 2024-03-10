from spy4cast import Dataset, Region, Month
from spy4cast.spy4cast import Preprocess, MCA, Crossvalidation


DATASETS_FOLDER = '/Users/Shared/datasets'


PREDICTOR_NAME = "oisst_v2_mean_monthly.nc"
PREDICTOR_VAR = "sst"

PREDICTAND_NAME = "chl_1km_monthly_Sep1997_Dec2020.nc"
PREDICTAND_VAR = "CHL"

predictor = Dataset(PREDICTOR_NAME, DATASETS_FOLDER).open(PREDICTOR_VAR)  # JAN-1870 : MAY-2020
oisst_region = Region(
    lat0=5, latf=45,
    lon0=-90, lonf=-5,
    month0=Month.JUN, monthf=Month.JUL,
    year0=1997, yearf=2019,
)  # PREDICTOR: Y
predictor.slice(oisst_region, skip=3)

predictand = Dataset(PREDICTAND_NAME, DATASETS_FOLDER).open(PREDICTAND_VAR)  # JAN-1959 : DEC-2004
chl_region = Region(
    lat0=36, latf=37,
    lon0=-5.3, lonf=-2,
    month0=Month.MAR, monthf=Month.APR,
    year0=1998, yearf=2020,
)  # PRECITAND: Z
predictand.slice(chl_region, skip=3)

DATA_FOLDER = 'data-03122022'
PLOTS_FOLDER = 'plots-03122022'

PREDICTOR_PREPROCESSED_PREFIX = 'predictor_'
PREDICTAND_PREPROCESSED_PREFIX = 'predictand_'
MCA_PREFIX = 'mca_'
MCA_PLOT_NAME = 'mca.png'
CROSS_PREFIX = 'cross_'
CROSS_PLOT_NAME = 'cross.png'

LOAD_PREPROCESSED = True
LOAD_MCA = False
LOAD_CROSS = True

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

mca.plot(save_fig=True, cmap='viridis', name=MCA_PLOT_NAME, folder=PLOTS_FOLDER, ruy_ticks=[-0.25, -0.125, 0, 0.125, 0.25],
         ruz_ticks=[-0.15, -0.075, 0, 0.075, 0.15])

if LOAD_CROSS:
    cross = Crossvalidation.load(CROSS_PREFIX, DATA_FOLDER, dsy=predictor_preprocessed, dsz=predictand_preprocessed)
else:
    cross = Crossvalidation(predictor_preprocessed, predictand_preprocessed, nm, alpha)
    cross.save(CROSS_PREFIX, DATA_FOLDER)

# cross.plot(save_fig=True, folder=PLOTS_FOLDER, name=CROSS_PLOT_NAME, version='default', mca=mca)
cross.plot(save_fig=True, folder=PLOTS_FOLDER, name=CROSS_PLOT_NAME, version='elena', mca=mca)


import matplotlib.pyplot as plt

plt.show()
