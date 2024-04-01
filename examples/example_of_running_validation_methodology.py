from spy4cast import Dataset, Region, Month
from spy4cast.spy4cast import Preprocess, MCA, Validation


DATASETS_DIR = '/Users/Shared/datasets'


PREDICTOR_NAME = "oisst_v2_mean_monthly.nc"
PREDICTOR_VAR = "sst"

PREDICTAND_NAME = "chl_1km_monthly_Sep1997_Dec2020.nc"
PREDICTAND_VAR = "CHL"

training_predictor = Dataset(PREDICTOR_NAME, DATASETS_DIR).open(PREDICTOR_VAR)  # JAN-1870 : MAY-2020
training_oisst_region = Region(
    lat0=5, latf=45,
    lon0=-90, lonf=-5,
    month0=Month.JUN, monthf=Month.JUL,
    year0=1997, yearf=2007,
)  # PREDICTOR: Y
training_predictor.slice(training_oisst_region, skip=3)

training_predictand = Dataset(PREDICTAND_NAME, DATASETS_DIR).open(PREDICTAND_VAR)  # JAN-1959 : DEC-2004
training_chl_region = Region(
    lat0=36, latf=37,
    lon0=-5.3, lonf=-2,
    month0=Month.MAR, monthf=Month.APR,
    year0=1998, yearf=2008,
)  # PRECITAND: Z
training_predictand.slice(training_chl_region, skip=3)

validating_predictor = Dataset(PREDICTOR_NAME, DATASETS_DIR).open(PREDICTOR_VAR)  # JAN-1870 : MAY-2020
validating_oisst_region = Region(
    lat0=5, latf=45,
    lon0=-90, lonf=-5,
    month0=Month.JUN, monthf=Month.JUL,
    year0=2008, yearf=2018,
)  # PREDICTOR: Y
validating_predictor.slice(validating_oisst_region, skip=3)

validating_predictand = Dataset(PREDICTAND_NAME, DATASETS_DIR).open(PREDICTAND_VAR)  # JAN-1959 : DEC-2004
validating_chl_region = Region(
    lat0=36, latf=37,
    lon0=-5.3, lonf=-2,
    month0=Month.MAR, monthf=Month.APR,
    year0=2009, yearf=2019,
)  # PRECITAND: Z
validating_predictand.slice(validating_chl_region, skip=3)

DATA_DIR = 'data-03042023'
PLOTS_DIR = 'plots-03042023'

TRAINING_PREDICTOR_PREPROCESSED_PREFIX = 'training_predictor_'
TRAINING_PREDICTAND_PREPROCESSED_PREFIX = 'training_predictand_'
VALIDATING_PREDICTOR_PREPROCESSED_PREFIX = 'validating_predictor_'
VALIDATING_PREDICTAND_PREPROCESSED_PREFIX = 'validating_predictand_'
MCA_PREFIX = 'mca_'
MCA_PLOT_NAME = 'mca.png'
VALIDATION_PREFIX = 'validation_'
VALIDATION_PLOT_NAME = 'validation.png'

LOAD_PREPROCESSED = False
LOAD_MCA = False
LOAD_VALIDATION = False

if LOAD_PREPROCESSED:
    training_predictor_preprocessed = Preprocess.load(TRAINING_PREDICTOR_PREPROCESSED_PREFIX, DATA_DIR)
    training_predictand_preprocessed = Preprocess.load(TRAINING_PREDICTAND_PREPROCESSED_PREFIX, DATA_DIR)
    validating_predictor_preprocessed = Preprocess.load(VALIDATING_PREDICTOR_PREPROCESSED_PREFIX, DATA_DIR)
    validating_predictand_preprocessed = Preprocess.load(VALIDATING_PREDICTAND_PREPROCESSED_PREFIX, DATA_DIR)
else:
    training_predictor_preprocessed = Preprocess(training_predictor)
    training_predictand_preprocessed = Preprocess(training_predictand)
    validating_predictor_preprocessed = Preprocess(validating_predictor)
    validating_predictand_preprocessed = Preprocess(validating_predictand)

    training_predictor_preprocessed.save(TRAINING_PREDICTOR_PREPROCESSED_PREFIX, DATA_DIR)
    training_predictand_preprocessed.save(TRAINING_PREDICTAND_PREPROCESSED_PREFIX, DATA_DIR)
    validating_predictor_preprocessed.save(VALIDATING_PREDICTOR_PREPROCESSED_PREFIX, DATA_DIR)
    validating_predictand_preprocessed.save(VALIDATING_PREDICTAND_PREPROCESSED_PREFIX, DATA_DIR)

nm = 3
alpha = .1

if LOAD_MCA:
    training_mca = MCA.load(MCA_PREFIX, DATA_DIR, dsy=training_predictor_preprocessed, dsz=training_predictand_preprocessed)
else:
    training_mca = MCA(training_predictor_preprocessed, training_predictand_preprocessed, nm, alpha)
    training_mca.save(MCA_PREFIX, DATA_DIR)

training_mca.plot(save_fig=True, cmap='viridis', name=MCA_PLOT_NAME, folder=PLOTS_DIR)

if LOAD_VALIDATION:
    validation = Validation.load(VALIDATION_PREFIX, DATA_DIR, validating_dsy=validating_predictor_preprocessed, validating_dsz=validating_predictand_preprocessed, training_mca=training_mca)
else:
    validation = Validation(training_mca, validating_predictor_preprocessed, validating_predictand_preprocessed)
    validation.save(VALIDATION_PREFIX, DATA_DIR)

validation.plot(save_fig=True, folder=PLOTS_DIR, name=VALIDATION_PLOT_NAME, version='default')
validation.plot_zhat(2015, save_fig=True, folder=PLOTS_DIR, name="zhat_" + VALIDATION_PLOT_NAME)


import matplotlib.pyplot as plt

plt.show()
