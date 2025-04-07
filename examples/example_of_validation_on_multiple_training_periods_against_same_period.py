from typing import Tuple
from spy4cast import Dataset, Region, Month
from spy4cast.spy4cast import Preprocess, MCA, Validation


DATASETS_DIR = '/Users/Shared/datasets'


PREDICTOR_NAME = "HadISST_sst.nc"
PREDICTOR_VAR = "sst"

PREDICTAND_NAME = "slp_ERA20_1900-2010.nc"
PREDICTAND_VAR = "msl"


DATA_DIR = 'data-01042024'
PLOTS_DIR = 'plots-01042024'

TRAINING_PREDICTOR_PREPROCESSED_PREFIX = 'training_predictor_'
TRAINING_PREDICTAND_PREPROCESSED_PREFIX = 'training_predictand_'
VALIDATING_PREDICTOR_PREPROCESSED_PREFIX = 'validating_predictor_'
VALIDATING_PREDICTAND_PREPROCESSED_PREFIX = 'validating_predictand_'
MCA_PREFIX = 'mca_'
MCA_PLOT_NAME = 'mca_'
VALIDATION_PREFIX = 'validation_'
VALIDATION_PLOT_NAME = 'validation_'

LOAD_PREPROCESSED = False
LOAD_MCA = False
LOAD_VALIDATION = False



def get_training_preprocessed(year0: int, yearf: int) -> Tuple[Preprocess, Preprocess]:
    training_predictor = Dataset(PREDICTOR_NAME, DATASETS_DIR).open(PREDICTOR_VAR)  # JAN-1870 : MAY-2020
    training_y_region = Region(
        lat0=-30, latf=30, 
        lon0=-230, lonf=-40, 
        month0=Month.SEP, monthf=Month.OCT, 
        year0=year0, yearf=yearf
    )  # PREDICTOR: Y
    training_predictor.slice(training_y_region, skip=2)

    training_predictand = Dataset(PREDICTAND_NAME, DATASETS_DIR).open(PREDICTAND_VAR)  # JAN-1959 : DEC-2004
    training_z_region = Region(
        lat0=10, latf=70,
        lon0=-100, lonf=40,
        month0=Month.NOV, monthf=Month.DEC,
        year0=year0, yearf=yearf,
    )  # PRECITAND: Z
    training_predictand.slice(training_z_region, skip=2)

    if LOAD_PREPROCESSED:
        training_predictor_preprocessed = Preprocess.load(TRAINING_PREDICTOR_PREPROCESSED_PREFIX + f'{year0}-{yearf}_', DATA_DIR)
        training_predictand_preprocessed = Preprocess.load(TRAINING_PREDICTAND_PREPROCESSED_PREFIX + f'{year0}-{yearf}_', DATA_DIR)
    else:
        training_predictor_preprocessed = Preprocess(training_predictor)
        training_predictand_preprocessed = Preprocess(training_predictand)

        training_predictor_preprocessed.save(TRAINING_PREDICTOR_PREPROCESSED_PREFIX + f'{year0}-{yearf}_', DATA_DIR)
        training_predictand_preprocessed.save(TRAINING_PREDICTAND_PREPROCESSED_PREFIX + f'{year0}-{yearf}_', DATA_DIR)

    return training_predictor_preprocessed, training_predictand_preprocessed


validating_predictor = Dataset(PREDICTOR_NAME, DATASETS_DIR).open(PREDICTOR_VAR)  # JAN-1870 : MAY-2020
validating_y_region = Region(
    lat0=-30, latf=30, 
    lon0=-230, lonf=-40, 
    month0=Month.SEP, monthf=Month.OCT, 
    year0=1980, yearf=2010
)  # PREDICTOR: Y
validating_predictor.slice(validating_y_region, skip=2)

validating_predictand = Dataset(PREDICTAND_NAME, DATASETS_DIR).open(PREDICTAND_VAR)  # JAN-1959 : DEC-2004
validating_z_region = Region(
    lat0=10, latf=70,
    lon0=-100, lonf=40,
    month0=Month.NOV, monthf=Month.DEC,
    year0=1980, yearf=2010,
)  # PRECITAND: Z
validating_predictand.slice(validating_z_region, skip=2)


if LOAD_PREPROCESSED:
    validating_predictor_preprocessed = Preprocess.load(VALIDATING_PREDICTOR_PREPROCESSED_PREFIX, DATA_DIR)
    validating_predictand_preprocessed = Preprocess.load(VALIDATING_PREDICTAND_PREPROCESSED_PREFIX, DATA_DIR)
else:
    validating_predictor_preprocessed = Preprocess(validating_predictor)
    validating_predictand_preprocessed = Preprocess(validating_predictand)

    validating_predictor_preprocessed.save(VALIDATING_PREDICTOR_PREPROCESSED_PREFIX, DATA_DIR)
    validating_predictand_preprocessed.save(VALIDATING_PREDICTAND_PREPROCESSED_PREFIX, DATA_DIR)

nm = 3
alpha = .1


cases = [
    (1950, 1980),
    (1940, 1970),
    (1930, 1960),
    (1920, 1950),
]


import matplotlib.pyplot as plt


for year0, yearf in cases:
    print(f'case {year0=}, {yearf=}')

    t_y, t_z = get_training_preprocessed(year0, yearf)

    if LOAD_MCA:
        training_mca = MCA.load(MCA_PREFIX + f'{year0}-{yearf}_', DATA_DIR, dsy=t_y, dsz=t_z)
    else:
        training_mca = MCA(t_y, t_z, nm, alpha)
        training_mca.save(MCA_PREFIX + f'{year0}-{yearf}_', DATA_DIR)

    training_mca.plot(save_fig=True, cmap='viridis', name=MCA_PLOT_NAME + f'{year0}-{yearf}.png', folder=PLOTS_DIR)

    if LOAD_VALIDATION:
        validation = Validation.load(VALIDATION_PREFIX + f'{year0}-{yearf}_', DATA_DIR, validating_dsy=validating_predictor_preprocessed, validating_dsz=validating_predictand_preprocessed, training_mca=training_mca)
    else:
        validation = Validation(training_mca, validating_predictor_preprocessed, validating_predictand_preprocessed)
        validation.save(VALIDATION_PREFIX + f'{year0}-{yearf}_', DATA_DIR)

    validation.plot(save_fig=True, folder=PLOTS_DIR, name=VALIDATION_PLOT_NAME + f'{year0}-{yearf}.png', version='default')
    validation.plot_zhat(2001, save_fig=True, folder=PLOTS_DIR, name=VALIDATION_PLOT_NAME + f'zhat_{year0}-{yearf}.png')

    plt.close('all')


# 
# plt.show()

