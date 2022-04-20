from spy4cast.dataset import Dataset
from spy4cast import spy4cast, Month, Slise, set_silence, F

set_silence(False)


DATASET_DIR = '/Users/Shared/datasets/'
PLOTS_DIR = 'plots'
PLOT_DATA_DIR = 'data-tna'

MCA_PLOT_NAME = 'mca_spy4cast_chlorMED_tna.png'
CROSS_PLOT_NAME = 'cross_spy4cast_chlorMED_tna.png'
ZHAT_PLOT_NAME = 'zhat_spy4cast_chlorMED_tna.png'

chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'
oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'  # Clorofila perdictando con años bien
CHL = 'CHL'

oisst_v2_mean_monthly_Jan1996_Dec2020 = 'oisst_v2_mean_monthly_Jan1996_Dec2020.nc'  # SST predictor
SST = 'sst'

nm = 3
alpha = .1

oisst_slise = Slise(5, 45, -90, -5, Month.JUN, Month.JUL, 1997, 2019)
sst = Dataset(
    name=oisst_v2_mean_monthly, dir=DATASET_DIR
).open(var=SST).slice(oisst_slise)


chl_slise = Slise(36, 37, -5.3, -2, Month.MAR, Month.APR, 1998, 2020)
chl = Dataset(
    name=chl_1km_monthly_Sep1997_Dec2020, dir=DATASET_DIR
).open(var=CHL).slice(chl_slise)


sst_ppcessed = spy4cast.Preprocess(sst)  # Optional keyword arguments for order and period to apply filter
chl_ppcessed = spy4cast.Preprocess(chl)

sst_ppcessed.plot(F.SHOW_PLOT)
chl_ppcessed.plot(F.SHOW_PLOT, cmap='viridis')

sst_ppcessed.save('save_preprocessed_y_', dir=PLOT_DATA_DIR)
chl_ppcessed.save('save_preprocessed_z_', dir=PLOT_DATA_DIR)

sst_ppcessed_loaded = spy4cast.Preprocess.load('save_preprocessed_y_', dir=PLOT_DATA_DIR)
chl_ppcessed_loaded = spy4cast.Preprocess.load('save_preprocessed_z_', dir=PLOT_DATA_DIR)

sst_ppcessed_loaded.plot(F.SHOW_PLOT)
chl_ppcessed_loaded.plot(F.SHOW_PLOT, cmap='viridis')

assert (sst_ppcessed_loaded.data == sst_ppcessed.data).all()
assert (chl_ppcessed_loaded.data == chl_ppcessed.data).all()
assert (sst_ppcessed_loaded.time == sst_ppcessed.time).all()
assert (chl_ppcessed_loaded.time == chl_ppcessed.time).all()
assert (sst_ppcessed_loaded.lat == sst_ppcessed.lat).all()
assert (chl_ppcessed_loaded.lat == chl_ppcessed.lat).all()
assert (sst_ppcessed_loaded.lon == sst_ppcessed.lon).all()
assert (chl_ppcessed_loaded.lon == chl_ppcessed.lon).all()
assert (chl_ppcessed_loaded.slise.as_numpy() == chl_ppcessed.slise.as_numpy()).all()
