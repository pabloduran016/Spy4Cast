from spy4cast.meteo import Anom
from spy4cast import Month, Slise, set_silence, Dataset, F

# Enabel debug ouput
set_silence(False)

# Define constants ---------------------------------------------------------------------------------- #
DATASET_DIR = '/Users/Shared/datasets/'
PLOTS_DIR = 'plots'
PLOT_DATA_DIR = 'data-tna'

chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'
CHL = 'CHL'

chl_slise = Slise(36, 37, -5.3, -2, Month.MAR, Month.APR, 1998, 2020)

ds = Dataset(chl_1km_monthly_Sep1997_Dec2020, DATASET_DIR).open(CHL).slice(chl_slise)

map_anom = Anom(ds, 'map')

map_anom.save('map_anomaly', PLOT_DATA_DIR)
map_anom = Anom.load('map_anomaly', PLOT_DATA_DIR, type='map')
map_anom.plot(F.SHOW_PLOT | F.SAVE_FIG, year=1999, name='anom-map-example.png', cmap='jet', dir=PLOTS_DIR)

ts_anom = Anom(ds, 'ts')
ts_anom.save('ts_anomaly', PLOT_DATA_DIR)
ts_anom = Anom.load('ts_anomaly', PLOT_DATA_DIR, type='ts')
ts_anom.plot(F.SHOW_PLOT | F.SAVE_FIG, name='anom-ts-example.png', dir=PLOTS_DIR)
