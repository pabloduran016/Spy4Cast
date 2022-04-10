from src.spy4cast.meteo import Clim
from src.spy4cast import Month, Slise, set_silence, Dataset, F

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

map_clim = Clim(ds, 'map')
map_clim.save('map_climatology', PLOT_DATA_DIR)
map_clim = Clim.load('map_climatology', PLOT_DATA_DIR, type='map')

map_clim.plot(F.SHOW_PLOT | F.SAVE_FIG, name='clim-map-example.png', cmap='jet')

ts_clim = Clim(ds, 'ts')
ts_clim.save('ts_climatology', PLOT_DATA_DIR)
ts_clim = Clim.load('ts_climatology', PLOT_DATA_DIR, type='ts')

ts_clim.plot(F.SHOW_PLOT | F.SAVE_FIG, name='clim-ts-example.png')
