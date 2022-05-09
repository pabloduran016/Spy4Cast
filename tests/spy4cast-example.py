from spy4cast import F
from spy4cast.dataset import Dataset
from spy4cast import spy4cast, Month, Slise, set_silence

# Enabel debug ouput
set_silence(False)

# Define constants ---------------------------------------------------------------------------------- #
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

oisst_slise = Slise(5, 45, -90, -5, Month.JUN, Month.JUL, 1997, 2019)
chl_slise = Slise(36, 37, -5.3, -2, Month.MAR, Month.APR, 1998, 2020)

# Indicates if get the already conputed data -------------------------------------------------------- #
LOAD_PREPROCESSED = True
LOAD_MCA = False
LOAD_CROSSVALIDATION = True

# Settings ------------------------------------------------------------------------------------------ #
nm = 3
alpha = .1

# Opening and preprocessing ------------------------------------------------------------------------ #
if LOAD_PREPROCESSED:
    # Load the precomputed data
    sst_ppcessed = spy4cast.Preprocess.load('save_preprocessed_y_', dir=PLOT_DATA_DIR)
    chl_ppcessed = spy4cast.Preprocess.load('save_preprocessed_z_', dir=PLOT_DATA_DIR)
else:
    # Create two datasets for predictor and predictand
    sst = Dataset(
        name=oisst_v2_mean_monthly, dir=DATASET_DIR
    ).open(var=SST).slice(oisst_slise, skip=1)

    chl = Dataset(
        name=chl_1km_monthly_Sep1997_Dec2020, dir=DATASET_DIR
    ).open(var=CHL).slice(chl_slise, skip=3)
    
    # Apply the preprocessing methodology
    sst_ppcessed = spy4cast.Preprocess(sst)  # Optional keyword arguments for order and period to apply filter
    chl_ppcessed = spy4cast.Preprocess(chl)

    # Optional saving to load later
    sst_ppcessed.save('save_preprocessed_y_', dir=PLOT_DATA_DIR)
    chl_ppcessed.save('save_preprocessed_z_', dir=PLOT_DATA_DIR)

# Plot the matrices. Use `F.SHOW_PLOT` if you want to show it right away.
# This will only plot and later we will call `plt.show()` and show all the plots at once
sst_ppcessed.plot(F.SAVE_FIG, name='sst-preprocessed.png', dir=PLOTS_DIR)
chl_ppcessed.plot(F.SAVE_FIG, name='chl-preprocessed.png', dir=PLOTS_DIR, cmap='viridis')

if LOAD_MCA:
    # Load the precomputed methodology. Notice we need to pass in as arguments the predictar and predictor preprocessed
    mca = spy4cast.MCA.load('save_mca_', PLOT_DATA_DIR, dsy=sst_ppcessed, dsz=chl_ppcessed)
else:
    # Apply MCA procedure and save the ouput for future loading
    mca = spy4cast.MCA(chl_ppcessed, sst_ppcessed, nm, alpha)
    mca.save('save_mca_', dir=PLOT_DATA_DIR)

# Here we plot the MCA but we wait t o show it until the end of the file
mca.plot(F.SAVE_FIG, name='mca.png', dir=PLOTS_DIR, cmap='viridis', signs=(True, True, False))

if LOAD_CROSSVALIDATION:
    # Load the precomputed crossvalidation
    cross = spy4cast.Crossvalidation.load('save_cross_', PLOT_DATA_DIR, dsy=sst_ppcessed, dsz=chl_ppcessed)
else:
    # Apply the corssvalidation and save the output
    cross = spy4cast.Crossvalidation(sst_ppcessed, chl_ppcessed, nm, alpha)
    cross.save('save_cross_', dir=PLOT_DATA_DIR)

# Plot crossvalidation and zhat but not showing it yet
cross.plot(F.SAVE_FIG, name='crossvalidation.png', dir=PLOTS_DIR, map_ticks=[-.1, 0, 1])
cross.plot_zhat(2006, F.SAVE_FIG, name='zhat-2006.png', dir=PLOTS_DIR, cmap='viridis')

# Show all the created plots
from matplotlib import pyplot as plt
plt.show()
