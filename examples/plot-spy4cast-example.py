from spy4cast import spy4cast
import matplotlib.pyplot as plt

PLOTS_DIR = 'plots'
SAVED_DATA_DIR = 'saved_tna'

y_ppcessed = spy4cast.Preprocess.load('save_preprocessed_y_', dir=SAVED_DATA_DIR)
z_ppcessed = spy4cast.Preprocess.load('save_preprocessed_z_', dir=SAVED_DATA_DIR)
mca = spy4cast.MCA.load('save_mca_', SAVED_DATA_DIR, dsy=y_ppcessed, dsz=z_ppcessed)
cross = spy4cast.Crossvalidation.load('save_cross_', SAVED_DATA_DIR, dsy=y_ppcessed, dsz=z_ppcessed)

# FAST PLOTS
y_ppcessed.plot(save_fig=True, selected_year=2005, dir=PLOTS_DIR, name='sst-2005.png')
z_ppcessed.plot(cmap='viridis', selected_year=2006, dir=PLOTS_DIR, name='chl-2006.png')
mca.plot(save_fig=True, cmap='viridis', dir=PLOTS_DIR, name='mca-sst-chl.png')
cross.plot(save_fig=True, dir=PLOTS_DIR, name='crossvalidation-chl-sst.png')
cross.plot_zhat(2005, dir=PLOTS_DIR, name='zhat-2005.png')
cross.plot_zhat(2006, dir=PLOTS_DIR, name='zhat-2006.png')
cross.plot_zhat(2007, dir=PLOTS_DIR, name='zhat-2007.png')

# Show all the created plots
plt.show()
