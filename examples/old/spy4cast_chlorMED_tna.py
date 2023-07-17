import spy4cast as spy
from spy4cast.stypes import F, Month, Region, RDArgs

DATASET_FOLDER = '/Users/Shared/datasets/'
PLOTS_FOLDER = 'plots'
PLOTS_DATA_FOLDER = 'saved_tna/'
MATS_PLOT_NAME = 'mats_spy4cast_chlorMED_tna.png'
MCA_PLOT_NAME = 'mca_spy4cast_chlorMED_tna.png'
CROSS_PLOT_NAME = 'cross_spy4cast_chlorMED_tna.png'
ZHAT_PLOT_NAME = 'zhat_spy4cast_chlorMED_tna.png'
chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'
# Clorofila perdictando

oisst_v2_mean_monthly = 'oisst_v2_mean_monthly.nc'  # Clorofila perdictando con años bien
CHL = 'CHL'
oisst_v2_mean_monthly_Jan1996_Dec2020 = 'oisst_v2_mean_monthly_Jan1996_Dec2020.nc'  # SST predictor
SST = 'sst'

def main() -> None:
    nm=3
    alpha=.1

    oisst_region = Region(
        lat0=5, latf=45,
        lon0=-90, lonf=-5,
        month0=Month.JUN, monthf=Month.JUL,
        year0=1997, yearf=2019,
    )  # PREDICTOR: Y

    chl_region = Region(
        lat0=36, latf=37,
        lon0=-5.3, lonf=-2,
        month0=Month.MAR, monthf=Month.APR,
        year0=1998, yearf=2020,
    )  # PRECITAND: Z
    selected_year = 2006

    s = spy.Spy4Caster(
            yargs=RDArgs(dataset_folder=DATASET_FOLDER, dataset_name=oisst_v2_mean_monthly, variable=SST, chunks=100),
            zargs=RDArgs(dataset_folder=DATASET_FOLDER, dataset_name=chl_1km_monthly_Sep1997_Dec2020, variable=CHL, chunks=100),
            plot_folder=PLOTS_FOLDER, mats_plot_name=MATS_PLOT_NAME, mca_plot_name=MCA_PLOT_NAME, cross_plot_name=CROSS_PLOT_NAME, zhat_plot_name=ZHAT_PLOT_NAME,
            plot_data_folder=PLOTS_DATA_FOLDER)

    # TODO: Implement `ray` for multiprocessing in crossvalidation
    load = False
    if not load:
        s.open_datasets()
        s.slice_datasets(yregion=oisst_region, zregion=chl_region, yskip=0, zskip=3)
        s.load_preprocessed(PLOTS_DATA_FOLDER, 'save_preprocessed_', '.npy')  # s.preprocess()  # Primero sin filtro y luego con filtro de 8 años
        s.mca(nm=nm, alpha=alpha)
        s.crossvalidation(nm=nm, alpha=alpha, multiprocessing=False)
        s.run(show_plot=True, save_fig=True, sy=selected_year, cmap='viridis', yregion=oisst_region, zregion=chl_region)
    else:
        s.load_preprocessed(PLOTS_DATA_FOLDER, 'save_preprocessed_', '.npy')
        s.load_mca(PLOTS_DATA_FOLDER, 'save_mca_', '.npy')
        s.load_crossvalidation(PLOTS_DATA_FOLDER, 'save_cross_', '.npy')
        s.run(show_plot=True, save_fig=True, sy=selected_year, cmap='viridis', yregion=oisst_region, zregion=chl_region)

if __name__ == '__main__':
    main()