import spy4cast as spy
from spy4cast.stypes import F, Month, Slise, RDArgs

DATASET_DIR = '/Users/Shared/datasets/'
PLOTS_DIR = 'plots'
PLOTS_DATA_DIR = 'saved_tna/'
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

    oisst_slise = Slise(
        lat0=5, latf=45,
        lon0=-90, lonf=-5,
        month0=Month.JUN, monthf=Month.JUL,
        year0=1997, yearf=2019,
    )  # PREDICTOR: Y

    chl_slise = Slise(
        lat0=36, latf=37,
        lon0=-5.3, lonf=-2,
        month0=Month.MAR, monthf=Month.APR,
        year0=1998, yearf=2020,
    )  # PRECITAND: Z
    selected_year = 2006

    s = spy.Spy4Caster(
            yargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=oisst_v2_mean_monthly, variable=SST, chunks=100),
            zargs=RDArgs(dataset_dir=DATASET_DIR, dataset_name=chl_1km_monthly_Sep1997_Dec2020, variable=CHL, chunks=100),
            plot_dir=PLOTS_DIR, mats_plot_name=MATS_PLOT_NAME, mca_plot_name=MCA_PLOT_NAME, cross_plot_name=CROSS_PLOT_NAME, zhat_plot_name=ZHAT_PLOT_NAME,
            plot_data_dir=PLOTS_DATA_DIR)

    # TODO: Implement `ray` for multiprocessing in crossvalidation
    load = True
    if not load:
        s.open_datasets()
        s.slice_datasets(yslise=oisst_slise, zslise=chl_slise, yskip=1, zskip=1)
        s.preprocess()  # Primero sin filtro y luego con filtro de 8 años
        s.mca(nm=nm, alpha=alpha)
        # s.crossvalidation(nm=nm, alpha=alpha, multiprocessing=False)
        s.run(F.SHOW_PLOT | F.SAVE_FIG | F.SAVE_DATA, sy=selected_year, cmap='viridis')
    else:
        s.load_preprocessed(PLOTS_DATA_DIR, 'save_preprocessed_', '.npy')
        s.load_mca(PLOTS_DATA_DIR, 'save_mca_', '.npy')
        # Cor, Pvalue, Cor_sig, reg, reg_sig = index_regression(s._y, s._mca_out.Us[1, :], alpha)
        # lats = s._ylat
        # lons = s._ylon
        # nlat = len(lats)
        # nlon = len(lons)
        #
        # arr = Cor.reshape((nlat, nlon))
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(projection=ccrs.PlateCarree())
        # im = ax.contourf(lons, lats, arr, cmap='bwr', extend='both', transform=ccrs.PlateCarree())
        # fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.02)
        # ax.coastlines()
        # plt.show()
        s.load_crossvalidation(PLOTS_DATA_DIR, 'save_cross_', '.npy')
        s.run(F.SHOW_PLOT | F.SAVE_FIG, sy=selected_year, cmap='viridis', yslise=oisst_slise, zslise=chl_slise)
    # s.plot_preprocessed()
    # s.plot_mca(F.SHOW_PLOT | F.SAVE_FIG)
    # s.plot_mca(F.SHOW_PLOT | F.SAVE_FIG)
    # s.plot_crossvalidation(F.SHOW_PLOT | F.SAVE_FIG)
    # s.plot_zhat(F.SHOW_PLOT | F.SAVE_FIG, sy=selected_year)


if __name__ == '__main__':
    spy.set_silence(False)
    main()