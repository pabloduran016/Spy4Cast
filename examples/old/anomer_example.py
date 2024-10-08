from spy4cast import Region, Month, F, AnomerMap


DATASETS_FOLDER = '/Users/Shared/datasets/'
HadISST_sst = 'HadISST_sst.nc'



sl = Region(
    lat0=-45, latf=45,
    lon0=-100, lonf=100,
    month0=Month.JAN, monthf=Month.MAR,
    year0=1871, yearf=2020,
    sy=1990,
)
a = AnomerMap(dataset_folder=DATASETS_FOLDER, dataset_name=HadISST_sst)
a.open_dataset()
a.slice_dataset(sl)
a.apply()
a.run(show_plot=True, save_fig=True, region=sl)
