from spy4cast import spy4cast, Month, Region, Dataset
from spy4cast.land_array import LandArray
from matplotlib import pyplot as plt

DATASET_DIR = '/Users/Shared/datasets/'
predictand_dataset = 'chl_1km_monthly_Sep1997_Dec2020.nc'
predictor_dataset = 'oisst_v2_mean_monthly.nc'

y = Dataset(name=predictor_dataset, folder=DATASET_DIR).open(var='sst')
y.slice(Region(5, 25, -75, -20, Month.AUG, Month.SEP, 1997, 2019), skip=3)

z = Dataset(name=predictand_dataset, folder=DATASET_DIR).open(var="CHL")
z.slice(Region(36, 37, -5.3, -2, Month.MAR, Month.APR, 1998, 2020), skip=2)

# To use a custom preprocessing step override Preprocess __init__
# Need to set variable ._ds, ._land_data, ._time, ._lat, ._lon
# RECOMMEND: copy implementation in file src/spy4cast/spy4cast/preprocess.py 
# and change what is needed
class CustomPreprocess(spy4cast.Preprocess):
    def __init__(self, ds: Dataset) -> None:
        self._ds = ds
        
        a = ds.data.groupby('year').mean()
        anomaly = a - a.mean('year')
        anomaly = anomaly.transpose(
            'year', ds._lat_key,  ds._lon_key
        )

        nt, nlat, nlon = anomaly.shape
        data = anomaly.values.reshape((nt, nlat * nlon)).transpose()  # space x time

        # Takes into account NaN and makes it easier to perform operations that need non-NaN arrays
        self._land_data = LandArray(data)

        self._time = anomaly['year']
        self._lat = anomaly[ds._lat_key]
        self._lon = anomaly[ds._lon_key]


y_ppcessed = CustomPreprocess(y)
z_ppcessed = CustomPreprocess(z)


nm = 3
alpha = 0.1
mca = spy4cast.MCA(y_ppcessed, z_ppcessed, nm, alpha)

mca.plot(width_ratios=[1, 2, 2], cmap="viridis")
plt.show()
