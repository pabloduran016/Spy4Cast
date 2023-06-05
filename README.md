# Spy4Cast
![Icon](docs/source/_static/favicon.png)

Python framework for working with .nc files and applying methodologies to them as well as plotting


## Installation
**WARNING**: The environment must be compatible with all the dependencies and Cartopy probably needs it to be 3.9 or lower
**NOTE**: Cartopy has to be installed with conda because pip version does not work

To get the latest version:
```console
    $ conda create -n <your-env-name>
    $ conda activate <your-env-name>
    (<your-env-name>) $ conda install pip
    (<your-env-name>) $ conda install cartopy
    (<your-env-name>) $ pip install git+https://github.com/pabloduran016/Spy4Cast
    (<your-env-name>) $ conda install cartopy
```

To get the latest stable version:
```console
    $ pip install spy4cast
```

## Example

```python
from spy4cast.meteo import Anom
from spy4cast import Month, Region, set_silence, Dataset, F

# Define constants ---------------------------------------------------------------------------------- #
DATASET_DIR = '/Users/Shared/datasets/'
PLOTS_DIR = 'plots'
PLOT_DATA_DIR = 'data-anom'

chl_1km_monthly_Sep1997_Dec2020 = 'chl_1km_monthly_Sep1997_Dec2020.nc'
CHL = 'CHL'

chl_region = Region(30, 90, -5.3, -2, Month.MAR, Month.APR, 1998, 2020)

ds = Dataset(chl_1km_monthly_Sep1997_Dec2020, DATASET_DIR).open(CHL).slice(chl_region)

map_anom = Anom(ds, 'map')

map_anom.save('map_anomaly', PLOT_DATA_DIR)
# map_anom = Anom.load('map_anomaly', PLOT_DATA_DIR, type='map')
map_anom.plot(show_plot=True, save_fig=True, year=1999, name='anom-map-example.png', cmap='jet')

ts_anom = Anom(ds, 'ts')
ts_anom.save('ts_anomaly', PLOT_DATA_DIR)
# ts_anom = Anom.load('ts_anomaly', PLOT_DATA_DIR, type='ts')
ts_anom.plot(show_plot=True, save_fig=True, name='anom-ts-example.png')
```

**Output:**

![Example 1 plot](examples/anomer_example.png)

## Documentation
The documentation for this project is in https://spy4cast-docs.netlify.app

## References
- [xarray](https://www.xarray.pydata.org/en/stable/)
- [numpy](https://www.numpy.org/)
- [cartopy](https://www.scitools.org.uk/cartopy/docs/latest/)
- [matplotlib](https://www.matplotlib.org/stable/api/text_api.html#matplotlib.text.Text)
- [dask](https://www.dask.org/)
- [sphinx](https://www.sphinx-doc.org/)
