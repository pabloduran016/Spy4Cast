# Spy4Cast
![Icon](docs/source/_static/favicon.png)

Python framework for working with .nc files and applying methodologies to them as well as plotting


## Installation

To get the latest version:
```console
    $ conda create -n <your-env-name>
    $ conda activate <your-env-name>
    (<your-env-name>) $ conda install pip
    (<your-env-name>) $ pip install git+https://github.com/pabloduran016/Spy4Cast
    (<your-env-name>) $ conda install cartopy
```

To get the latest stable version:
```console
    $ pip install spy4cast
```

## Example

```python
from spy4cast import AnomerMap, Slise, Month, F

DATASETS_DIR = "/datasets/"
HadISST_sst = "HadISST_sst.nc"

sl = Slise(
    lat0=-45, latf=45,
    lon0=-100, lonf=100,
    month0=Month.JAN, monthf=Month.MAR,
    year0=1871, yearf=2020,
    sy=1990,
)
AnomerMap(dataset_dir=DATASETS_DIR, dataset_name=HadISST_sst) \ 
    .open_dataset() \
    .slice_dataset(sl) \
    .apply() \
    .run(F.SHOW_PLOT | F.SAVE_FIG, slise=sl)
```
**Output:**    
  
![Example 1 plot](examples/anomer_example.png)

## Important Notes:

· If you see anywhere in the docs or in the files `slise` and think it is a typo, it is not. Python has a 
built-in function called `slice` and in this library we have decided to use `slise` to avoid unexpected 
behaviours. I hope it is not too ugly...

## Documentation
The documentation for this project is in [https://www.spy4cast-docs.netlify.app]

## References
- [xarray](https://www.xarray.pydata.org/en/stable/)
- [numpy](https://www.numpy.org/)
- [cartopy](https://www.scitools.org.uk/cartopy/docs/latest/)
- [matplotlib](https://www.matplotlib.org/stable/api/text_api.html#matplotlib.text.Text)
- [dask](https://www.dask.org/)
- [sphinx](https://www.sphinx-doc.org/)
