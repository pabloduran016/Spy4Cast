# Spy4Cast

Python framework for working with .nc files and applying methodologies to them as well as plotting


## Setting up
Install dependicies by running the following
```console
pip install -r requirements.txt
```
**Example:**  
```python
import spy4cast as spy
from spy4cast.stypes import Slise, Month, F


DATASETS_DIR = "/datasets/"
HadISST_sst = "HadISST_sst.nc"

sl = Slise(
    latitude_min=-45,
    latitude_max=45,
    longitude_min=-100,
    longitude_max=100,
    initial_month=Month.JAN,
    final_month=Month.MAR,
    initial_year=1871,
    final_year=2020,
    selected_year=1990,
)
spy.AnomerMap(dataset_dir=DATASETS_DIR, dataset_name=HadISST_sst) \
    .load_dataset() \
    .slice_dataset(sl) \
    .apply() \
    .run(F.SHOW_PLOT | F.SAVE_FIG, slise=sl)
```
**Output:**    
  
![Example 1 plot](examples/anomer_example.png)

## ReadData
TBD

## Meteo
TBD

## Plotters and Prokers
TBD
### AnomerTS
TBD
### ClimerTS
TBD
### AnomerMap
TBD
### ClimerMap
TBD

## Spy4Caster
TBD

## References
- [xarray](https://xarray.pydata.org/en/stable/)
- [numpy](https://numpy.org/)
- [cartopy](https://scitools.org.uk/cartopy/docs/latest/)
- [matplotlib](https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text)
