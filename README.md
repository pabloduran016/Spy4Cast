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

## Important Note:

· If you see anywhere in the docs or in the files `slise` and think it is a typo, it is not. Python has a 
built-in function called `slice` and in this library we have decided to use `slise` to avoid unexpected 
behaviours. I hope it is not too ugly...

## Stypes
TBD

## Errors
TBD

## ReadData
    class spy4cast.read_data.ReadData 
        This class enables you to load, slice and modify the data in a netcdf4 file confortably.
        
        It is concatenable, meaning that the output of most methods is the object itself so that you can 
        concatenate methods easily.

        def __init__(dataset_dir=None, dataset_name=None, variable=None, plot_dir=None,
            plot_name=None, plot_data_dir=None, force_name=False, chunks=None)
                                               
            · dataset_dir (optional str, default to ''): Directory where the dataset you want to 
                use is located
            · dataset_name (optional str, default to 'dataset.nc'): Name of the dataset
            · variable (optional str): Variable to evaluate. If it is not set, the program will try 
                to recgnise it by discarding time and coordinate variables 
            · plot_dir (optional str, default to ''): Directory to store the plot if later created
            · plot_name (optional str, default to 'plot.png'): Name of the plot saved if later created
            · plot_data_dir (optional str, default to ''): Directory of the data saved if later saved
            · force_name (optional bool, default top): Indicates wether or not inforce the names set above. 
                If false the name will be modified not to overwrite any existing files
            · chunks (optional int | tuple[int] | tuple[tuple[int]] | dict[str | int: int]): Argument passed 
                when loading the datasets (see chunks in dask library) 

        property time: Returns the time variable of the data evaluated. They key used is recognised automatically
        
        property latitude: Returns the latitude variable of the data evaluated. They key used is recognised automatically
        
        property longtude: Returns the longtude variable of the data evaluated. They key used is recognised automatically
        
        property shape: Returns the shape variable of the data evaluated.

        def check_variables(slise=None)
            Checks if the variable selected and the slise (only time-related part), if provided, is valid for the given dataset.

            · slise (optional spy4cast.stypes.Slise): Determines how the data has been cut

            Raises:
                · ValueError: if the dataset ha not been loaded
                · spy4cast.errors.VariableSelectionError: if the variable selected is not valid
                · spy4cast.errors.TimeBoundsSelectionError: if the time slise is not valid 
                · spy4cast.errors.SelectedYearError: if the selected_year (if provided) is not valid         

        def slice_dataset(slise):        
            Method that slices for you the dataset accorging to a slise. 
            It first calls check_slise method
            
            · slise: Slise to use

            Note: If the season contains months from different years (NOV-DEC-JAN-FEB for example)
            the initial year is applied to the month which comes at last (FEB). In this example, the
            data that will be used for NOV is on year before the initial year so keep this in mind Meteo
            if your dataset doesn't contain that specific year.

        def save_fig_data():
            Saves the data as a netcdf4 file in the path specified in __init__

        def get_dataset_info():
            Returns a tuple where the first element is the dataset name and the second is a dict with keys:
                `title`: dataset name without the extension 
                `from`: initial date of the dataset loaded
                `to`: final date of the dataset
                `variable`: variable used
            Example: 
              ('HadISST_sst.nc', {'title': 'HadISST_sst', 'from': 'Jan 1870', 'to': 'May 2020', 'variable': 'sst'})
        

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
