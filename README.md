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
    lat0=-45, latf=45,
    lon0=-100, lonf=100,
    month0=Month.JAN, monthf=Month.MAR,
    year0=1871, yearf=2020,
    sy=1990,
)
spy.AnomerMap(dataset_dir=DATASETS_DIR, dataset_name=HadISST_sst) \
    .load_dataset() \
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

## Stypes
    Collection of data sctructures used across the API and for the users convenience  
    
    Color: equivalent to tuple[float, float, float]
    
    TimeStamp: timepstamp type equivalent to pd.Timestamp | datetime.datetime

    T_FORMAT: format used in dates compatible with datetime.strftime '%d/%m/%Y %H:%M:%S'
    
    Month: IntEnum for each month. (1 -> JAN, 12 -> DEC)
    
    Slise: dataclass that is sed for slicing. Slise insetead of Slice explained in `# Important Notes` 
        lat0 (optional float): minimum latitude 
        latf (optional float): maximum latitude
        lon0 (optional float): minimum longitude
        lonf (optional float): maximum longitude
        month0 (optional int | syp4cast.stypes.Month): initial month (1 -> JAN, 12 -> DEC) 
        monthf (optional int | syp4cast.stypes.Month): final month (1 -> JAN, 12 -> DEC)
        year0 (optional int): initial month
        yearf (optional int): final month
        sy (optional int): Selected year. It is optional, only readers like Anomer need a specified year
          to plot
        
        Slise.default(month0=Month.JAN, monthf=Month.DEC, year0=0, yearf=2000, sy=None)
          Returns a Slise that has latitude and longitude as wide as possible.
            `Slise(-90, 90, -180, 180, month0, monthf, year0, yearf, sy)`
    
    F: IntFlag enum used is plotting:
        SAVE_DATA
        SAVE_FIG
        SILENT_ERRORS
        SHOW_PLOT

        staticmethod F.checkf(f: F, other: int | F) -> bool:
            return (other & f) == f 
    
    ChunkType: Type of the chunk passed intop dask. 
      Equivalent to int | tuple[int, ...] | tuple[tuple[int, ...] ,...] | dict[str | int: int]

    RDArgs: dataclass that can be used to create the arguments passed into readers like 
      syp4cast.readers.Spy4caster that create multiple ReadData objects
        
        RDArgs.as_dict: returns its attributes as dict to pass into ReadData with the `**` operator
    
    RDArgsDict: TypedDict of RDArgs.as_ddict return value
    

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
