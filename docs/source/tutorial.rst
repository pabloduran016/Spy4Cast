.. _tutorial:

Tutorial
========

Spy4Cast offers a frontend to manage datasets in .nc format in python

Open
----

You can open a dataset really easily with a line of code using the `Dataset` interface:

.. code:: python

    from spy4cast import Dataset

    DIR = "data/"  # If no dir is specified it look sin the current directory
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR)

Slice
-----

Most of the times you would like to slice a region of the dataset. You will need to use `Region` and `Month` for that:

.. code:: python

    from spy4cast import Dataset, Month, Region  # --- new --- #

    DIR = "data/"
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR)

    # --- new --- #

    # Latitulde goes from -90 to 90 and longitude form -180 to 180.
    # If the dataset you use doesn't work like that and longiutde
    # goes from 0 to 360 when you open the dataset this will be
    # changed so you ALWAYS have to latitude from -90 to 90
    # and longitude form -180 to 180
    region = Region(
        lat0=-45,
        latf=0,
        lon0=-6,
        lonf=40,
        month0=Month.DEC,
        monthf=Month.MAR,
        # If the initial month is bigger than the final month (DEC --> MAR)
        # the dataset uses the year before for the initial month (1874 in thiss case)
        year0=1875,
        yearf=1990,
    )
    ds.slice(region)

    # You can also do this in one line like: ds = Dataset(NAME, dir=DIR).open(VAR).slice(region)

Save
----

Every methodology can be saved for future usage (`.save("prefix_", dir='saved_data_directory')`)

.. code:: python

    from spy4cast import Dataset, Region, Month
    from spy4cast.spy4cast import Preprocess

    DIR = "data/"
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR).slice(
        Region(-90, 90, -180, 180, Month.JAN, Month.MAR, 1870, 1995)
    )
    preprocesed = Preprocess(ds)
    preprocesed.save('save_preprocess_', dir='saved')

.. code:: python

    from spy4cast import Dataset, Region, Month
    from spy4cast.meteo import Clim

    DIR = "data/"
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR).slice(
        Region(-90, 90, -180, 180, Month.JAN, Month.MAR, 1870, 1995)
    )
    clim = Clim(ds, 'map')  # You can plot a time series with Clim(ds, 'ts')
    clim.plot(show_plot=True, save_fig=True, cmap='jet', dir='plots', name='plot.png')
    # --- new --- #
    clim.save('save_clim_', dir='saved')


Load
----

You can use the saved data with a simple line of code

.. code:: python

    from spy4cast.spy4cast import Preprocess

    preprocessed = Preprocess.load('save_preprocess_', dir='saved')
    preprocessed.plot(selected_year=1990, show_plot=True, save_fig=True, cmap='jet', dir='plots', name='plot.png')

.. code:: python

    from spy4cast.meteo import Clim

    clim = Clim.load('save_clim_', dir='saved')
    clim.plot(show_plot=True, save_fig=True, cmap='jet', dir='plots', name='plot.png')

.. note::

    Load and Save work for Clim, Anom, Preprocess, MCA, Crossvalidation and Validation (every methodology the API supports)

.. _spy4cast-tutorial:

Spy4Cast
--------

The main methodology of spy4cast is Spy4Cast :-).

It requires a predictor dataset and a predictand dataset. Here is an example which you can download :download:`here <_static/scripts/docs-spy4cast-example.py>`

.. code:: python

    from spy4cast import Dataset, Region, Month
    from spy4cast.spy4cast import Preprocess, MCA, Crossvalidation, Validation

    predictor = Dataset('predictor.nc').open('predictor-var').slice(
        Region(-20, 30, -5, 40, Month.DEC, Month.MAR, 1870, 1990)
    )

    predictand = Dataset('predictand.nc').open('predictand-var').slice(
        Region(-50, -10, -40, 40, Month.JUN, Month.AUG, 1871, 1991)
    )


Preprocess
++++++++++

We now preprocess everything. `nm` and `alpha` are required parameters

.. code:: python

    nm = 3
    alpha = 0.1

    predictor_preprocessed = Preprocess(predictor, order=5, period=11)  # If we supply `order` and `period` parameters, it applies a filter
    predictand_preprocessed = Preprocess(predictand)


MCA
+++

Apply MCA

.. code:: python

    mca = MCA(dsy=predictor_preprocessed, dsz=predictand_preprocessed, nm=nm, alpha=alpha)

Crossvalidation
+++++++++++++++

Apply Crossvalidation

.. code:: python

    cross = Crossvalidation(dsy=predictor_preprocessed, dsz=predictand_preprocessed, nm=nm, alpha=alpha)

Validation
++++++++++

Apply Validation: needs a training period to compute the training MCA which then applies through out the validting period

.. code:: python

    training_preprocessed_y = Preprocess(training_predictor)
    training_preprocessed_z = Preprocess(training_predictand)
    training_mca = MCA(training_preprocessed_y, training_preprocessed_z, nm=3, alpha=0.1)

    validating_preprocessed_y = Preprocess(validating_predictor)
    validating_preprocessed_z = Preprocess(validating_predictand)

    validation = Validation(training_mca, validating_preprocessed_y, validating_preprocessed_z)


Visualization
+++++++++++++

Check out the :ref:`plotting<plotting>` section.

Plot
----

You can learn all about plotting in the :ref:`Plotting section<plotting>`.

To plot the results of a methodology you can use the built in plot function. Its purpose is to
be fast and to serve you as a debugging tool. For final results we reccommend you to create your own
plotting functions.

Spy4Cast
++++++++

Each spy4cast methodology has its own plotting functions: :ref:`spy4cast-tutorial`.

.. _clim-tutorial:

Clim
++++

Clim performs the climatology for the given region

.. code:: python

    from spy4cast import Dataset, Region, Month
    from spy4cast.meteo import Clim

    DIR = "data/"
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR).slice(
        Region(-90, 90, -180, 180, Month.JAN, Month.MAR, 1870, 1995)
    )
    clim = Clim(ds, 'map')  # You can plot a time series with Clim(ds, 'ts')
    clim.plot(show_plot=True, save_fig=True, cmap='jet', dir='plots', name='plot.png')

You can slice a dataset with only a Month and a year (:code:`Region(-90, 90, -180, 180, Month.JAN, Month.JAN, 1900, 1900)`)
and plot the clmatollogy of this dataset if you want to plot a certain month and year.

.. _anom-tutorial:

Anom
++++

Anom performs the anomaly for the given region

.. code:: python

    from spy4cast import Dataset, Region, Month
    from spy4cast.meteo import Anom

    DIR = "data/"
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR).slice(
        Region(-90, 90, -180, 180, Month.JAN, Month.MAR, 1870, 1995)
    )
    anom = Anom(ds, 'map')  # You can plot a time series with Clim(ds, 'ts')
    # A year is needed because Anom produces lots of maps (if you use 'ts', the year parameter becomes invalid)
    anom.plot(show_plot=True, save_fig=True, year=1990, cmap='jet', dir='plots', name='plot.png')
