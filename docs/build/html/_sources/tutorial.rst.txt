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

Most of the times you would like to slice a region of the dataset. You will need to use `Slise` and `Month` for that:

.. code:: python

    from spy4cast import Dataset, Month, Slise  # --- new --- #

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
    slise = Slise(
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
    ds.slice(slise)

    # You can also do this in one line like: ds = Dataset(NAME, dir=DIR).open(VAR).slice(slise)

Plot
----

You can plot the dataset using two methodologies

.. _clim-tutorial:

Clim
++++

Clim performs the climatology for the given region

.. code:: python

    from spy4cast import Dataset, Slise, Month
    from spy4cast.meteo import Clim

    DIR = "data/"
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR).slice(
        Slise(-90, 90, -180, 180, Month.JAN, Month.MAR, 1870, 1995)
    )
    clim = Clim(ds, 'map')  # You can plot a time series with Clim(ds, 'ts')
    clim.plot(show_plot=True, save_fig=True, cmap='jet', dir='plots', name='plot.png')

You can slice a dataset with only a Month and a year (:code:`Slise(-90, 90, -180, 180, Month.JAN, Month.JAN, 1900, 1900)`)
and plot the clmatollogy of this dataset if you want to plot a certain month and year.

.. _anom-tutorial:

Anom
++++

Anom performs the anomaly for the given region

.. code:: python

    from spy4cast import Dataset, Slise, Month
    from spy4cast.meteo import Anom

    DIR = "data/"
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR).slice(
        Slise(-90, 90, -180, 180, Month.JAN, Month.MAR, 1870, 1995)
    )
    anom = Anom(ds, 'map')  # You can plot a time series with Clim(ds, 'ts')
    # A year is needed because Anom produces lots of maps (if you use 'ts', the year parameter becomes invalid)
    anom.plot(show_plot=True, save_fig=True, year=1990, cmap='jet', dir='plots', name='plot.png')


Save
----

Every methodology can be saved for future usage

.. code:: python

    from spy4cast import Dataset, Slise, Month
    from spy4cast.meteo import Clim

    DIR = "data/"
    NAME = "dataset.nc"
    VAR = "sst"  # Variable to use in the dataset. For example `sst`
    ds = Dataset(NAME, dir=DIR).open(VAR).slice(
        Slise(-90, 90, -180, 180, Month.JAN, Month.MAR, 1870, 1995)
    )
    clim = Clim(ds, 'map')  # You can plot a time series with Clim(ds, 'ts')
    clim.plot(show_plot=True, save_fig=True, cmap='jet', dir='plots', name='plot.png')
    # --- new --- #
    clim.save('save_clim_', dir='saved')


Load
----

You can use the saved data with a simple line of code

.. code:: python

    from spy4cast.meteo import Clim

    clim.load('save_clim_', dir='saved')
    clim.plot(show_plot=True, save_fig=True, cmap='jet', dir='plots', name='plot.png')


.. note::

    Load and Save work for Clim, Anom, Preprocess, MCA and Crossvalidation (every methodology the API supports)

.. _spy4cast-tutorial:

Spy4Cast
--------

The main methodology of spy4cast is Spy4Cast :-).

It requires a predictor dataset and a predictand dataset. Here is an example which you can download :download:`here <_static/docs-spy4cast-example.py>`

.. code:: python

    from spy4cast import Dataset, Slise, Month
    from spy4cast.spy4cast import Preprocess, MCA, Crossvalidation

    predictor = Dataset('predictor.nc').open('predictor-var').slice(
        Slise(-20, 30, -5, 40, Month.DEC, Month.MAR, 1870, 1990)
    )

    predictand = Dataset('predictand.nc').open('predictand-var').slice(
        Slise(-50, -10, -40, 40, Month.JUN, Month.AUG, 1871, 1991)
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

    # We save the MCA data to avoid runnning it again as it takes some time to do it
    # We can also send this data across to other which the can load with just:
    # `MCA.load('mca_', dir='saved', dsy=predictor_preprocessed`, dsz=predictand_preprocessed)`
    # NOTE: predictor and predictand datasets can also be saved and load if necessary
    mca.save('mca_', dir='saved')
    mca.plot(save_fig=True, name='mca.png')  # We don't add F.SHOW_PLOT because we will show all the plots together afterwards


Crossvalidation
+++++++++++++++

Apply Crossvalidation

.. code:: python

    cross = Crossvalidation(dsy=predictor_preprocessed, dsz=predictand_preprocessed, nm=nm, alpha=alpha)
    cross.save('cross_', dir='saved')
    cross.plot(save_fig=True, name='cross.png')


Visualization
+++++++++++++

.. code:: python

    # We can show all the plots together by using the matplotlib library
    # which was used to create them. Support for this mechanism is not
    # garanteed.

    import matplotlib.pyplot as plt
    plt.show()

Customization
+++++++++++++

.. todo:: Not documented yet