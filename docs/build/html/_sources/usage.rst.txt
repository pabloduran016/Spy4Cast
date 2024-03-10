Usage
=====

Installation
------------

To use Spy4Cast, first install it using git:

To get the latest version:

.. warning::

   The environment must be compatible with all the dependencies and Cartopy probably needs it to be 3.9 or lower


.. note::

    You have to have installed git (`Install Git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_)

.. note::

    You have to have installed anaconda (`Install Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_)


.. code-block:: console

    $ conda create -n <your-env-name> python=3.9
    $ conda activate <your-env-name>
    (<your-env-name>) $ conda install pip
    (<your-env-name>) $ conda install cartopy
    (<your-env-name>) $ pip install git+https://github.com/pabloduran016/Spy4Cast

.. note::

    Cartopy has to be installed with conda because pip version does not work


..
    To get the latest stable version:

    .. code-block:: console

       $ pip install spy4cast


Use cfgrib
----------

To use cfgrib you need to install the cfgrib package that depends on the eccodes package

On Linux
++++++++

.. code-block:: console

    (<your-env-name>) $ sudo apt-get install libeccodes0
    (<your-env-name>) $ pip install cfgrib


On Mac
++++++

.. code-block:: console

    (<your-env-name>) $ brew install eccodes
    (<your-env-name>) $ pip install cfgrib


Upgrade Version
---------------

.. code-block:: console

    (<your-env-name>) $ pip uninstall spy4cast
    (<your-env-name>) $ pip install --upgrade --no-cache-dir git+https://github.com/pabloduran016/Spy4Cast

.. warning::

     Sometimes the command above is not enough and you need to uninstall spy4cast
     before upgrading (:code:`pip uninstall spy4cast`)

Example
-------

Spy4Cast: Preprocess, MCA and Crossvalidation
+++++++++++++++++++++++++++++++++++++++++++++

Here is an example of how you can use Spy4Cast API to **RUN** the full Spy4Cast methodology and use the included plotting functions.

**Click** :download:`here <../../examples/example_of_crossvalidation.py>` **to download**

.. literalinclude:: ../../examples/example_of_crossvalidation.py

Here is an example of how you can use Spy4Cast API to **PLOT** the previously ran Spy4Cast methodology.

**Click** :download:`here <../../examples/example_of_plotting_precalculated_crossvalidation_and_mca.py>` **to download**

.. literalinclude:: ../../examples/example_of_plotting_precalculated_crossvalidation_and_mca.py

Here is an example of how you can use Spy4Cast API to **RUN** the Spy4Cast methodology and plot it using **CUSTOM PLOTTING FUNCTIONS**.

**Click** :download:`here <../../examples/example_of_crossvalidation_with_custom_plotting.py>` **to download**

.. literalinclude:: ../../examples/example_of_crossvalidation_with_custom_plotting.py

Climatology
+++++++++++

Here is an example of how you can use Spy4Cast API to plot the climatology of a given .nc dataset

.. literalinclude:: ../../examples/example_of_ploting_climatology_maps_and_time_series.py

**Output:**

.. image:: _static/images/clim-map-example.png
    :alt: Output for clim map
    :height: 25em
    :align: center


.. image:: _static/images/clim-ts-example.png
    :alt: Output for clim ts
    :height: 25em
    :align: center


Anomaly
+++++++

Here is an example of how you can use Spy4Cast API to plot the anomaly of a given .nc dataset

.. literalinclude:: ../../examples/example_of_ploting_anomaly_maps_and_time_series.py


**Output:**

.. image:: _static/images/anom-map-example.png
    :alt: Output for anom map
    :height: 25em
    :align: center

.. image:: _static/images/anom-ts-example.png
    :alt: Output for anom ts
    :height: 25em
    :align: center



