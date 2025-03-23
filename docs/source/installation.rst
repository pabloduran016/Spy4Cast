Installation
============


To use Spy4Cast, first install it using git:

To get the latest version:

.. warning::

   The environment must be compatible with all the dependencies and Cartopy may need python version to be 3.9 or lower


.. note::

    You have to have installed git (`Install Git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_)

.. note::

    You have to have installed anaconda (`Install Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_)


.. code-block:: console

    $ conda create -n <your-env-name>
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
    (<your-env-name>) $ pip install cfgrib findlibs==0.0.5


On Mac
++++++

.. code-block:: console

    (<your-env-name>) $ brew install eccodes
    (<your-env-name>) $ pip install cfgrib findlibs==0.0.5

.. note::
    
    ``eccodes`` depends on ``findlibs``, whose newer version does not work with ``python3.9`` so make
    sure to specify the version number.


Upgrade Version
---------------

.. code-block:: console

    (<your-env-name>) $ pip uninstall spy4cast
    (<your-env-name>) $ pip install --upgrade --no-cache-dir git+https://github.com/pabloduran016/Spy4Cast

.. warning::

     Sometimes the command above is not enough and you need to uninstall spy4cast
     before upgrading (:code:`pip uninstall spy4cast`)

