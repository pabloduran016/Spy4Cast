Usage
=====

Installation
------------

To use Spy4Cast, first install it using git:

To get the latest version:

.. code-block:: console
    $ conda create -n <your-env-name>
    $ conda activate <your-env-name>
    (<your-env-name>) $ conda install pip
    (<your-env-name>) $ pip install git+https://github.com/pabloduran016/Spy4Cast
    (<your-env-name>) $ conda install cartopy

To get the latest stable version:

.. code-block:: console

   $ pip install spy4cast

Example
-------

Here is an example of how you can use Spy4Cast API to plot the anomaly of a given .nc dataset

.. literalinclude:: ../../examples/anomer_example.py