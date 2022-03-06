Usage
=====

Installation
------------

To use Spy4Cast, first install it using git:

To get the latest version:

.. code-block:: console

    $ pip install git+https://github.com/pabloduran016/Spy4Cast
    $ git clone https://github.com/pabloduran016/Spy4Cast
    $ cp Spy4Cast/requirements.txt requirements.txt
    $ rm -r Spy4Cast
    $ pip install -r requirements.txt


To get the latest stable version:

.. code-block:: console

   $ pip install spy4cast

Example
-------

Here is an example of how you can use Spy4Cast API to plot the anomaly of a given .nc dataset

.. literalinclude:: ../../examples/anomer_example.py