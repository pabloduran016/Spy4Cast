.. _variables:

Variables
=========

Each methodology holds a different number of variables. You can lear about it here!

**Remember you can check the variables inside by accessing** `.var_names` **property.**

Dimension Length Meaning
++++++++++++++++++++++++

- `nt`: Length of the time coordinate
- `nlat`: Length of the latitude coordinate
- `nlon`: Length of the longitude coordinate
- `ns`: Number of space data points: `nlat * nlon`
- `nm`: Number of modes selected

.. note::

    For dimension length of different variables it can be stated by prefixing `y_`, `z_`, `validating_z_`, `training_z_`, etc ...
    For example, for `space` coordinate of predictor (y): **y_ns**.

.. toctree::
    :maxdepth: 1

    dataset
    preprocess
    mca
    crossvalidation
    validation
    anom
    clim
