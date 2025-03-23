.. crossvalidation:

Crossvalidation
===============

To evaluate the skill of the model a crossvalidated hindcast is produced by applying the leave 
one year out methodology. In this way, the data corresponding to each year is omitted from the 
:math:`Y` and :math:`Z` matrices  and :ref:`MCA <mca>` is applied with the remaining years. In each iteration,
:math:`\Psi` is calculated, as well as the predicted :math:`\hat{Z}` for the omitted year,
using the value of the predictor :math:`Y` for that particular year. With the crossvalidated 
hindcast, the skill is evaluated by calculating the anomaly correlation coefficient (ACC) 
between :math:`Z` and the crossvalidated :math:`\hat{Z}`. For each :ref:`MCA <mca>` iteration, the scf, 
:math:`U_s` and :math:`V_s`, :math:`S_{UZ}` and :math:`S_{UY}` are stored in order to test the 
stability of the modes and its sensitivity for each particular year. 

Crossvalidation is performed with class :class:`spy4cast.spy4cast.Crossvalidation`

To perform Crossvalidation on predictor, :math:`Y` and a predictand :math:`Z` with Spy4Cast you need to load the 
:class:`datasets <spy4cast.dataset.Dataset>`, :py:meth:`slice <spy4cast.dataset.Dataset.slice>` them and 
:class:`preprocess <spy4cast.spy4cast.Preprocess>` them first:

.. code:: python

   from spy4cast import Dataset, Region, Month, spy4cast

   y = Dataset("predictor.nc", folder="datasets").open().slice(
       Region(lat0=-20, latf=20, lon0=-10, lonf=40, 
              month0=Month.JAN, monthf=Month.MAR, year0=1964, yearf=1994))
   z = Dataset("predictand.nc", folder="datasets").open().slice(
       Region(lat0=-10, latf=-40, lon0=-180, lonf=-100, 
              month0=Month.FEB, monthf=Month.APR, year0=1964, yearf=1994))
   yp = spy4cast.Preprocess(y)
   zp = spy4cast.Preprocess(z)
   cross = spy4cast.Crossvalidation(yp, zp, nm=3, alpha=0.1)
   cross.plot(show_plot=True, halt_program=True)


The following figure is an example of the figures that can be created with Crossvalidation.
It is done with the code from the manual :ref:`spy4cast-manual` of example 
`EquatorialAtlantic_Impact_Nino.ipynb <https://github.com/pabloduran016/Spy4CastManual/blob/main/EquatorialAtlantic_Impact_Nino.ipynb>`_

.. image:: _static/images/cross_EquatorialAtlantic_Impact_Nino.png
    :alt: Example Crossvalidation: EquatorialAtlantic_Impact_Nino.ipynb

