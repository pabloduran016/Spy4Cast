.. validation:

Validation
==========

Althogh :ref:`MCA <mca>` provides a prediction based on principal modes of variability, 
validation is required for all models in order to test its reliability, assuming stationarity,
i.e: the training period contains the same relations that the validation period. In this case 
the predictand field :math:`\hat{Z}` is predicted calculating :math:`\Psi` by applying the 
:ref:`MCA <mca>` for a certain training period and multiplying it by the predictor field 
:math:`Y` for a validation period.

Spy4Cast is able to perform a validation methodology to look for non-stationary relations. 
To run validation, four datasets need to be preprocessed, corresponding to the predictor and 
predictand field, for the training period and for the validating period. With the training 
dataset, :ref:`MCA <mca>` is applied to produce the prediction model parameters. In this way, 
the :math:`\Psi` matrix is calculated for the :ref:`MCA <mca>` modes characteristic of the 
training period. Then, using  :math:`\Psi` and the preprocessed predictor field for the validation 
period, the validated predictand field is calculated. Thus, the model is  tested against a 
predictor and predictand field of a different period than the training one. 

Validation is performed with class :class:`spy4cast.spy4cast.Validation`

To perform Validation on predictor, :math:`Y` and a predictand :math:`Z` with Spy4Cast you need to load the 
:class:`datasets <spy4cast.dataset.Dataset>`, :py:meth:`slice <spy4cast.dataset.Dataset.slice>` them and 
:class:`preprocess <spy4cast.spy4cast.Preprocess>` them first:

.. code:: python

   from spy4cast import Dataset, Region, Month, spy4cast

   t_y = Dataset("predictor.nc", folder="datasets").open().slice(
       Region(lat0=-20, latf=20, lon0=-10, lonf=40, 
              month0=Month.JAN, monthf=Month.MAR, year0=1964, yearf=1994))
   t_z = Dataset("predictand.nc", folder="datasets").open().slice(
       Region(lat0=-10, latf=-40, lon0=-180, lonf=-100, 
              month0=Month.FEB, monthf=Month.APR, year0=1964, yearf=1994))
   t_yp = spy4cast.Preprocess(t_y)
   t_zp = spy4cast.Preprocess(t_z)
   t_mca = spy4cast.MCA(t_yp, t_zp, nm=3, alpha=0.1)


   v_y = Dataset("predictor.nc", folder="datasets").open().slice(
       Region(lat0=-20, latf=20, lon0=-10, lonf=40, 
              month0=Month.JAN, monthf=Month.MAR, year0=1964, yearf=1994))
   v_z = Dataset("predictand.nc", folder="datasets").open().slice(
       Region(lat0=-10, latf=-40, lon0=-180, lonf=-100, 
              month0=Month.FEB, monthf=Month.APR, year0=1964, yearf=1994))
   v_yp = spy4cast.Preprocess(v_y)
   v_zp = spy4cast.Preprocess(v_z)

   val = Validation(t_mca, v_yp, z_yp)
   val.plot(show_plot=True, halt_program=True)


The following figure is an example of the figures that can be created with Validation.
It is done with the code from the manual :ref:`spy4cast-manual` of example 
`EquatorialAtlantic_Impact_Nino.ipynb <https://github.com/pabloduran016/Spy4CastManual/blob/main/EquatorialAtlantic_Impact_Nino.ipynb>`_

.. image:: _static/images/validation_EquatorialAtlantic_Impact_Nino.png
    :alt: Example Validation: EquatorialAtlantic_Impact_Nino.ipynb

