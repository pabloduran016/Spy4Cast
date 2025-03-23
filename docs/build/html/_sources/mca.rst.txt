.. _mca:

MCA
===

.. **MCA** is a well known discriminant analysis technique used for finding coupled 
.. patterns in climate data. This technique consists on finding linear combinations 
.. of the time series of a predictand and a predictor field in a way that their covariance is maximized. 
.. This is done through the diagonalization of the covariance matrix, a problem which is solved by using 
.. singular value decomposition (SVD).
..
.. MCA is a powerful tool when wrapping large amount of climate information in a few modes of variability. 
.. In this context, MCA analysis provides spatial patterns of the predictor and predictand field which are 
.. related by teleconnections. For example, when using as predictor field the tropical Pacific sea surface 
.. temperature (SST) anomalies, and, as predictand field, the overlying anomalous sea level pressure (SLP),
.. the leading mode is found to be a couple of anomalous SST and SLP patterns which are highly connected. 
.. The SST pattern will have the structure of El Niño and the SLP pattern will be the Southern Oscillation, 
.. which is the covarying SLP pattern forced by El Niño. The time expansion coefficients associated to each 
.. mode will represent how strongly each mode loads on each year and will be highly related to the Southern
.. Oscillation and Niño indices.
..
.. Spy4Cast implements the methodology of Maximum Covariance (MCA) analysis which needs  a predictor field,
.. $\bm Y$, and a predictand field, $\bm Z$. These fields are organized in space-time matrices where each 
.. row represents the evolution of a certain point in space ($n_y$ and $n_z$ points for $\bm Y$ and $\bm Z$
.. respectively) across $n_t$ discrete time intervals. When covarying on time, $\bm Y$ and $\bm Z$ must 
.. have the same time dimension ($n_t$), even though they can have a certain lag. However, they do not need
.. to have (in general) the same space dimension and so the covariance matrix, defined as $\bm C = \bm Y 
.. \cdot \bm Z^T$, has dimensions $(n_y, n_z)$.

Maximum Covariance Analysis (MCA) is a widely used discriminant analysis technique for identifying coupled 
patterns in climate data. It works by finding linear combinations of a predictor and predictand field that 
maximize their covariance through singular value decomposition (SVD). MCA is particularly useful for 
summarizing large climate datasets into a few dominant modes of variability, revealing spatial patterns 
linked by teleconnections. For example, applying MCA to tropical Pacific sea surface temperature (SST) 
anomalies and anomalous sea level pressure (SLP) identifies El Niño-related SST patterns and the Southern 
Oscillation in SLP. Spy4Cast implements MCA by organizing predictor and predictand fields into space-time 
matrices, ensuring they share a common time dimension while allowing for spatial differences, with their 
covariance matrix defined as :math:`C = Y \cdot Z^T`.

MCA is performed with class :class:`spy4cast.spy4cast.MCA`

To perform MCA on predictor, :math:`Y` and a predictand :math:`Z` with Spy4Cast you need to load the 
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
   mca = spy4cast.MCA(yp, zp, nm=3, alpha=0.1)
   mca.plot(show_plot=True, halt_program=True)


The following figure is an example of the figures that can be created with MCA. 
It is done with the code from the manual :ref:`spy4cast-manual` of example 
`EquatorialAtlantic_Impact_Nino.ipynb <https://github.com/pabloduran016/Spy4CastManual/blob/main/EquatorialAtlantic_Impact_Nino.ipynb>`_

.. image:: _static/images/mca_EquatorialAtlantic_Impact_Nino.png
    :alt: Example MCA: EquatorialAtlantic_Impact_Nino.ipynb

