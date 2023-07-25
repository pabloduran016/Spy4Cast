from spy4cast import Dataset, Region, Month
from spy4cast.spy4cast import Preprocess, MCA, Crossvalidation, Validation

predictor = Dataset('predictor.nc').open('predictor-var').slice(
    Region(-20, 30, -5, 40, Month.DEC, Month.MAR, 1870, 1990)
)

predictand = Dataset('predictand.nc').open('predictand-var').slice(
    Region(-50, -10, -40, 40, Month.JUN, Month.AUG, 1871, 1991)
)

nm = 3
alpha = 0.1

predictor_preprocessed = Preprocess(predictor, order=5, period=11)  # If we supply `order` and `period` parameters, it applies a filter
predictand_preprocessed = Preprocess(predictand)
mca = MCA(dsy=predictor_preprocessed, dsz=predictand_preprocessed, nm=nm, alpha=alpha)

# We save the MCA data to avoid runnning it again as it takes some time to do it
# We can also send this data across to other which the can load with just:
# `MCA.load('mca_', folder='saved', dsy=predictor_preprocessed`, dsz=predictand_preprocessed)`
# NOTE: predictor and predictand datasets can also be saved and load if necessary
mca.save('mca_', folder='saved')
mca.plot(save_fig=True, name='mca.png')  # We don't add F.SHOW_PLOT because we will show all the plots together afterwards

cross = Crossvalidation(dsy=predictor_preprocessed, dsz=predictand_preprocessed, nm=nm, alpha=alpha)
cross.save('cross_', folder='saved')
cross.plot(save_fig=True, name='cross.png')


training_predictor = Dataset('predictor.nc').open('predictor-var')  # JAN-1870 : MAY-2020
training_predictor.slice(Region(5, 45, -90, -5, Month.JUN, Month.JUL, 1997, 2007), skip=3)

training_predictand = Dataset('predictand.nc').open('predictand-var')  # JAN-1959 : DEC-2004
training_predictand.slice(Region(36, 37, -5.3, -2, Month.MAR, Month.APR, 1998, 2008), skip=3)

validating_predictor = Dataset('predictor.nc').open('predictor-var')  # JAN-1870 : MAY-2020
validating_predictor.slice(Region(5, 45, -90, -5, Month.JUN, Month.JUL, 2008, 2018), skip=3)

validating_predictand = Dataset('predictand.nc').open('predictand-var')  # JAN-1959 : DEC-2004
validating_predictand.slice(Region(36, 37, -5.3, -2, Month.MAR, Month.APR, 2009, 2019), skip=3)

training_preprocessed_y = Preprocess(training_predictor)
training_preprocessed_z = Preprocess(training_predictand)
training_mca = MCA(training_preprocessed_y, training_preprocessed_z, nm=3, alpha=0.1)

validating_preprocessed_y = Preprocess(validating_predictor)
validating_preprocessed_z = Preprocess(validating_predictand)

validation = Validation(training_mca, validating_preprocessed_y, validating_preprocessed_z)
validation.plot()

# We can show all the plots together by using the matplotlib library
# which was used to create them. Support for this mechanism is not
# garanteed.

import matplotlib.pyplot as plt
plt.show()