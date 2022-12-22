from spy4cast import Dataset, Slise, Month
from spy4cast.spy4cast import Preprocess, MCA, Crossvalidation

predictor = Dataset('predictor.nc').open('predictor-var').slice(
    Slise(-20, 30, -5, 40, Month.DEC, Month.MAR, 1870, 1990)
)

predictand = Dataset('predictand.nc').open('predictand-var').slice(
    Slise(-50, -10, -40, 40, Month.JUN, Month.AUG, 1871, 1991)
)

nm = 3
alpha = 0.1

predictor_preprocessed = Preprocess(predictor, order=5, period=11)  # If we supply `order` and `period` parameters, it applies a filter
predictand_preprocessed = Preprocess(predictand)
mca = MCA(dsy=predictor_preprocessed, dsz=predictand_preprocessed, nm=nm, alpha=alpha)

# We save the MCA data to avoid runnning it again as it takes some time to do it
# We can also send this data across to other which the can load with just:
# `MCA.load('mca_', dir='saved', dsy=predictor_preprocessed`, dsz=predictand_preprocessed)`
# NOTE: predictor and predictand datasets can also be saved and load if necessary
mca.save('mca_', dir='saved')
mca.plot(save_fig=True, name='mca.png')  # We don't add F.SHOW_PLOT because we will show all the plots together afterwards

cross = Crossvalidation(dsy=predictor_preprocessed, dsz=predictand_preprocessed, nm=nm, alpha=alpha)
cross.save('cross_', dir='saved')
cross.plot(save_fig=True, name='cross.png')

# We can show all the plots together by using the matplotlib library
# which was used to create them. Support for this mechanism is not
# garanteed.

import matplotlib.pyplot as plt
plt.show()