"""
MCA with PCA based dimensionality reduction
"""
import numpy as np
import os
from spy4cast import Dataset, Region, Month
from spy4cast.spy4cast import Crossvalidation, Preprocess, MCA
import numpy as np

""" Atlantic Impact on Pacific El Niño """

filename = os.path.splitext(os.path.basename(__file__))[0]

y_ds = Dataset("HadISST_sst_chopped.nc", "./datasets").open("sst")
y_ds = y_ds.slice(
    Region(lat0=-30, latf=10, lon0=-60, lonf=15, month0=Month.JUN, monthf=Month.AUG, year0=1970, yearf=2019),
    skip=1
)
y = Preprocess(y_ds, period=8, order=4)

z_ds = Dataset("HadISST_sst_chopped.nc", "./datasets").open("sst")
z_ds = z_ds.slice(
    Region(lat0=-30, latf=30, lon0=-200, lonf=-60, month0=Month.DEC, monthf=Month.FEB,  
           # year0 and yearf refer to monthf so the slice will span from DEC 1970 to FEB 2020 
           year0=1971, yearf=2020),
    skip=1
)
z = Preprocess(z_ds)

pca_modes = 10
alpha = .1

mca_y = MCA(y, y, nm=pca_modes, alpha=alpha)
mca_z = MCA(z, z, nm=pca_modes, alpha=alpha)

# Y reconstructed
y_r = y.copy(data=np.dot(mca_y.SUY, mca_y.Us))
# Z reconstructed
z_r = z.copy(data=np.dot(mca_z.SUY, mca_z.Us))

nm = 3

mca = MCA(y, z, nm=nm, alpha=alpha)
mca.plot(
    save_fig=True, name="elinino_mca.png",
    folder=f"./plots-{filename}",
)

mca = MCA(y_r, z_r, nm=nm, alpha=alpha)
mca.plot(
    save_fig=True, name="elinino_mca_reduced_dimensionality.png",
    folder=f"./plots-{filename}",
)

cross = Crossvalidation(y, z, nm=nm, alpha=alpha, num_svdvals=nm)
cross.plot(
    save_fig=True, name="elinino_cross.png",
    folder=f"./plots-{filename}",
)

cross = Crossvalidation(y_r, z_r, nm=nm, alpha=alpha, num_svdvals=nm)
cross.plot(
    save_fig=True, name="elinino_cross_reduced_dimensionality.png",
    folder=f"./plots-{filename}",
)

""" Pacific Impact Sahelian Rainfall """

y_ds = Dataset("HadISST_sst_chopped.nc", "./datasets").open("sst")
y_ds = y_ds.slice(
    Region(lat0=-30, latf=30,
           lon0=-200, lonf=-60,
           month0=Month.APR, monthf=Month.JUN,
           year0=1970, yearf=2000),
)
y = Preprocess(y_ds, period=4, order=4)

z_ds = Dataset("cru_ts4_pre_chopped.nc", "./datasets").open("pre")
z_ds = z_ds.slice(
    Region(lat0=0, latf=25,
           lon0=-20, lonf=20,
           month0=Month.JUL, monthf=Month.SEP,
           year0=1970, yearf=2000),
)
z = Preprocess(z_ds)

pca_modes = 10
alpha = .1

mca_y = MCA(y, y, nm=pca_modes, alpha=alpha)
mca_z = MCA(z, z, nm=pca_modes, alpha=alpha)

# Y reconstructed
y_r = y.copy(data=np.dot(mca_y.SUY, mca_y.Us))
# Z reconstructed
z_r = z.copy(data=np.dot(mca_z.SUY, mca_z.Us))

nm = 3

mca = MCA(y, z, nm=nm, alpha=alpha)
mca.plot(
    save_fig=True, name="sahel_mca.png",
    folder=f"./plots-{filename}",
)

mca = MCA(y_r, z_r, nm=nm, alpha=alpha)
mca.plot(
    save_fig=True, name="sahel_mca_reduced_dimensionality.png",
    cmap="BrBG",
    folder=f"./plots-{filename}",
)

cross = Crossvalidation(y, z, nm=nm, alpha=alpha, num_svdvals=nm)
cross.plot(
    save_fig=True, name="sahel_cross.png",
    cmap="BrBG", 
    folder=f"./plots-{filename}",
)

cross = Crossvalidation(y_r, z_r, nm=nm, alpha=alpha, num_svdvals=nm)
cross.plot(
    save_fig=True, name="sahel_cross_reduced_dimensionality.png",
    folder=f"./plots-{filename}",
)

import matplotlib.pyplot as plt

plt.show()


