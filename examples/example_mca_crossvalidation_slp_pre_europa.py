"""
This example is an example of handling datasets with over 10^4 data points
"""
import os
from spy4cast import Dataset, Region, Month
from spy4cast.spy4cast import Crossvalidation, Preprocess, MCA

DATASETS_DIR = "/Users/Shared/datasets/"

y = Dataset("CDS_era5_slp_monthly_1940-2023.grib", DATASETS_DIR).open("msl").slice(Region(
    26, 70, -80, 40, Month.FEB, Month.APR, 1970, 2000
), skip=2)
z = Dataset("cru_ts4.06.1901.2021.pre.dat.nc", DATASETS_DIR).open("pre").slice(Region(
    26, 70, -26, 40, Month.FEB, Month.APR, 1970, 2000
))

yp = Preprocess(y)
zp = Preprocess(z)


DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")

MCA_PREFIX = f"mca_{os.path.splitext(os.path.basename(__file__))[0]}_" 
mca = MCA(yp, zp, nm=3, alpha=.1)
mca.save(prefix=MCA_PREFIX, folder=DATA_FOLDER) 
# mca = MCA.load(MCA_PREFIX, DATA_FOLDER, dsy=yp, dsz=zp)

mca.plot(show_plot=True)

CROSS_PREFIX = f"cross_{os.path.splitext(os.path.basename(__file__))[0]}_" 
cross = Crossvalidation(yp, zp, nm=3, alpha=.1, num_svdvals=3)
cross.save(prefix=CROSS_PREFIX, folder=DATA_FOLDER) 
# cross = Crossvalidation.load(CROSS_PREFIX, DATA_FOLDER, dsy=yp, dsz=zp)

cross.plot(show_plot=True)
cross.plot(show_plot=True, halt_program=True, 
           version=2, mca=mca)
