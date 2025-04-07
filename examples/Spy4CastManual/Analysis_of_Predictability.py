"""
Example to analyse the predictability of a predictand variable (Z) 
against a predictor variable (Y) with different lags

Author: Pablo Duran
"""
import math
import numpy as np
import os
from spy4cast import Dataset, Region, Month, get_central_longitude_from_region, \
    get_xlim_from_region, region2str, season2str, spy4cast, plot_map
import cartopy.crs as ccrs


# Predictand variable is the same in every run:

DATASETS_FOLDER = "./datasets"
Z_DATASET_NAME = "cru_ts4_pre_chopped.nc"
Z_VAR = "pre"
Y_DATASET_NAME = "HadISST_sst_chopped.nc"
Y_VAR = "sst"
year0, yearf = 1971, 2000


filename = os.path.splitext(os.path.basename(__file__))[0] + f"-{year0}_{yearf}"

DATA_FOLDER = f"data-{filename}"
PLOTS_FOLDER = f"plots-{filename}"

# 1. Preprocess and Calculate Z variable

z_ds = Dataset(Z_DATASET_NAME, folder=DATASETS_FOLDER).open(Z_VAR)
z_region = Region(
    lat0=0, latf=25, lon0=-20, lonf=20,
    month0=Month.JUL, monthf=Month.SEP,
    year0=year0, yearf=yearf,
)
z_ds.slice(z_region)

z_pre = spy4cast.Preprocess(z_ds)

print(f"[INFO] Z({Z_VAR}): {region2str(z_region)} ({z_region.year0} -> {z_region.yearf})")

nm = 1
alpha = 0.1

max_lag = 16
min_lag = 0
lags = np.arange(max_lag, min_lag - 1, -1)
seasons = np.zeros(lags.shape, dtype=object)
regions = np.zeros(lags.shape, dtype=object)

# 2. Calculate Y variables for different lags

for i in range(len(lags)):
    lag = lags[i]

    month0_i = z_region.month0 - lag
    monthf_i = z_region.monthf - lag
    year0 = z_region.year0
    yearf = z_region.yearf

    while month0_i <= 0:
        month0_i += 12

    while monthf_i <= 0:
        monthf_i += 12
        # Years refer to monthf
        yearf -= 1
        year0 -= 1

    month0, monthf = Month(month0_i), Month(monthf_i)

    y_region = Region(
        lat0=-30, latf=30, lon0=-200, lonf=-60,
        month0=month0, monthf=monthf, year0=year0, yearf=yearf,
    )

    print(f"[INFO] [lag={lag} month{'s' if lag > 1 else ''}] Y({Y_VAR}): {region2str(y_region)} ({year0} -> {yearf})")

    mca_prefix = f"mca_{lag}_"
    if not os.path.exists(os.path.join(DATA_FOLDER, mca_prefix+"RUY.npy")):
        y_ds = Dataset(Y_DATASET_NAME, folder=DATASETS_FOLDER).open(Y_VAR).slice(y_region)
        y_pre = spy4cast.Preprocess(y_ds)
        mca = spy4cast.MCA(y_pre, z_pre, nm=nm, alpha=alpha)
        mca.save(mca_prefix, folder=DATA_FOLDER)

    # Crossvalidation for all lags takes around 10 min but you only need to run it once 
    # because we can load the data aftewards
    cross_prefix = f"cross_{lag}_"
    if not os.path.exists(os.path.join(DATA_FOLDER, cross_prefix+"r_uv.npy")):
        y_ds = Dataset(Y_DATASET_NAME, folder=DATASETS_FOLDER).open(Y_VAR).slice(y_region)
        y_pre = spy4cast.Preprocess(y_ds)

        # num_svdvals=1 to only calculate the first singular value (scf will not be precise but it will run faster)
        cross = spy4cast.Crossvalidation(y_pre, z_pre, nm=nm, alpha=alpha, num_svdvals=nm)
        cross.save(cross_prefix, folder=DATA_FOLDER)

    regions[i] = y_region
    seasons[i] = season2str(y_region.month0, y_region.monthf)


# 3. Create figures

import matplotlib.pyplot as plt
from matplotlib import gridspec

n_figures = 8
indices = list(range(len(lags) - 1, len(lags) - n_figures - 1, -1))

fig = plt.figure(figsize=(9, 10))
ncols = 4
n_runs = len(indices)
nrows = math.ceil(n_runs/ncols)
height_ratios = [1 for _ in range(nrows)] + [.05]
gs = gridspec.GridSpec(nrows + 1, 1, wspace=0, hspace=.1, height_ratios=height_ratios)
y_pre = None
for row in range(nrows):
    ncols_i = min(n_runs - row * ncols, ncols)
    subgs = gs[row].subgridspec(1, ncols_i, wspace=0.1, hspace=0)
    for col in range(ncols_i):
        i = indices[row * ncols + col]
        subsubgs = subgs[col].subgridspec(2, 1, height_ratios=[1, 2], hspace=.1, wspace=0)

        lag = lags[i]
        y_region = regions[i]
        
        if y_pre is None:  # We dont care about the variables, just the shape of lat and lon
            y_ds = Dataset(Y_DATASET_NAME, folder=DATASETS_FOLDER).open(Y_VAR).slice(y_region)
            y_pre = spy4cast.Preprocess(y_ds)

        cross = spy4cast.Crossvalidation.load(f"cross_{lag}_", folder=DATA_FOLDER, dsy=y_pre, dsz=z_pre)
        skill = cross.r_z_zhat_s_separated_modes[0].copy()
        pval = cross.p_z_zhat_s_separated_modes[0].copy()

        c_lon_z = get_central_longitude_from_region(z_region.lon0, z_region.lonf)
        z_xlim = get_xlim_from_region(z_region.lon0, z_region.lonf, c_lon_z)
        ax0 = fig.add_subplot(subsubgs[0], projection=ccrs.PlateCarree(central_longitude=c_lon_z))
        lats, lons = cross.dsz.lat, cross.dsz.lon
        sig_and_pos = (pval <= cross.alpha) & (skill >= 0)
        skill[~sig_and_pos] = np.nan
        t = skill.reshape((len(lats), len(lons)))
        add_cyclic_point = z_region.lon0 >= z_region.lonf
        im0 = plot_map(
            t, lats, lons, fig, ax0, levels=np.arange(0, 1.01, .1),
            cmap="magma", labels=False,
            colorbar=False, add_cyclic_point=add_cyclic_point, xlim=z_xlim
        )
        # ax0.contourf(
        #     lons, lats, np.where(sig_and_pos.reshape((len(lats), len(lons))), t, np.nan), 
        #     colors='none', hatches='xx', extend='both',
        #     transform=ccrs.PlateCarree()
        # )
        ax0.set_title(f"lag = ${lag}$", fontweight="bold")

        sub2 = subsubgs[1].subgridspec(2, 1, wspace=0, hspace=0.02)

        mca = spy4cast.MCA.load(f"mca_{lag}_", folder=DATA_FOLDER, dsy=y_pre, dsz=z_pre)
        ruy = mca.RUY_sig[:, 0]
        ruz = mca.RUZ_sig[:, 0]
        ruy_mean = np.nanmean(ruy)
        sign = -1 if ruy_mean < 0 else 1

        c_lon_z = get_central_longitude_from_region(z_region.lon0, z_region.lonf)
        z_xlim = get_xlim_from_region(z_region.lon0, z_region.lonf, c_lon_z)
        ax1 = fig.add_subplot(sub2[0], projection=ccrs.PlateCarree(central_longitude=c_lon_z))
        lats, lons = mca.dsz.lat, mca.dsz.lon
        t = ruz.reshape((len(lats), len(lons))) * sign
        add_cyclic_point = z_region.lon0 >= z_region.lonf
        im1 = plot_map(
            t, lats, lons, fig, ax1, levels=np.arange(-1, 1.01, .1),
            cmap="BrBG", labels=False,
            colorbar=False, add_cyclic_point=add_cyclic_point, xlim=z_xlim
        )
        ax1.set_title(f"$scf = {mca.scf[0]*100:.01f}\\%$")
        
        c_lon_y = get_central_longitude_from_region(y_region.lon0, y_region.lonf)
        y_xlim = get_xlim_from_region(y_region.lon0, y_region.lonf, c_lon_y)
        ax2 = fig.add_subplot(sub2[1], projection=ccrs.PlateCarree(central_longitude=c_lon_y))
        lats, lons = mca.dsy.lat, mca.dsy.lon
        t = ruy.reshape((len(lats), len(lons))) * sign
        add_cyclic_point = y_region.lon0 >= y_region.lonf
        im2 = plot_map(
            t, lats, lons, fig, ax2, levels=np.arange(-1, 1.01, .1),
            cmap="bwr", labels=False,
            colorbar=False, add_cyclic_point=add_cyclic_point, xlim=y_xlim
        )

        if row == 0 and col == 0:
            sgs = gs[nrows].subgridspec(1, 3, wspace=.1)
            cax0 = fig.add_subplot(sgs[0])
            cb0 = fig.colorbar(im0, cax0, orientation="horizontal")
            cb0.ax.tick_params(rotation=30)
            cax1 = fig.add_subplot(sgs[1])
            cb1 = fig.colorbar(im1, cax1, orientation="horizontal")
            cb1.ax.tick_params(rotation=30)
            cax2 = fig.add_subplot(sgs[2])
            cb2 = fig.colorbar(im2, cax2, orientation="horizontal")
            cb2.ax.tick_params(rotation=30)

fig.suptitle("Analysis of skill for different lags", fontweight="bold")

if not os.path.exists(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)

fig.savefig(os.path.join(PLOTS_FOLDER, "map_skill.svg"), bbox_inches="tight")
fig.savefig(os.path.join(PLOTS_FOLDER, "map_skill.png"), bbox_inches="tight")

mean_skills = np.zeros(lags.shape, dtype=np.float32)
for i in range(len(lags)):
    lag = lags[i]
    y_region = regions[i]
    
    if y_pre is None:
        y_ds = Dataset(Y_DATASET_NAME, folder=DATASETS_FOLDER).open(Y_VAR).slice(y_region)
        y_pre = spy4cast.Preprocess(y_ds)

    cross = spy4cast.Crossvalidation.load(f"cross_{lag}_", folder=DATA_FOLDER, dsy=y_pre, dsz=z_pre)
    skill = cross.r_z_zhat_s_separated_modes[0].copy()
    pval = cross.p_z_zhat_s_separated_modes[0].copy()
    sig_and_pos = (pval <= cross.alpha) & (skill >= 0)
    skill_mean = np.sum(skill[sig_and_pos])

    mean_skills[i] = skill_mean

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot()
ax.plot(lags, mean_skills, color="black", alpha=0.5)
ax.plot(lags, mean_skills, ".", color="black")

ax.set_xlabel("lag (months)")
ax.set_ylabel("Total skill")
ax.xaxis.set_ticks(lags)
ax.xaxis.set_ticklabels(
    [f"{lags[i]} ({seasons[i]})" for i in range(len(lags))],
    rotation=30,
)

ax.set_title("Analysis of skill for different lags")
ax.grid()
        
fig.savefig(os.path.join(PLOTS_FOLDER, "skill_lag.svg"), bbox_inches="tight")
fig.savefig(os.path.join(PLOTS_FOLDER, "skill_lag.png"), bbox_inches="tight")

plt.show()
