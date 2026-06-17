# %%
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm, ListedColormap
import datetime as dt
import cartopy.feature as cfeature
from matplotlib import cm
from matplotlib import colors as clrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.colors as mcolors
import os
import pandas as pd
import metview as mv
import glob
import numpy.ma as ma
import warnings
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def read_caravan_metadata(region_name):
      if region_name == 'Iceland':
        file_name = '/perm/paga/data_observations/caravan/attributes/lamahice/attributes_other_lamahice.csv'
      elif region_name == 'Alps':
        file_name = '/perm/paga/data_observations/caravan/attributes/lamah/attributes_other_lamah.csv'
      df = pd.read_csv(file_name)

      return df['gauge_lat'], df['gauge_lon'], df['area'],df['gauge_id']

def read_caravan_obsdis(region_name,area,fname,iniD,endD):
      if region_name == 'Iceland':
        file_name = f'/perm/paga/data_observations/lamah_ice/Caravan_extension_lamahice/timeseries/csv/lamahice/{fname}.csv'
      elif region_name == 'Alps':
        file_name = f'/perm/paga/data_observations/caravan/timeseries/csv/lamah/{fname}.csv'
      df = pd.read_csv(file_name)
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
      #df = df[(df['date'] >= iniD) & (df['date'] <= endD)]
      dis_time=df['date'][(df['date'] >= iniD) & (df['date'] <= endD)].values
      dis=df['streamflow'][(df['date'] >= iniD) & (df['date'] <= endD)].values
      dis=dis*area/86.4 # mm/day to m3/s

      return dis,dis_time

# Global definitions:
expids=['ij8z',
       'ilhe',
       ]

datadir='/perm/paga/glacier_discharge/'
ctlDir_y='/perm/paga/glacier_discharge/ctl/'
expDir_y='/perm/paga/glacier_discharge/glacier/'
camaRes_y='15min'

# %%
iniD='19900101'
iniD_p1 = (pd.to_datetime(iniD, format='%Y%m%d') + pd.DateOffset(days=1)).strftime('%Y%m%d')
endD='20101231'
# Read the data or download if not present
data=dict()
# Open a CDO instance
fDir='/perm/paga/glacier_discharge/'
storeDir='/perm/paga/glacier_discharge/'

# %%
for ee in expids:
  data[ee] = xr.open_dataset(f'{fDir}/dis_{ee}_{iniD}-{endD}.nc')

# %%
obs=xr.open_dataset(f"/ec/vol/destine-hydro/OBSERVATIONS/station_attributes_with_obsdis24h_197001-202312_v4.0_20231103_withEFAS.nc")
obs_metadata=pd.read_csv('/ec/vol/destine-hydro/OBSERVATIONS/outlets_v4.0_20231103_withEFAS.csv')

glofas=xr.open_zarr(f'/perm/paga/data_observations/glofas/glofas_1990-2010_Himalaya.zarr')

# grdc mask
grdc_mask = (obs_metadata['Provider_1'] == 'GRDC') | (obs_metadata['Provider_2'] == 'GRDC')
# Get the indices corresponding to this condition
grdc_indices = obs_metadata[grdc_mask].index

obs_red=obs['obsdis'].sel(time=slice(iniD_p1,endD))

data_red=dict()
for ee in expids:
  data[ee]=data[ee].drop_duplicates(dim='time')
  data_red[ee]=data[ee] #.sel(time=slice(iniD,endD)).resample(time='D').mean()


obs_camaLat = obs['CamaLat1_' + camaRes_y][:].values
obs_camaLon = obs['CamaLon1_' + camaRes_y][:].values
glofas_lat=obs['lisfloodY_3min'].values
glofas_lon=obs['lisfloodX_3min'].values

obs_discharge = obs_red[:]  # e.g., observed discharge
obs_rivername=obs['rivername'][:].values
obs_stationame=obs['stationname'][:].values

model_discharge=dict()
for ee in expids:
   model_discharge[ee] = data_red[ee]['dis'][:]  # e.g., model discharge

regions = {
    "Alps": {"lat_min": 43.0, "lat_max": 48.0, "lon_min": 5.0, "lon_max": 15.0},
    "Iceland": {"lat_min": 62.0, "lat_max": 68.0, "lon_min": -26.0, "lon_max": -12.0},
    "Himalaya": {"lat_min": 20.0, "lat_max": 35.0, "lon_min": 70.0, "lon_max": 92.0},
    "Alaska": {"lat_min": 53.0, "lat_max": 73.0, "lon_min": -170.0, "lon_max": -120.0}
}

lats_to_plot = []
lons_to_plot = []
kges_to_plot = []
kge_ctl = {}
kge_exp = {}
corr_ctl = {}
corr_exp = {}
obs_mean = {}
mean_ctl = {}
mean_exp = {}
std_exp = {}
std_ctl = {}
obs_std = {}

# Loop over regions, find indices, compute KGE for each station
kge_region=dict()
for region_name, bbox in regions.items():
#for region_name, bbox in list(regions.items())[:1]:
    print(region_name)
    lat_min = bbox["lat_min"]
    lat_max = bbox["lat_max"]
    lon_min = bbox["lon_min"]
    lon_max = bbox["lon_max"]

    # Read dataframe from tab delimited file
    if region_name == "Alps":
      file_name = f"{region_name}_kge_{expids[1]}-ctl_v2.txt"
    else:
      file_name = f"{region_name}_kge_{expids[1]}-ctl.txt"
    df = pd.read_csv(file_name, sep='\t')

    region_idx = df['station_idx'].values

    kge_region[region_name]=list()

    kge_ctl[region_name]=np.zeros(len(region_idx))
    kge_exp[region_name]=np.zeros(len(region_idx))
    corr_ctl[region_name]=np.zeros(len(region_idx))
    corr_exp[region_name]=np.zeros(len(region_idx))
    obs_mean[region_name]=np.zeros(len(region_idx))
    mean_ctl[region_name]=np.zeros(len(region_idx))
    mean_exp[region_name]=np.zeros(len(region_idx))
    std_ctl[region_name]=np.zeros(len(region_idx))
    std_exp[region_name]=np.zeros(len(region_idx))
    obs_std[region_name]=np.zeros(len(region_idx))

    obs_Lat=df['obs_lat'].values
    obs_Lon=df['obs_lon'].values
    station_idx=df['station_idx'].values
    # Compute KGE for each station in this region
    for i in region_idx:
        # IF in GRDC list then use the GRDC data, otherwise look in CARAVAN dataset:
        #**if i in grdc_indices:
        #**  obs_nn=obs_red.isel(station=i)
        #**elif region_name in ['Iceland', 'Alps']:
        #**  caravan_lat,caravan_lon,caravan_area,caravan_name = read_caravan_metadata(region_name)
        #**  caravan_idx=df['caravan_idx'][station_idx==i].values[0]
        #**  obs_nn   = read_caravan_obsdis(region_name, caravan_area[caravan_idx],caravan_name[caravan_idx],iniD_p1, endD)
        #**elif region_name in ['Himalaya']:
        #**  obs_nn_tmp=obs_red.isel(station=i)
        #**  nmiss=np.sum(np.isnan(obs_nn_tmp))
        #**  if (nmiss/len(obs_nn_tmp)<0.50):
        #**      print('glofas, only for points where we have observations.')
        #**      obs_nn=glofas['dis24'].sel(lat=glofas_lat[i], lon=glofas_lon[i], method='nearest')[:-1]
        #**      # Convert to numpy array
        #**      obs_nn=obs_nn.compute()
        #**  else:
        #**      continue
        #**else:
        #**    continue

        #**nmiss=np.sum(np.isnan(obs_nn))
        #**if (nmiss/len(obs_nn)<0.50):
          lat_i = obs_camaLat[i]
          lon_i = obs_camaLon[i]

          lats_to_plot.append(lat_i)
          lons_to_plot.append(lon_i)

    kges_to_plot.extend(df["KGE_EXP-KGE_CTL"].values)
    kge_ctl[region_name]=df["KGE_CTL"].values
    kge_exp[region_name]=df["KGE_EXP"].values
    corr_ctl[region_name]=df["CORR_CTL"].values
    corr_exp[region_name]=df["CORR_EXP"].values
    obs_mean[region_name]=df["OBS_MEAN"].values
    mean_ctl[region_name]=df["MEAN_CTL"].values
    mean_exp[region_name]=df["MEAN_EXP"].values
    obs_std[region_name]=df["OBS_STD"].values
    std_ctl[region_name]=df["STD_CTL"].values
    std_exp[region_name]=df["STD_EXP"].values

lats_to_plot = np.array(lats_to_plot)
lons_to_plot = np.array(lons_to_plot)
kges_to_plot = np.array(kges_to_plot)


# Plot on a global map with Cartopy
#*# 1) Define discrete boundaries and a matching colormap
min_val_plot=-0.02
max_val_plot=0.02
levels = [-0.50, -0.25, -0.10, -0.05, min_val_plot, max_val_plot, 0.05, 0.10, 0.25, 0.50]
cmap_list = [
    "darkblue", "blue", "deepskyblue",    # lower KGE < -0.05
    "lightblue",                           # up to -0.05
    "white",                               # -0.05 to 0.05
    "moccasin", "orange", "crimson", "darkred"  # higher KGE > 0.05
]
cmap = mcolors.ListedColormap(cmap_list)
norm = mcolors.BoundaryNorm(levels, cmap.N)

# 2) Create an alpha array that is zero for |KGE| ≤ 0.05, otherwise depends on |KGE|
alpha_array =np.clip(np.abs(kges_to_plot), 0, 1)
mask_zero = (kges_to_plot >= min_val_plot) & (kges_to_plot <= max_val_plot)
alpha_array[mask_zero] = 0.0  # fully transparent if -0.05 ≤ KGE ≤ 0.05
alpha_array[~mask_zero] = 0.80  # fully transparent if -0.05 ≤ KGE ≤ 0.05

# 3) Create a size array that increases with |KGE|
size_array = 5 + 1.0 * np.abs(kges_to_plot)
#

#*fig = plt.figure(figsize=(10, 6))
#*ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent([-180, 105, 10, 80], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.75)
ax.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.3)

sc = ax.scatter(
    lons_to_plot, lats_to_plot,
    c=kges_to_plot,
    cmap=cmap,
    norm=norm,
    s=size_array,
    alpha=alpha_array,
    transform=ccrs.PlateCarree()
)

for region, coords in regions.items():
    # Enlarge the boxes by 1 degree
    lat_min, lat_max = coords["lat_min"]-1.0, coords["lat_max"]+1.0
    lon_min, lon_max = coords["lon_min"]-1.0, coords["lon_max"]+1.0
    if region in ['Himalaya']:
       colorPatch='blue'
    else:
       colorPatch='red'
    ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                               edgecolor=colorPatch, facecolor='none', linewidth=1.0, transform=ccrs.PlateCarree()))

    # Add tables with values of kge_ctl, kge_exp, corr_ctl, corr_exp, bias_score_ctl, and bias_score_exp
    for region_name, bbox in regions.items():
      lat_min = bbox["lat_min"]
      lat_max = bbox["lat_max"]
      lon_min = bbox["lon_min"]
      lon_max = bbox["lon_max"]

      region_idx = np.where(
        (obs_camaLat >= lat_min) & (obs_camaLat <= lat_max) &
        (obs_camaLon >= lon_min) & (obs_camaLon <= lon_max)
      )[0]

      bias_score_ctl=mean_ctl[region_name]/obs_mean[region_name] - 1.0
      bias_score_exp=mean_exp[region_name]/obs_mean[region_name] - 1.0
      std_score_ctl=(std_ctl[region_name]/mean_ctl[region_name])/(obs_std[region_name]/obs_mean[region_name]) - 1.0
      std_score_exp=(std_exp[region_name]/mean_exp[region_name])/(obs_std[region_name]/obs_mean[region_name]) - 1.0
      if len(region_idx) > 0:
        mean_kge_ctl = np.mean(kge_ctl[region_name][:])
        mean_kge_exp = np.mean(kge_exp[region_name][:])
        mean_corr_ctl = np.mean(corr_ctl[region_name][:])
        mean_corr_exp = np.mean(corr_exp[region_name][:])
        mean_bias_ctl = np.mean(bias_score_ctl[:])
        mean_bias_exp = np.mean(bias_score_exp[:])
        mean_std_ctl = np.mean(std_score_ctl[:])
        mean_std_exp = np.mean(std_score_exp[:])

        # Decide text position per region
        if region_name == "Alaska":
            # Underneath the box
            x_text = lon_min
            y_text = lat_min - 4.5
        elif region_name == "Iceland":
            # On the left of the box
            x_text = lon_min - 20.0
            y_text = lat_min - 4.0
        elif region_name == "Alps":
            # On the bottom of the box
            x_text = lon_min
            y_text = lat_min - 4.5
        elif region_name == "Himalaya":
            # On the right of the box
            x_text = lon_min - 5
            y_text = lat_max + 35
        else:
            # Default to just below the box
            x_text = lon_min
            y_text = lat_min - 1.5

       #* table_text = (
       #*     f"CTL Corr: {{CTL_COLOR}}{mean_corr_ctl:.2f}{{ENDC}}\n"
       #*     f"GLA Corr: {{GLA_COLOR}}{mean_corr_exp:.2f}{{ENDC}}\n"
       #*     f"CTL B-score: {{CTL_COLOR}}{mean_bias_ctl:.2f}{{ENDC}}\n"
       #*     f"GLA B-score: {{GLA_COLOR}}{mean_bias_exp:.2f}{{ENDC}}\n"
       #*     f"CTL S-score: {{CTL_COLOR}}{mean_std_ctl:.2f}{{ENDC}}\n"
       #*     f"GLA S-score: {{GLA_COLOR}}{mean_std_exp:.2f}{{ENDC}}"
       #* )
       #* # Replace color placeholders with HTML color tags for matplotlib
       #* table_text = (
       #*     table_text
       #*     .replace("{CTL_COLOR}", r"$\color{red}{")
       #*     .replace("{GLA_COLOR}", r"$\color{blue}{")
       #*     .replace("{ENDC}", "}$")
       #* )
       #* # Draw text with a white background
       #* ax.text(
       #*     x_text,
       #*     y_text,
       #*     table_text,
       #*     fontsize=8,
       #*     verticalalignment='top',
       #*     transform=ccrs.PlateCarree(),
       #*     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
       #*     usetex=False,
       #*     wrap=True
       #* )

       # Build & draw multi‑line colored stats (avoid mathtext \color that caused error)
        lines = [
            ("CTL KGE:",  mean_kge_ctl, 'red'),
            ("GLA KGE:",  mean_kge_exp, 'blue'),
            ("CTL Corr:",  mean_corr_ctl, 'red'),
            ("GLA Corr:",  mean_corr_exp, 'blue'),
            ("CTL B-score:", mean_bias_ctl, 'red'),
            ("GLA B-score:", mean_bias_exp, 'blue'),
            ("CTL S-score:", mean_std_ctl, 'red'),
            ("GLA S-score:", mean_std_exp, 'blue'),
        ]
        line_height = 5.00  # vertical spacing in data (lat) units
        pad_x = 0.4
        pad_y = 0.5
        total_height = line_height * (len(lines)-1)
        box_width = 20.0  # widen a bit for readability

        ax.add_patch(plt.Rectangle(
            (x_text - pad_x, y_text - total_height - pad_y),
            box_width,
            total_height + (pad_y * 2) + 0.2,
            facecolor='white', edgecolor='none', alpha=0.85,
            transform=ccrs.PlateCarree(), zorder=2
        ))

        for li, (label, value, color) in enumerate(lines):
            y_line = y_text - li * line_height
            ax.text(
                x_text,
                y_line,
                f"{label} {value:.2f}",
                fontsize=9,
                color=color,
                va='top',
                ha='left',
                transform=ccrs.PlateCarree(),
                zorder=3
            )

plt.colorbar(sc, ax=ax, label='ΔKGE (GLA - CTL)', shrink=0.4)
plt.tight_layout()
plt.savefig("kge_map_regions_discharge_v2.svg", format="svg")
plt.savefig('kge_map_regions_discharge_v2.pdf')

# Define the list of river names to plot
river_names_to_plot = ["JOEKULSA I FLJOTSDAL", "Copper", "Karnali", "Isel"]
station_names_to_plot = ["HOLL", "Million Dollar Bridge Near Cordova Ak", "Asaraghat", "Lienz"]
river_plot_name={"JOEKULSA I FLJOTSDAL":"Joekulsa i Fljotsdal, Iceland",
                 "Copper":"Copper, Alaska",
                 "Isel":"Isel, Alps",
                 "Karnali":"Karnali, Himalaya",}
fig, axes = plt.subplots(2, 2, figsize=(8, 5))
axes = axes.flatten()

for ax, river_name in zip(axes, river_names_to_plot):
  print(river_name)
  river_idx = np.where(obs_rivername == river_name)[0]
  if len(river_idx) == 0:
    raise ValueError(f"No data found for river: {river_name}")

  ts_idx = None
  for idx in river_idx:
    if any(pattern in obs_stationame[idx] for pattern in station_names_to_plot):
      ts_idx = idx
      break
    else:
       if river_name == 'Copper':
         ts_idx=region_idx[1]
       elif river_name == 'Karnali':
         ts_idx=region_idx[40]
       elif river_name == 'Isel':
         ts_idx=region_idx[1]
       else:
         raise ValueError(f"No matching station found for river: {river_name}")

  if ts_idx is None:
    raise ValueError(f"No matching station found for river: {river_name}")
 #*"JOEKULSA I FLJOTSDAL":"Joekulsa i Fljotsdal, Iceland",
 #*                "Copper":"Copper, Alaska",
 #*                "Karnali":"Karnali, Himalaya",
 #*                "Isel":"Isel, Alps
  if river_name == 'JOEKULSA I FLJOTSDAL':
    iniD='19990101'
    endD='20011231'
    caravan_lat,caravan_lon,caravan_area,caravan_name = read_caravan_metadata("Iceland")
    caravan_idx=40 #station_idx[station_idx==ts_idx] #[0]
    obs_ts,obs_time = read_caravan_obsdis("Iceland", caravan_area[caravan_idx],caravan_name[caravan_idx],iniD, endD)
    xticks = [np.datetime64('1999-01'), np.datetime64('2000-01'), np.datetime64('2001-01'), np.datetime64('2002-01')]
    xticklabels = ['1999-01', '2000-01', '2001-01', '2002-01']
  elif river_name=='Isel':
    iniD='20000101'
    endD='20021231'
    caravan_lat,caravan_lon,caravan_area,caravan_name = read_caravan_metadata("Alps")
    caravan_idx=664 #station_idx[station_idx==ts_idx] #[0]
    obs_ts,obs_time = read_caravan_obsdis("Alps", caravan_area[caravan_idx],caravan_name[caravan_idx],iniD, endD)
    xticks = [np.datetime64('2000-01'), np.datetime64('2001-01'), np.datetime64('2002-01'), np.datetime64('2003-01')]
    xticklabels = ['2000-01', '2001-01', '2002-01', '2003-01']
  elif river_name=='Copper':
    iniD='20050101'
    endD='20071231'
    obs_ts=obs_red.isel(station=ts_idx).sel(time=slice(iniD, endD)).values
    obs_time=obs_red['time'].sel(time=slice(iniD, endD)).values
    xticks = [np.datetime64('2005-01'), np.datetime64('2006-01'), np.datetime64('2007-01'), np.datetime64('2008-01')]
    xticklabels = ['2005-01', '2006-01', '2007-01', '2008-01']
  elif river_name=='Karnali':
    iniD='20050101'
    endD='20071231'
    obs_ts=glofas['dis24'].sel(time=slice(iniD, endD)).sel(lat=glofas_lat[ts_idx], lon=glofas_lon[ts_idx], method='nearest')
    # Convert to numpy array
    obs_ts=obs_ts.compute()
    obs_time=glofas['time'].sel(time=slice(iniD, endD)).compute().values
    xticks = [np.datetime64('2005-01'), np.datetime64('2006-01'), np.datetime64('2007-01'), np.datetime64('2008-01')]
    xticklabels = ['2005-01', '2006-01', '2007-01', '2008-01']

  lat_i = obs_camaLat[ts_idx]
  lon_i = obs_camaLon[ts_idx]
  ctl_nn=data_red[expids[0]]['dis'].sel(lat=lat_i, lon=lon_i, method='nearest').sel(time=slice(iniD, endD))
  exp_nn=data_red[expids[1]]['dis'].sel(lat=lat_i, lon=lon_i, method='nearest').sel(time=slice(iniD, endD))

  ax.plot(obs_time, obs_ts, color='black', label='Observations')
  ax.plot(ctl_nn.time.values, ctl_nn.values, color='red', label='CTL',alpha=0.6, ls='--')
  ax.plot(ctl_nn.time.values, exp_nn.values, color='blue', label='GLA',alpha=0.6, ls='--')

  ax.set_title(river_plot_name[river_name])
  ax.set_xlabel('Time')
  ax.set_ylabel('Discharge (m³/s)')
  # Set x-ticks and x-labels at 1999-01, 2000-01, 2001-01, 2002-01
  ax.set_xticks(xticks)
  ax.set_xticklabels(xticklabels)
  for xtick in xticks:
    ax.axvline(x=xtick, color='black', linewidth=1.2, linestyle='-', ymin=0, ymax=0.05, zorder=5)

  # Add small horizontal lines at the position of the yticks
  ax.set_ylim(bottom=0)
  for ytick in ax.get_yticks():
    ax.axhline(y=ytick, color='black', linewidth=1.2, linestyle='-', xmin=0, xmax=0.025, zorder=5)
  ax.tick_params(axis='x')
  ax.legend()
  ax.spines["bottom"].set_visible(True)
  ax.spines["bottom"].set_color("black")
  ax.spines["left"].set_visible(True)
  ax.spines["left"].set_color("black")

plt.tight_layout()
plt.savefig('ts_discharge_rivers.svg',format='svg')
plt.savefig('ts_discharge_rivers.pdf')
