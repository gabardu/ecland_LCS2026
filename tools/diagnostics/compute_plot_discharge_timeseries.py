# %%
import xarray as xr
from cdo import Cdo
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import metview as mv
import glob
import numpy.ma as ma
import warnings
import dask.array

# Output root for per-station time-series figures
TIMESERIES_DIR = 'figures/timeseries'

def download_offline_data(id, iniD, endD, fout, foutStore):
   """
   Downloads offline data for a given time range and saves it to a file.

   Parameters:
   - id (str): The identifier for the data.
   - iniD (str): The initial date in the format 'YYYYMMDD'.
   - endD (str): The end date in the format 'YYYYMMDD'.
   - fout (str): The output file path.
   - foutStore (str): The store/backup file path.

   Returns:
   - xarray.Dataset: The downloaded data.
   """
   _cdo = Cdo()  # create a local cdo instance – no global dependency
   _cdo.cleanTempDir()

   ini_date = pd.to_datetime(iniD, format='%Y%m%d')
   end_date = pd.to_datetime(endD, format='%Y%m%d')

   date_list = pd.date_range(start=ini_date, end=end_date, freq='MS').strftime('%Y%m%d').tolist()
   tmpDir = '/scratch/paga/tmp_files/'
   if not os.path.exists(tmpDir):
      os.makedirs(tmpDir)

   filename = os.path.basename(fout)
   created_nc_files = []  # track only files created in this run

   for date in date_list:
      print('processing date: ', date)
      date_p1 = pd.to_datetime(date, format='%Y%m%d') + pd.DateOffset(months=1)
      num_days = (date_p1 - pd.to_datetime(date, format='%Y%m%d')).days
      steps = list(np.arange(24, (num_days + 1) * 24, 24))
      tmp = mv.retrieve(
         expver=f"{id}",
         Type="fc",
         class_="rd",
         levtype='sfc',
         stream="oper",
         param="240013",
         date=date,
         step=steps,
         time="00:00:00",
      )
      fin = f'{tmpDir}/{id}_{date}.grib'
      mv.write(f'{fin}', tmp)

      # Convert grib to netcdf and track the output
      nc_file = f'{tmpDir}/{filename}.{date}'
      _cdo.copy(input=f"{fin}", output=nc_file, options='-f nc')
      created_nc_files.append(nc_file)

      # Remove grib immediately to keep tmpDir clean
      os.remove(fin)

   print('Merging files:', created_nc_files)
   _cdo.mergetime(input=' '.join(created_nc_files), output=f'{fout}')
   os.system(f'cp  {fout}  {foutStore}')

   # Remove only the temporary nc files created in this run
   for file in created_nc_files:
      if os.path.exists(file):
         os.remove(file)

   return xr.open_dataset(f'{fout}')  # Return the dataset as an xarray object

def read_caravan_metadata(region_name):
      if region_name == 'Iceland':
        file_name = '/perm/paga/data_observations/caravan/attributes/lamahice/attributes_other_lamahice.csv'
      elif region_name == 'Alps':
        file_name = '/perm/paga/data_observations/caravan/attributes/lamah/attributes_other_lamah.csv'
      df = pd.read_csv(file_name)

      return df['gauge_lat'], df['gauge_lon'], df['area'],df['gauge_id']

def find_caravan_idx(obs_lat, obs_lon, car_lat, car_lon):
        distances = np.sqrt((car_lat - obs_lat)**2 + (car_lon - obs_lon)**2)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        if min_distance < 0.01:
            print(f"Found a caravan station within 0.01 degrees, distance={min_distance}, at idx: {min_distance_idx}")
            return min_distance_idx
        else:
            return np.nan

def read_caravan_obsdis(region_name,idx,area,fname,iniD,endD):
      if region_name == 'Iceland':
        file_name = f'/perm/paga/data_observations/lamah_ice/Caravan_extension_lamahice/timeseries/csv/lamahice/{fname}.csv'
      elif region_name == 'Alps':
        file_name = f'/perm/paga/data_observations/caravan/timeseries/csv/lamah/{fname}.csv'
      df = pd.read_csv(file_name)
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
      #df = df[(df['date'] >= iniD) & (df['date'] <= endD)]
      dis=df['streamflow'][(df['date'] >= iniD) & (df['date'] <= endD)].values
      dis=dis*area/86.4 # mm/day to m3/s

      return dis


def kge(observed, simulated):
    # Mask missing values
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    mask_check=np.sum(mask)
    if (mask_check>1): # There is at least 1 value to make the calculation
      # Apply mask to observed and simulated values
      #if isinstance(mask, dask.array.Array):
      #  mask = mask.compute()  # Ensure the mask is fully evaluated

      observed_masked  = observed[mask]
      simulated_masked = simulated[mask]

       # Check if both arrays have the same length after masking
      if len(observed_masked) != len(simulated_masked):
          raise ValueError("Lengths of observed and simulated arrays after masking are different")

      # Calculate the mean of observed and simulated values
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        obs_mean = np.nanmean(observed_masked)
        sim_mean = np.nanmean(simulated_masked)

      # Calculate the standard deviation of observed and simulated values
      obs_std = np.nanstd(observed_masked)
      sim_std = np.nanstd(simulated_masked)

      # Calculate the correlation coefficient between observed and simulated values
      correlation = ma.corrcoef(ma.masked_invalid(observed_masked), ma.masked_invalid(simulated_masked))[0,1] #np.nancorrcoef(observed_masked, simulated_masked)[0, 1]

      # Calculate the KGE
      kge_value = 1 - np.sqrt((correlation - 1)**2 + ((sim_std/sim_mean) / (obs_std/obs_mean) -1 )**2 + (sim_mean / obs_mean -1)**2)
    else:
      kge_value=np.nan
      correlation=np.nan
      obs_mean=np.nan
      sim_mean=np.nan
      obs_std=np.nan
      sim_std=np.nan

    return kge_value, correlation, obs_mean, sim_mean, obs_std, sim_std


def plot_station_timeseries(region_name, station_idx,
                            lat_i, lon_i,
                            obs_nn, ctl_nn, exp_nn,
                            kge_ctl, kge_exp,
                            iniD_p1, endD,
                            out_root=TIMESERIES_DIR):
    """Plot and save observed vs modelled discharge time series for one station.

    Saves to  <out_root>/<region_name>/station_<station_idx>.png
    """
    out_dir = os.path.join(out_root, region_name)
    os.makedirs(out_dir, exist_ok=True)

    # Build a common daily time axis
    time_index = pd.date_range(start=iniD_p1, end=endD, freq='D')

    def _to_array(x):
        """Convert DataArray or masked/numpy array to a plain float64 1-D array."""
        if hasattr(x, 'values'):
            arr = np.asarray(x.values, dtype=np.float64).ravel()
        else:
            arr = np.asarray(x, dtype=np.float64).ravel()
        return arr

    obs_arr = _to_array(obs_nn)
    ctl_arr = _to_array(ctl_nn)
    exp_arr = _to_array(exp_nn)

    # Align lengths to the shortest of the three + time_index
    n = min(len(time_index), len(obs_arr), len(ctl_arr), len(exp_arr))
    time_index = time_index[:n]
    obs_arr = obs_arr[:n]
    ctl_arr = ctl_arr[:n]
    exp_arr = exp_arr[:n]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(time_index, obs_arr, color='black',    lw=1.2, label='Obs',                  zorder=3)
    ax.plot(time_index, ctl_arr, color='steelblue', lw=0.9, alpha=0.85,
            label=f'CTL ({expids[0]})')
    ax.plot(time_index, exp_arr, color='tomato',    lw=0.9, alpha=0.85,
            label=f'EXP ({expids[1]})')

    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title(
        f'Station {station_idx} | lat={lat_i:.3f}, lon={lon_i:.3f} | {region_name}'
    )
    ax.legend(loc='upper right', fontsize=8)

    # Statistics box
    kge_diff = kge_exp - kge_ctl
    stats_text = (
        f"KGE CTL  = {kge_ctl:.3f}\n"
        f"KGE EXP  = {kge_exp:.3f}\n"
        f"\u0394KGE      = {kge_diff:+.3f}"
    )
    ax.text(
        0.01, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='lightyellow', alpha=0.85,
                  edgecolor='grey')
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'station_{station_idx:06d}.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  Saved {out_path}')
    return out_path


# Experiment definitions (0: ctl, 1: exp):
#expids=['ij8z',
#       'ilhe',
#       ]
expids=['ivkr',
       'ivnl',
       ]

fDir='/perm/paga/glacier_discharge/'
storeDir='/perm/paga/glacier_discharge/'

datadir='/perm/paga/glacier_discharge/'
ctlDir_y='/perm/paga/glacier_discharge/ctl/'
expDir_y='/perm/paga/glacier_discharge/glacier/'
camaRes_y='15min'

# %%
iniD='20251101'
endD='20260531'
iniD_p1 = (pd.to_datetime(iniD, format='%Y%m%d') + pd.DateOffset(days=1)).strftime('%Y%m%d')
# Read the data or download if not present
data=dict()


# %%
for ee in expids:
  fout_ee = f'{fDir}/dis_{ee}_{iniD}-{endD}.nc'
  fstore_ee = f'{storeDir}/dis_{ee}_{iniD}-{endD}.nc'

  try:
      data[ee] = xr.open_dataset(fout_ee)
      print(f'Loaded {fout_ee}')
  except FileNotFoundError:
      print(f'File not found, downloading for expid={ee}...')
      try:
          data[ee] = download_offline_data(ee, iniD, endD, fout_ee, fstore_ee)
          print(f'Download complete for expid={ee}')
      except Exception as download_exc:
          # Download may have created the file even if it raised at the end;
          # try opening it before giving up.
          print(f'Download raised: {download_exc}. Trying to open {fout_ee} anyway...')
          data[ee] = xr.open_dataset(fout_ee)


# 2024 data:
obs=xr.open_dataset(f"/ec/vol/destine-hydro/OBSERVATIONS/station_attributes_with_obsdis24h_197001-202412_v4.0_20241203_withEFAS.nc")
obs_metadata=pd.read_csv('/ec/vol/destine-hydro/OBSERVATIONS/outlets_v4.0_20241203_withEFAS.csv')
glofas=xr.open_zarr(f'/perm/paga/data_observations/glofas/glofas_1990-2010_Himalaya.zarr')

# grdc mask
grdc_mask = (obs_metadata['Provider_1'] == 'GRDC') | (obs_metadata['Provider_2'] == 'GRDC')
# Get the indices corresponding to this condition
grdc_indices = obs_metadata[grdc_mask].index

#obs_red=obs['obsdis'].sel(time=slice(iniD_p1,endD))
#obs=obs.isel(station=grdc_indices)
obs_red = obs['obsdis'].sel(time=slice(iniD_p1, endD))

# Read caravan attribute:

data_red=dict()
for ee in expids:
  data[ee]=data[ee].drop_duplicates(dim='time')
  data_red[ee]=data[ee] #.sel(time=slice(iniD,endD)).resample(time='D').mean()

# %%
obs_camaLat = obs['CamaLat1_' + camaRes_y][:].values
obs_camaLon = obs['CamaLon1_' + camaRes_y][:].values
obs_Lat = obs['lon'][:].values
obs_Lon = obs['lat'][:].values
model_discharge=dict()
for ee in expids:
   model_discharge[ee] = data_red[ee]['dis'][:]  # e.g., model discharge

regions = {
    "NorthHemET": {"lat_min": 30.0, "lat_max": 90.0, "lon_min": -179.0, "lon_max": 179.0},
    #*"Himalaya": {"lat_min": 19.0, "lat_max": 36.0, "lon_min": 67.0, "lon_max": 95.0},
    #*"Alps": {"lat_min": 43.0, "lat_max": 48.0, "lon_min": 5.0, "lon_max": 15.0},
    #*"Iceland": {"lat_min": 62.0, "lat_max": 68.0, "lon_min": -26.0, "lon_max": -12.0},
    #*"Alaska": {"lat_min": 53.0, "lat_max": 73.0, "lon_min": -170.0, "lon_max": -120.0}
}

# Loop over regions, find indices, compute KGE for each station
kge_region=dict()
lat_region=dict()
lon_region=dict()

kge_ctl_region=dict()
kge_exp_region=dict()
corr_ctl_region=dict()
corr_exp_region=dict()
obs_mean_region=dict()
mean_ctl_region=dict()
mean_exp_region=dict()
obs_std_region=dict()
std_ctl_region=dict()
std_exp_region=dict()

obs_lat_region=dict()
obs_lon_region=dict()
statid_region=dict()
caridx_region=dict()
for region_name, bbox in regions.items():
#for region_name, bbox in list(regions.items())[:1]:
    print(region_name)
    lat_min = bbox["lat_min"]
    lat_max = bbox["lat_max"]
    lon_min = bbox["lon_min"]
    lon_max = bbox["lon_max"]

    region_idx = np.where(
        (obs_camaLat >= lat_min) & (obs_camaLat <= lat_max) &
        (obs_camaLon >= lon_min) & (obs_camaLon <= lon_max)
    )[0]
    kge_region[region_name]=list()
    lat_region[region_name]=list()
    lon_region[region_name]=list()
    kge_ctl_region[region_name]=list()
    kge_exp_region[region_name]=list()
    corr_ctl_region[region_name]=list()
    corr_exp_region[region_name]=list()
    obs_mean_region[region_name]=list()
    mean_ctl_region[region_name]=list()
    mean_exp_region[region_name]=list()
    obs_std_region[region_name]=list()
    std_ctl_region[region_name]=list()
    std_exp_region[region_name]=list()
    obs_lat_region[region_name]=list()
    obs_lon_region[region_name]=list()
    statid_region[region_name]=list()
    caridx_region[region_name]=list()
    # Compute KGE for each station in this region
    for i in region_idx:
        print(i)
        #obs_grdc = obs_red[:]  # e.g., observed discharge
        #obs=obs.isel(station=grdc_indices)
        # IF in GRDC list then use the GRDC data, otherwise look in CARAVAN dataset:
        if i in grdc_indices:
          print(f'GRDC station: {i}')
          caravan_idx_value = np.nan  # default is NaN
          obs_nn=obs_red.isel(station=i)
        elif region_name in ['Iceland', 'Alps']:
          caravan_idx_value = np.nan  # default is NaN
          caravan_lat,caravan_lon,caravan_area,caravan_name = read_caravan_metadata(region_name)
          caravan_idx = find_caravan_idx(obs_Lat[i], obs_Lon[i], caravan_lat, caravan_lon)
          if np.isnan(caravan_idx):
            continue
          else:
            obs_nn   = read_caravan_obsdis(region_name, caravan_idx, caravan_area[caravan_idx],caravan_name[caravan_idx],iniD_p1, endD)
            caravan_idx_value = caravan_idx
        elif region_name in ['Himalaya']:
          obs_nn_tmp=obs_red.isel(station=i)
          nmiss=np.sum(np.isnan(obs_nn_tmp))
          if (nmiss/len(obs_nn_tmp)<0.50):
              print('glofas, only for points where we have observations.')
              glofas_lat=obs['lisfloodY_3min'].values
              glofas_lon=obs['lisfloodX_3min'].values
              obs_nn=glofas['dis24'].sel(lat=glofas_lat[i], lon=glofas_lon[i], method='nearest')[:-1]
              # Convert to numpy array
              obs_nn=obs_nn.compute()
          else:
              continue
        else:
            caravan_idx_value = np.nan
            continue

        nmiss=np.sum(np.isnan(obs_nn))
        if (nmiss/len(obs_nn)<0.50):
          print(f'Computing KGE for station {i} in region {region_name}')
          lat_i = obs_camaLat[i]
          lon_i = obs_camaLon[i]
          ctl_nn=data_red[expids[0]]['dis'].sel(lat=lat_i, lon=lon_i, method='nearest')
          exp_nn=data_red[expids[1]]['dis'].sel(lat=lat_i, lon=lon_i, method='nearest')
          kge_ctl, corr_ctl, obs_mean, mean_ctl, obs_std, std_ctl =kge(obs_nn[:], ctl_nn[:-1])
          kge_exp, corr_exp, obs_mean, mean_exp, obs_std, std_exp =kge(obs_nn[:], exp_nn[:-1])
          kge_val = kge_exp - kge_ctl # KGE(EXP) - KGE(CTL)

          # Plot time series for this station
          plot_station_timeseries(
              region_name, i, lat_i, lon_i,
              obs_nn, ctl_nn[:-1], exp_nn[:-1],
              kge_ctl, kge_exp,
              iniD_p1, endD,
          )

          # Save in dictionary
          kge_region[region_name].append(kge_val)
          lat_region[region_name].append(lat_i)
          lon_region[region_name].append(lon_i)

          obs_lat_region[region_name].append(obs_Lat[i])
          obs_lon_region[region_name].append(obs_Lon[i])
          statid_region[region_name].append(i)
          caridx_region[region_name].append(caravan_idx_value)
          # additional scores
          kge_ctl_region[region_name].append(kge_ctl)
          kge_exp_region[region_name].append(kge_exp)
          corr_ctl_region[region_name].append(corr_ctl)
          corr_exp_region[region_name].append(corr_exp)
          obs_mean_region[region_name].append(obs_mean)
          mean_ctl_region[region_name].append(mean_ctl)
          mean_exp_region[region_name].append(mean_exp)
          obs_std_region[region_name].append(obs_std)
          std_ctl_region[region_name].append(std_ctl)
          std_exp_region[region_name].append(std_exp)

for region_name in kge_region.keys():
  # Convert the lists of lat, lon, and KGE values to a DataFrame
  df = pd.DataFrame({
    "station_idx": statid_region[region_name],
    "caravan_idx": caridx_region[region_name],
    "lat": lat_region[region_name],
    "lon": lon_region[region_name],
    "obs_lat": obs_lat_region[region_name],
    "obs_lon": obs_lon_region[region_name],
    "KGE_EXP-KGE_CTL": kge_region[region_name],
    "KGE_CTL": kge_ctl_region[region_name],
    "KGE_EXP": kge_exp_region[region_name],
    "CORR_CTL": corr_ctl_region[region_name],
    "CORR_EXP": corr_exp_region[region_name],
    "OBS_MEAN": obs_mean_region[region_name],
    "MEAN_CTL": mean_ctl_region[region_name],
    "MEAN_EXP": mean_exp_region[region_name],
    "OBS_STD": obs_std_region[region_name],
    "STD_CTL": std_ctl_region[region_name],
    "STD_EXP": std_exp_region[region_name]
  })

  # Define the file name for each region
  file_name = f"{region_name}_kge_{expids[1]}-ctl.txt"

  # Save the DataFrame to an ASCII file (tab-delimited for readability)
  df.to_csv(file_name, index=False, sep='\t')

  print(f"Saved KGE values for {region_name} to {file_name}")

print(f"\nDone. Time-series figures saved under '{TIMESERIES_DIR}/'")


