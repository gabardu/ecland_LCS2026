# Module/program to generate climatology of discharge
# adapted from s01-channel_params.sh / calc_outclm.F90
# E. Dutra Oct 2019

from __future__ import print_function
from netCDF4 import Dataset
import numpy as np
import sys
import os
import time
import xarray as xr
import cython_ext as ce


def accumulate_river(i1seqx,i1seqy,i2nextx,i2nexty,xin):
  ## do the routing
  xout=np.ma.masked_all(i2nextx.shape,dtype=xin.dtype)

  for iseq in range(len(i1seqx)):
    ix = i1seqx[iseq]
    iy = i1seqy[iseq]
    xout[iy,ix] = xin[iy,ix]

  for iseq in range(len(i1seqx)):
    ix = i1seqx[iseq]
    iy = i1seqy[iseq]
    if (i2nextx[iy,ix] > 0):
      jx = i2nextx[iy,ix] -1
      jy = i2nexty[iy,ix] -1
      xout[jy,jx] = xout[jy,jx]+xout[iy,ix]
  return xout

def get_args():
  from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter

  description = 'Compute discharge climatology'
  parser = ArgumentParser(description=description,formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('-iriv',dest='friv',type=str,default=None,
                     help="river network file")
  parser.add_argument('-irof',dest='frunoff',type=str,default=None,
                     help="Input runoff file")
  parser.add_argument('-imap',dest='finpmat',type=str,default=None,
                     help="Input inpmat")
  parser.add_argument('-o',dest='foutput',type=str,default='outclm.nc',
                     help="output file")
  parser.add_argument("-d", dest='debug',help="swith debug",default=False,
                    action="store_true")

  opts = parser.parse_args()
  return opts

zmiss=-9999
#opts = get_args()
#print(opts)
expid="ctl_48r1_mod"
expid=sys.argv[1]
site=sys.argv[2] #"YU-011_2013-2016"
friv=f"/perm/paga/ecland_input/clim/Yukon/ncdata_{site}.nc"
finpmat=f"/perm/paga/ecland_input/clim/Yukon/inpmat_{site}.nc"
frunoff=f"/perm/paga/Yukon/{expid}/{site}/o_wat.nc"
foutput=f"/perm/paga/Yukon/{expid}/{site}/o_wat_accum.nc"
##=================================
## 1. Load data
t0 = time.time()
  # river network
#**nc_riv = Dataset(opts.friv,'r')
nc_riv = Dataset(friv,'r')
i2nextx = nc_riv['nextx'][:]
i2nexty = nc_riv['nexty'][:]
lat = nc_riv['lat'][:]
lon = nc_riv['lon'][:]
nc_riv.close()

 ## load runoff
#***nc_rof = Dataset(opts.frunoff,'r')
nc_rof = xr.open_dataset(frunoff)


  ## load input matrix
#***nc_inp = Dataset(opts.finpmat,'r')
nc_inp = Dataset(finpmat,'r')
inpa = nc_inp.variables['inpa'][:]
inpx = nc_inp.variables['inpx'][:]
inpy = nc_inp.variables['inpy'][:]
inpn = nc_inp.variables['nlev'][:]
nc_inp.close()
print("Loading data in %6.2f seconds"%(time.time()-t0))

##=================================================
## 2. River sequence
t0 = time.time()
rivseq = ce.gen_rivseq(i2nextx,i2nexty)
print("computing rivseq in %6.2f seconds"%(time.time()-t0))
#assert np.allclose(rivseq_riv,rivseq) , 'computed rivseq does not match clim file'

## compute i1seq
t0 = time.time()
i1seqx,i1seqy = ce.calc_1d_seq_rivseq(rivseq)
print("computing i1seq in %6.2f seconds "%(time.time()-t0))

##=================================================

# 3. Create empty xarray variables
rof=nc_rof['Qsm']
empty_data = np.zeros((len(rof['time']), len(lat), len(lon)))
rivout = xr.Dataset(
    {
        'Qsm_acc': (['time', 'lat', 'lon'], empty_data.copy()),
        'Qsb_acc': (['time', 'lat', 'lon'], empty_data.copy()),
        'Qs_acc': (['time', 'lat', 'lon'], empty_data.copy()),
        'Rainf_acc': (['time', 'lat', 'lon'], empty_data.copy()),
        'Snowf_acc': (['time', 'lat', 'lon'], empty_data.copy()),
    },
    coords={
        'time': rof['time'],
        'lat': lat,
        'lon': lon,
    },
)

rivout['Qsm_acc'].attrs['long_name'] = 'snowmelt accumulated over river network'
rivout['Qsm_acc'].attrs['units'] = 'm3/s'
rivout['Qsb_acc'].attrs['long_name'] = 'subsurface accumulated over river network'
rivout['Qsb_acc'].attrs['units'] = 'm3/s'
rivout['Qs_acc'].attrs['long_name'] = 'surface runoff accumulated over river network'
rivout['Qs_acc'].attrs['units'] = 'm3/s'
rivout['Rainf_acc'].attrs['long_name'] = 'Rainfall accumulated over river network'
rivout['Rainf_acc'].attrs['units'] = 'm3/s'
rivout['Snowf_acc'].attrs['long_name'] = 'Snowfall accumulated over river network'
rivout['Snowf_acc'].attrs['units'] = 'm3/s'

## 4. interpolate water variables to river network
variables=['Qsm_acc','Qsb_acc','Qs_acc','Rainf_acc','Snowf_acc']
for var in variables:
  varname=var.split("_")[0]
  t0 = time.time()
  runoff = nc_rof[varname].values # get the data only
  # Transform to m/s . Original data in kg m-2 / dt (day)
  runoff = runoff / (1000.0)

  for tt in np.arange(0,len(rof['time'])):
      runoff_river = ce.remap(inpn,inpx,inpy,inpa,runoff[tt,:,:].astype('f8'))

      ## 4. Accumulate over the river network
      rivout[var][tt,:,:] = ce.accumulate_river(i1seqx,i1seqy,i2nextx,i2nexty,runoff_river)
  print(f"Accumulate variable {varname} on river network in %6.2f seconds "%(time.time()-t0))

##=================================================
## 5. Save to output
print('Saving to',foutput)
rivout.to_netcdf(foutput)
