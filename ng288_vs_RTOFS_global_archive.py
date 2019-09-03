"""
Created on Thu Jun 13 14:46:02 2019

@author: aristizabal
"""

# files for global RTOFS output
Dir_rtofs= '/home/aristizabal/ncep_model/rtofs.20181014/'

# RTOFS grid file name same as GOFS 3.1
url_GOFS31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# RTOFS a/b file name
prefix_ab = 'rtofs_glo.t00z.n-48.archv'

# Name of 3D variable
var_name = 'temp'

# ng288
gdata = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc'

# date limits
date_ini = '2018-10-08T00:00:00Z'
date_end = '2018-10-13T00:00:00Z'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'


#%%
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime, timedelta
from matplotlib.dates import date2num, num2date
import matplotlib.dates as mdates

import sys
sys.path.append('/home/aristizabal/ncep_model/Python_code')
from utils4HYCOM import readBinz, readgrids, readVar

import os
import os.path
import glob

#%% Reading glider data

ncglider = xr.open_dataset(gdata,decode_times=False)
latglider = ncglider.latitude[:]
longlider = ncglider.longitude[:]
time_glider = ncglider.time
time_glider = netCDF4.num2date(time_glider[:],time_glider.units)
tempglider = np.array(ncglider.temperature[0,:,:])
saltglider = np.array(ncglider.salinity[0,:,:])
depthglider = np.array(ncglider.depth[0,:,:])

timestamp_glider = date2num(time_glider)[0]

#%%
tmin = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tmax = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

okg = np.where(np.logical_and(time_glider.T >= tmin, time_glider.T <= tmax))

timeg = time_glider[0,okg[0]]
timestampg = timestamp_glider[okg[0]]
latg = np.asarray(latglider[0,okg[0]])
long = np.asarray(longlider[0,okg[0]])
depthg = depthglider[okg[0],:]
tempg = tempglider[okg[0],:]
saltg = saltglider[okg[0],:]

#%% Reading RTOFS lat and lon (same as GOFS 3.1)

GOFS31 = xr.open_dataset(url_GOFS31,decode_times=False)

latRTOFS = GOFS31['lat'][:]
lonRTOFS = GOFS31['lon'][:]

#depth31 = GOFS31_ts['depth'][:]

#%% Reading time stamp

afiles = sorted(glob.glob(os.path.join(Dir_rtofs,prefix_ab+'*.a')))

lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]

time_stamp = lines[-1].split()[2]
hycom_days = lines[-1].split()[3]
tzero=datetime(1901,1,1,0,0)
timeRTOFS=tzero+timedelta(float(hycom_days))
timestamp_RTOFS = date2num(timeRTOFS)

#%% Reading RTOFS ab files

# Interpolating latgfrom utilslider and longlider into RTOFS grid
'''
sublonRTOFS = np.interp(timestamp_RTOFS,timestamp_glider,target_lon)
sublatRTOFS = np.interp(timestamp_RTOFS,timestamp_glider,target_lat)
oklonRTOFS = np.int(np.round(np.interp(sublonRTOFS,hlon[0,:],np.arange(len(hlon[0,:])))))
oklatRTOFS = np.int(np.round(np.interp(sublatRTOFS,hlat[:,0],np.arange(len(hlat[:,0])))))
'''

#sublonRTOFS = target_lon[-1]
#sublatRTOFS = target_lat[-1]
#oklonRTOFS


# reading variable
layers = np.arange(1,42)
targ_temp_RTOFS = np.empty(len(layers))
targ_temp_RTOFS[:] = np.nan
for lyr in layers:
    print(lyr)
    temp_rtofs = readVar(afiles[0][:-2],'archive',var_name,[lyr])
    x = 2508
    y = 1858
    targ_temp_RTOFS[lyr-1] = temp_rtofs[y,x]


#%% 
# read in "thknss" from archv*.[ab] and convert it to depth [m] in 3-D array

layers = np.arange(1,42)
ztmp=readVar(afiles[0][:-2],'archive','srfhgt',[0])*0.01 # converts [cm] to [m]
for lyr in tuple(layers):
    print(lyr)
    dp=readVar(afiles[0][:-2],'archive','thknss',[lyr])/2/9806
    ztmp=np.dstack((ztmp,dp))

z3d=np.cumsum(ztmp,axis=2)              # [idm,jdm,kdm+1]
z3d=np.squeeze(z3d[:,:,1:])             # [idm,jdm,kdm]
z3d=np.array(z3d)
z3d[z3d > 10**8] = np.nan

#%%
kw = dict(levels = np.linspace(0,6000))

fig, ax = plt.subplots()
plt.contourf(z3d[:,:,0],cmap='RdYlBu_r') #,**kw)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Depth (m)',fontsize=16)

#%%
kw = dict(levels = np.linspace(0,6000))

fig, ax = plt.subplots()
plt.contourf(ztmp[:,:,0],cmap='RdYlBu_r') #,**kw)
cbar = plt.colorbar()

#%%

plt.figure()
plt.scatter(target_temp_RTOFS)





