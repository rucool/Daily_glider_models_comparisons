#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:31:43 2019

@author: aristizabal
"""
#%%
# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'

# FTP server RTOFS
ftp_RTOFS = 'ftp.ncep.noaa.gov'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',] #\
                  #'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  #'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']
# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'


#%%
from erddapy import ERDDAP
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
from ftplib import FTP
import os
import os.path

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Get time bounds for the previous day

te = datetime.today()
tend = datetime(te.year,te.month,te.day)

ti = datetime.today() - timedelta(1)
tini = datetime(ti.year,ti.month,ti.day)

#%% Look for datasets in IOOS glider dac

print('Looking for glider data sets')
e = ERDDAP(server = url_glider)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': tini.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'max_time': tend.strftime('%Y-%m-%dT%H:%M:%SZ'),
}

search_url = e.get_search_url(response='csv', **kw)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

# Setting constraints
constraints = {
        'time>=': tini,
        'time<=': tend,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

variables = [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity'
        ]

e = ERDDAP(
        server=url_glider,
        protocol='tabledap',
        response='nc'
        )

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Login to ftp file
'''
ftp = FTP('ftp.ncep.noaa.gov')
ftp.login()
ftp.cwd('pub/data/nccf/com/rtofs/prod/rtofs.20190711/')

#ftp.retrlines('LIST')
#ftp.dir()
#ftp.retrlines('NLST')
'''

#%% Download a/b files
'''
filename = 'rtofs_glo.t00z.n-24.archv.b'
ftp.retrbinary('RETR '+filename, open(filename,'wb').write)

filename = 'rtofs_glo.t00z.n-24.archv.a'
ftp.retrbinary('RETR '+filename, open(filename,'wb').write)
'''

#%% Read time stamp from b files
'''
filename = 'rtofs_glo.t00z.n-24.archs.b'
r = BytesIO()
ftp.retrbinary('RETR '+filename, r.write)
ll = r.getvalue()
lines = ll.decode().split('\n')

timeRTOFS = []
time_stamp = lines[-2].split()[2]
hycom_days = lines[-2].split()[3]
tzero=datetime(1901,1,1,0,0)
timeRT = tzero+timedelta(float(hycom_days))
timeRTOFS.append(timeRT)
timestampRTOFS = mdates.date2num(timeRT)
'''

#%% load RTOFS nc files

for t in np.arange(len(nc_files_RTOFS)):
    file = nc_files_RTOFS[t]

    # Login to ftp file
    ftp = FTP('ftp.ncep.noaa.gov')
    ftp.login()
    ftp.cwd('pub/data/nccf/com/rtofs/prod/')
    if tend.month < 10:
        if tend.day < 10:
            ftp.cwd('rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day))
        else:
            ftp.cwd('rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day))
    else:
        if tend.day < 10:
            ftp.cwd('rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day))
        else:
            ftp.cwd('rtofs.' + str(tini.year) + str(tini.month) + str(tini.day))

    # Download nc files
    #ftp.retrbinary('RETR '+file, open(file,'wb').write)

#%% Read RTOFS grid and time
print('Retrieving coordinates from RTOFS')

ncRTOFS = xr.open_dataset(nc_files_RTOFS[0])
latRTOFS = ncRTOFS.Latitude[:]
lonRTOFS = ncRTOFS.Longitude[:]
depthRTOFS = ncRTOFS.Depth[:]

#for t in np.arange(len(nc_files_RTOFS)):
tRTOFS = []
for t in np.arange(2):
    ncRTOFS = xr.open_dataset(nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])

#%% Loop through gliders

for id in gliders:
    #id = gliders[0]
    print('Reading ' + id)
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables

    # Converting glider data to data frame
    df = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True,
            skiprows=(1,)  # units information can be dropped.
            ).dropna()

    # Coverting glider vectors into arrays
    timeg, ind = np.unique(df.index.values,return_index=True)
    latg = df['latitude (degrees_north)'].values[ind]
    long = df['longitude (degrees_east)'].values[ind]

    dg = df['depth (m)'].values
    #vg = df['temperature (degree_Celsius)'].values
    tg = df[df.columns[3]].values
    sg = df[df.columns[4]].values

    delta_z = 0.3
    zn = np.int(np.round(np.max(dg)/delta_z))

    depthg = np.empty((zn,len(timeg)))
    depthg[:] = np.nan
    tempg = np.empty((zn,len(timeg)))
    tempg[:] = np.nan
    saltg = np.empty((zn,len(timeg)))
    saltg[:] = np.nan

    # Grid variables
    depthg_gridded = np.arange(0,np.nanmax(dg),delta_z)
    tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    tempg_gridded[:] = np.nan
    saltg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    saltg_gridded[:] = np.nan

    for i,ii in enumerate(ind):
        if i < len(timeg)-1:
            depthg[0:len(dg[ind[i]:ind[i+1]]),i] = dg[ind[i]:ind[i+1]]
            tempg[0:len(tg[ind[i]:ind[i+1]]),i] = tg[ind[i]:ind[i+1]]
            saltg[0:len(sg[ind[i]:ind[i+1]]),i] = sg[ind[i]:ind[i+1]]
        else:
            depthg[0:len(dg[ind[i]:len(dg)]),i] = dg[ind[i]:len(dg)]
            tempg[0:len(tg[ind[i]:len(tg)]),i] = tg[ind[i]:len(tg)]
            saltg[0:len(sg[ind[i]:len(sg)]),i] = sg[ind[i]:len(sg)]

    for t,tt in enumerate(timeg):
        depthu,oku = np.unique(depthg[:,t],return_index=True)
        tempu = tempg[oku,t]
        saltu = saltg[oku,t]
        okdd = np.isfinite(depthu)
        depthf = depthu[okdd]
        tempf = tempu[okdd]
        saltf = saltu[okdd]

        okt = np.isfinite(tempf)
        if np.sum(okt) < 3:
            tempg_gridded[:,t] = np.nan
        else:
            okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),\
                                 depthg_gridded < np.max(depthf[okt]))
            tempg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okt],tempf[okt])

        oks = np.isfinite(saltf)
        if np.sum(oks) < 3:
            saltg_gridded[:,t] = np.nan
        else:
            okd = np.logical_and(depthg_gridded >= np.min(depthf[oks]),\
                                 depthg_gridded < np.max(depthf[oks]))
            saltg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[oks],saltf[oks])

    # Conversion from glider longitude and latitude to RTOFS convention
    target_lonRTOFS = long
    target_latRTOFS = latg

    # Narrowing time window of RTOFS to coincide with glider time window
    # because RTOFS output is 6 hourly with the time stamp at 0Z, 6Z, 12Z and 18Z then
    # I force the initial time to be at the beggining of the previous day so
    # we have four profiles
    #tmin = tini
    #tmax = tend
    tmin = mdates.num2date(mdates.date2num(timeg[0]))
    tmax = mdates.num2date(mdates.date2num(timeg[-1]))
    oktimeRTOFS = np.where(np.logical_and(tRTOFS >= tmin,tRTOFS <= tmax))
    timeRTOFS = mdates.num2date(mdates.date2num(tRTOFS[oktimeRTOFS]))

    # Changing times to timestamp
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
    tstamp_RTOFS = [mdates.date2num(timeRTOFS[i]) for i in np.arange(len(timeRTOFS))]

    # interpolating glider lon and lat to lat and lon on RTOFS time
    sublonRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,target_lonRTOFS)
    sublatRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,target_latRTOFS)

    # getting the model grid positions for sublonm and sublatm
    oklonRTOFS = np.round(np.interp(sublonRTOFS,lonRTOFS[0,:],np.arange(len(lonRTOFS[0,:])))).astype(int)
    oklatRTOFS = np.round(np.interp(sublatRTOFS,latRTOFS[:,0],np.arange(len(latRTOFS[:,0])))).astype(int)

    # Getting glider transect from RTOFS
    print('Getting glider transect from RTOFS')
    target_tempRTOFS = np.empty((len(depthRTOFS),len(oktimeRTOFS[0])))
    target_tempRTOFS[:] = np.nan
    for i in range(len(oktimeRTOFS[0])):
        print(len(oktimeRTOFS[0]),' ',i)
        nc_file = nc_files_RTOFS[i]
        ncRTOFS = xr.open_dataset(nc_file)
        target_tempRTOFS[:,i] = ncRTOFS.variables['temperature'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
    target_tempRTOFS[target_tempRTOFS < -100] = np.nan

    target_saltRTOFS = np.empty((len(depthRTOFS),len(oktimeRTOFS[0])))
    target_saltRTOFS[:] = np.nan
    for i in range(len(oktimeRTOFS[0])):
        print(len(oktimeRTOFS[0]),' ',i)
        nc_file = nc_files_RTOFS[i]
        ncRTOFS = xr.open_dataset(nc_file)
        target_saltRTOFS[:,i] = ncRTOFS.variables['salinity'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
    target_saltRTOFS[target_saltRTOFS < -100] = np.nan

        # Temperature profile
    fig, ax = plt.subplots()

    plt.plot(tempg,-depthg,'.',color='cyan',label='_nolegend_')
    plt.plot(np.nanmean(tempg_gridded,axis=1),-depthg_gridded,'.-b',\
             label=id[:-14]+' '+str(timeg[0])[0:4]+' '+'['+str(timeg[0])[5:19]+','+str(timeg[-1])[5:19]+']')


    plt.plot(target_tempRTOFS,-depthRTOFS,'.-',color='mediumseagreen',label='_nolegend_')
    plt.plot(np.nanmean(target_tempRTOFS,axis=1),-depthRTOFS,'.-g',markersize=12,linewidth=2,\
             label='RTOFS'+' '+str(timeRTOFS[0].year)+' '+'['+str(timeRTOFS[0])[5:13]+','+str(timeRTOFS[-1])[5:13]+']')
    plt.ylabel('Depth (m)',fontsize=20)
    plt.xlabel('Temperature ($^oC$)',fontsize=20)
    plt.title('Temperature Profile ' + id,fontsize=20)
    plt.ylim([-np.nanmax(depthg)+100,0])
    plt.ylim([-np.nanmax(depthg)-100,0.1])
    plt.legend(loc='lower left',bbox_to_anchor=(-0.2,0.0),fontsize=14)
    plt.grid('on')
