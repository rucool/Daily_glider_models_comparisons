#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:44:59 2019

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
url_RTOFS = 'http://nomads.ncep.noaa.gov:9090/dods/rtofs/rtofs_global'

# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%%

from erddapy import ERDDAP
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean

from datetime import datetime, timedelta
#from matplotlib.dates import date2num

import numpy as np
import xarray as xr
import netCDF4


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

#%% Read GOFS 3.1 output

print('Retrieving coordinates from model')
GOFS31 = xr.open_dataset(url_GOFS,decode_times=False)

latm = GOFS31.lat[:]
lonm = GOFS31.lon[:]
depthm = GOFS31.depth[:]
ttm = GOFS31.time
tm = netCDF4.num2date(ttm[:],ttm.units)

#tmin = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
#tmax = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

tmin = tini
tmax = tend

oktimem = np.where(np.logical_and(tm >= tmin, tm <= tmax))

timem = tm[oktimem]

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

'''
oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]
'''

#%%

for id in gliders:
    print('Reading ' + id )
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
    latg = np.unique(df['latitude (degrees_north)'].values)
    long = np.unique(df['longitude (degrees_east)'].values)

    dg = df['depth (m)'].values
    #vg = df['temperature (degree_Celsius)'].values
    vg = df[df.columns[3]].values

    delta_z = 0.3
    zn = np.int(np.round(np.max(dg)/delta_z))

    depthg = np.empty((zn,len(timeg)))
    depthg[:] = np.nan
    varg = np.empty((zn,len(timeg)))
    varg[:] = np.nan

    # Grid variables
    depthg_gridded = np.arange(0,np.nanmax(dg),delta_z)
    varg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    varg_gridded[:] = np.nan

    for i,ii in enumerate(ind):
        if i < len(timeg)-1:
            depthg[0:len(dg[ind[i]:ind[i+1]]),i] = dg[ind[i]:ind[i+1]]
            varg[0:len(vg[ind[i]:ind[i+1]]),i] = vg[ind[i]:ind[i+1]]
        else:
            depthg[0:len(dg[ind[i]:len(dg)]),i] = dg[ind[i]:len(dg)]
            varg[0:len(vg[ind[i]:len(vg)]),i] = vg[ind[i]:len(vg)]

    for t,tt in enumerate(timeg):
        depthu,oku = np.unique(depthg[:,t],return_index=True)
        varu = varg[oku,t]
        okdd = np.isfinite(depthu)
        depthf = depthu[okdd]
        varf = varu[okdd]
        ok = np.isfinite(varf)
        if np.sum(ok) < 3:
            varg_gridded[:,t] = np.nan
        else:
            okd = depthg_gridded < np.max(depthf[ok])
            varg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[ok],varf[ok])

    # Conversion from glider longitude and latitude to GOFS convention
    target_lon = np.empty((len(long),))
    target_lon[:] = np.nan
    for i,ii in enumerate(long):
        if ii < 0:
            target_lon[i] = 360 + ii
        else:
            target_lon[i] = ii
    target_lat = latg

    # Changing times to timestamp
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
    tstamp_model = [mdates.date2num(timem[i]) for i in np.arange(len(timem))]

    # interpolating glider lon and lat to lat and lon on model time
    sublonm=np.interp(tstamp_model,tstamp_glider,target_lon)
    sublatm=np.interp(tstamp_model,tstamp_glider,target_lat)

    # getting the model grid positions for sublonm and sublatm
    oklonm=np.round(np.interp(sublonm,lonm,np.arange(len(lonm)))).astype(int)
    oklatm=np.round(np.interp(sublatm,latm,np.arange(len(latm)))).astype(int)

    # Getting glider transect from GOFS 3.1
    print('Getting glider transect from GOFS 3.1. If it breaks is because\
          GOFS 3.1 server is not responding')
    target_varm = np.empty((len(depthm),len(oktimem[0])))
    target_varm[:] = np.nan
    for i in range(len(oktimem[0])):
        print(len(oktimem[0]),' ',i)
        target_varm[:,i] = GOFS31.variables['water_temp'][oktimem[0][i],:,oklatm[i],oklonm[i]]

    target_varm[target_varm < -100] = np.nan

    min_val = np.round(np.min([np.nanmin(df[df.columns[3]]),np.nanmin(target_varm)]))
    max_val = np.round(np.max([np.nanmax(df[df.columns[3]]),np.nanmax(target_varm)]))

    # plot
    fig, ax = plt.subplots(figsize=(12, 10))

    grid = plt.GridSpec(3, 5, wspace=0.4, hspace=0.3)

    # Scatter plot
    ax = plt.subplot(grid[0, :4])
    kw = dict(s=30, c=df[df.columns[3]], marker='*', edgecolor='none')
    cs = ax.scatter(df.index, -df['depth (m)'], **kw, cmap=cmocean.cm.thermal)
    cs.set_clim(min_val,max_val)
    ax.set_xlim(df.index[0], df.index[-1])
    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xticklabels(' ')
    cbar = fig.colorbar(cs, orientation='vertical')
    cbar.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
    ax.set_ylabel('Depth (m)',fontsize=14)
    plt.title('Along Track Temperature ' + id)


    nlevels = max_val - min_val + 1
    kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    ax = plt.subplot(grid[1, :4])
    #plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
    cs = plt.contourf(timeg,-depthg_gridded,varg_gridded,cmap=cmocean.cm.thermal,**kw)
    if np.logical_and(min_val<=26.0,max_val>=26.0):
        plt.contour(timeg,-depthg_gridded,varg_gridded,levels=[26],colors='k')

    cs = fig.colorbar(cs, orientation='vertical')
    cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xticklabels(' ')
    plt.title('Along Track Temperature ' + id)


    ax = plt.subplot(grid[2, :4])
    #plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
    cs = plt.contourf(timem,-depthm,target_varm,cmap=cmocean.cm.thermal,**kw)
    if np.logical_and(min_val<=26.0,max_val>=26.0):
        plt.contour(timem,-depthm,target_varm,[26],colors='k')

    cs = fig.colorbar(cs, orientation='vertical')
    cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_ylim(-np.max(df['depth (m)']), 0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
    ax.xaxis.set_major_formatter(xfmt)
    plt.title('Along Track Temperature GOFS 3.1')

    oklatbath = np.logical_and(bath_lat >= np.min(latg)-5,bath_lat <= np.max(latg)+5)
    oklonbath = np.logical_and(bath_lon >= np.min(long)-5,bath_lon <= np.max(long)+5)

    bath_latsub = bath_lat[oklatbath]
    bath_lonsub = bath_lon[oklonbath]
    bath_elevs = bath_elev[oklatbath,:]
    bath_elevsub = bath_elevs[:,oklonbath]

    ax = plt.subplot(grid[1, 4:])
    plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    #plt.yticks([])
    #plt.xticks([])
    plt.axis([np.min(long)-5,np.max(long)+5,np.min(latg)-5,np.max(latg)+5])
    plt.plot(long,latg,'.k')
    plt.title('Track ' + id)
    #plt.axis('equal')

    folder = '/www/web/rucool/hurricane/Hurricane_season_2019/' + ti.strftime('%b-%d') + '/'
    file = folder + 'along_track_temp_' + id + '_' + str(tini).split()[0] + '_' + str(tend).split()[0]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)


#plt.show()
