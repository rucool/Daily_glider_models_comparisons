#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:00:04 2019

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'

url_nhc = 'https://www.nhc.noaa.gov/gis/'

# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%%

from erddapy import ERDDAP
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import os
import requests
import urllib.request
from bs4 import BeautifulSoup
import glob
from zipfile import ZipFile

# Do not produce figures on screen
plt.switch_backend('agg')

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Get time bounds for the previous day
te = datetime.today()
tend = datetime(te.year,te.month,te.day)

#ti = datetime.today() - timedelta(1)
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
        'time'
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

#%% Download kmz files
os.system('rm -rf *best_track*')
os.system('rm -rf *TRACK_latest*')
os.system('rm -rf *CONE_latest*')

r = requests.get(url_nhc)
data = r.text

soup = BeautifulSoup(data,"html.parser")

for i,s in enumerate(soup.find_all("a")):
    ff = s.get('href')
    if type(ff) == str:
        if np.logical_and('kmz' in ff, str(tini.year) in ff):
            if 'CONE_latest' in ff:
                file_name = ff.split('/')[3]
                print(ff, file_name)
                urllib.request.urlretrieve(url_nhc[:-4] + ff , file_name)
            if 'TRACK_latest' in ff:
                file_name = ff.split('/')[3]
                print(ff, file_name)
                urllib.request.urlretrieve(url_nhc[:-4] + ff ,file_name)
            if 'best_track' in ff:
                file_name = ff.split('/')[1]
                print(ff,file_name)
                urllib.request.urlretrieve(url_nhc + ff ,file_name)

#%%
kmz_files = glob.glob('*.kmz')

# NOTE: UNTAR  the .kmz FILES AND THEN RUN FOLLOWING CODE
for f in kmz_files:
    os.system('cp ' + f + ' ' + f[:-3] + 'zip')
    #os.system('mkdir ' + f[:-4])
    os.system('unzip -o ' + f + ' -d ' + f[:-4])

#%% get names zip and kml files
zip_files = glob.glob('*.zip')
zip_files = [f for f in zip_files if np.logical_or('al' in f,'AL' in f)]

#%% Map of North Atlantic with glider tracks
col = ['red','darkcyan','gold','m','darkorange','crimson','lime',\
       'darkorchid','brown','sienna','yellow','orchid','gray']
mark = ['o','*','p','^','D','X','o','*','p','^','D','X','o']

fig, ax = plt.subplots(figsize=(10, 5))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,cmap='Blues_r')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.yticks([])
plt.xticks([])
plt.axis([-100,-10,0,50])
plt.title('Active Glider Deployments on ' + str(tini)[0:10],fontsize=20)

#%%
for i,f in enumerate(zip_files):
    kmz = ZipFile(f, 'r')
    if 'TRACK' in f:
        kml_f = glob.glob(f[:-4]+'/*.kml')
        kml_track = kmz.open(kml_f[0].split('/')[1], 'r').read()

        #%% Get TRACK coordinates
        soup = BeautifulSoup(kml_track,'html.parser')

        lon_forec_track = np.empty(len(soup.find_all("point")))
        lon_forec_track[:] = np.nan
        lat_forec_track = np.empty(len(soup.find_all("point")))
        lat_forec_track[:] = np.nan
        for i,s in enumerate(soup.find_all("point")):
            print(s.get_text("coordinates"))
            lon_forec_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
            lat_forec_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])
        '''
        # Get time stamp
        time = []
        for i,s in enumerate(soup.find_all("td")):
            #print(s.get_text(""))
            if len(s.get_text(' ').split('Valid at:'))>1:
                time.append(s.get_text(' ').split('Valid at:')[1])
        '''
        plt.plot(lon_forec_track, lat_forec_track,'.-',color='darkorange')

    else:
        if 'CONE' in f:
            kml_f = glob.glob(f[:-4]+'/*.kml')
            kml_cone = kmz.open(kml_f[0].split('/')[1], 'r').read()

            #%% CONE coordinates
            soup = BeautifulSoup(kml_cone,'html.parser')

            lon_forec_cone = []
            lat_forec_cone = []
            for i,s in enumerate(soup.find_all("coordinates")):
                coor = s.get_text('coordinates').split(',0')
                for st in coor[1:-1]:
                    lon_forec_cone.append(st.split(',')[0])
                    lat_forec_cone.append(st.split(',')[1])

            lon_forec_cone = np.asarray(lon_forec_cone).astype(float)
            lat_forec_cone = np.asarray(lat_forec_cone).astype(float)

            plt.plot(lon_forec_cone,lat_forec_cone,'.-b',markersize=1)

        else:
            kml_f = glob.glob(f[:-4]+'/*.kml')
            kml_best_track = kmz.open(kml_f[0].split('/')[1], 'r').read()

            #%% best track coordinates
            soup = BeautifulSoup(kml_best_track,'html.parser')

            lon_best_track = np.empty(len(soup.find_all("point")))
            lon_best_track[:] = np.nan
            lat_best_track = np.empty(len(soup.find_all("point")))
            lat_best_track[:] = np.nan
            for i,s in enumerate(soup.find_all("point")):
                print(s.get_text("coordinates"))
                lon_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
                lat_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])
            '''
            # get time stamp
            time_best_track = []
            for i,s in enumerate(soup.find_all("td")):
                if 'UTC' in s.get_text(' '):
                    time_best_track.append(s.get_text(' '))
            '''
            #get name
            for f in soup.find_all('name'):
                if 'AL' in f.get_text('name'):
                    name = f.get_text('name')

            plt.plot(lon_best_track,lat_best_track,'or',markersize=3)
            plt.text(np.mean(lon_best_track),np.mean(lat_best_track),name.split(' ')[-1],fontsize=14,\
                     fontweight = 'bold')

for i,id in enumerate(gliders):
    print('Reading ' + id)
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables

    # Checking data frame is not empty
    df = e.to_pandas()
    if len(df.index) != 0:

        # Convertimg glider data to data frame
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.nanmean(df['longitude (degrees_east)']),\
                np.nanmean(df['latitude (degrees_north)']),'-',color=col[i],\
                marker = mark[i],markeredgecolor = 'k',markersize=7,\
                label=id.split('-')[0])

plt.legend(loc='upper center',bbox_to_anchor=(1.05,1))
folder = '/www/web/rucool/hurricane/Hurricane_season_2019/' + ti.strftime('%b-%d') + '/'
file = folder + 'Daily_map_North_Atlantic_gliders_in_DAC_' + str(tini).split()[0] + '_' + str(tend).split()[0] + '.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%
os.system('rm -rf *best_track*')
os.system('rm -rf *TRACK_latest*')
os.system('rm -rf *CONE_latest*')
