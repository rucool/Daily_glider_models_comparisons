
#%%
# FTP server RTOFS
ftp_RTOFS = 'ftp.ncep.noaa.gov'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

#%%
from ftplib import FTP
import os
from datetime import datetime, timedelta
import numpy as np

#%% Get time bounds for the previous day
te = datetime.today()
tend = datetime(te.year,te.month,te.day)

ti = datetime.today() - timedelta(1)
tini = datetime(ti.year,ti.month,ti.day)

#%% Get name of folder in FTP server
if tini.month < 10:
    if tini.day < 10:
        folder = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day)
    else:
        folder = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
else:
    if tini.day < 10:
        folder = 'rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day)
    else:
        folder = 'rtofs.' + str(tini.year) + str(tini.month) + str(tini.day)

'''
if tend.month < 10:
    if tend.day < 10:
	if tini.day < 10:
        	folder = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day)
        else:
		folder = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
    else:
        folder = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
else:
    if tend.day < 10:
        folder = 'rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day)
    else:
        folder = 'rtofs.' + str(tini.year) + str(tini.month) + str(tini.day)
'''

#os.system('mkdir '+'/home/aristizabal/RTOFS_6hourly_North_Atlantic/'+folder)
os.system('mkdir '+'/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'+folder)

#%% load RTOFS nc files
print('Loading 6 hourly RTOFS nc files from FTP server')
for t in np.arange(len(nc_files_RTOFS)):
    #file = out_dir + '/' + nc_files_RTOFS[t]
    file = nc_files_RTOFS[t]

    # Login to ftp file
    ftp = FTP('ftp.ncep.noaa.gov')
    ftp.login()
    ftp.cwd('pub/data/nccf/com/rtofs/prod/'+ folder)

    # Download nc files
    print('loading ' + file)
    #Folder = '/home/aristizabal/RTOFS_6hourly_North_Atlantic/'+folder+'/'
    Folder = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'+folder+'/'
    ftp.retrbinary('RETR '+ file, open(Folder + file,'wb').write)
