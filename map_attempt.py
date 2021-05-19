# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:27:59 2021

@author: TFahry
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)


#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

import plotly.express as px
from plotly.offline import plot

root = 'C:/Users/Tfarhy/OneDrive - Network Rail/data'

df = pd.read_csv(root+'/mapping/nr_track_master.csv')
df2 = pd.read_csv(root+'/mapping/stanox-reference.csv')
df3 = pd.read_csv(root+'/mapping/all-stations-source-data.csv')

#%%
dbl = pd.read_excel(root+'/ppm-v2/delay-by-location.xlsx', sheet_name='2021', header = 2, usecols = ['Stanox',
                                                                                                     'Location',
                                                                                                     'CAUSED',
                                                                                                     'SUFFERED'], )

dbl['Stanox'] = dbl['Stanox'].dropna().astype(int)
dbl = dbl.dropna()
#%%
#df3a = df3[['Latitude', 'Longitude', 'Station', 'Operator']]
#df3a = df3a.dropna()

dbl2 = dbl.merge(df2[['STANOX','Latitude','Longitude']], left_on= 'Stanox', right_on='STANOX', how='left')
dbl2 = dbl2.dropna()

dbl2['CAUSEDLOG'] = np.log(dbl2['CAUSED'])

#%%
fig = px.scatter_mapbox(dbl2, lat="Latitude", lon="Longitude",
                      color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                      mapbox_style="carto-positron",
                      hover_name = 'Location', color = 'CAUSEDLOG',
                      center = {'lat' : 51.5, 'lon': 0}
                       )

plot(fig)