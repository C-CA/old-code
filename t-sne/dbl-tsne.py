# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:26:28 2021

@author: TFahry
"""
import pandas as pd
import numpy as np
from main import manual_extract
root = 'C:/Users/Tfarhy/OneDrive - Network Rail/data'

#df = pd.read_csv(root+'/mapping/nr_track_master.csv')
df2 = pd.read_csv(root+'/mapping/stanox-reference.csv')
#df3 = pd.read_csv(root+'/mapping/all-stations-source-data.csv')


dbl = pd.read_excel(root+'/ppm-v2/delay-by-location.xlsx', sheet_name='2021', header = 2, usecols = ['Stanox',
                                                                                                     'Location',
                                                                                                     'CAUSED',
                                                                                                     'SUFFERED',
                                                                                                     '901 R',
                                                                                                     '902 R',
                                                                                                     'PRIM',
                                                                                                     'REACT'] )

dbl['Stanox'] = dbl['Stanox'].dropna().astype(int)
dbl = dbl.dropna()

#dbl2 = dbl.merge(df2[['STANOX','Latitude','Longitude']], left_on= 'Stanox', right_on='STANOX', how='left')
#dbl2 = dbl2.dropna()
#%%
dbl['CAUSED'] = dbl['CAUSED']+0.1
dbl['SUFFERED'] = dbl['SUFFERED']+0.1

dbl['CAUSEDLOG'] = np.log(dbl['CAUSED'])
dbl['SUFFEREDLOG'] = np.log(dbl['SUFFERED'])

#%%

tsnedf2 = dbl.copy()
tsnedf = tsnedf2[['CAUSEDLOG',
                   'SUFFEREDLOG',
                    '901 R',
                    '902 R',
                    'PRIM',
                    'REACT']]
#tsnedf2 = tsnedf2.loc[tsnedf2['Primary Delay per 100 miles']<4]

dfs = []

#%%             

'''
last frame for a range of random states
'''
for i in range(10):
    random_state = np.random.randint(1,999999)
    z = manual_extract(tsnedf, iters = 500, early = 1, perplexity = 50, random_state =random_state )
    temp = pd.DataFrame(z[-1], columns = ['x','y'])
    temp = temp.merge(tsnedf2, left_index = True, right_index = True)
    temp['iter'] = random_state
    dfs.append(temp)
        
'''
last frame for a range of perplexities
'''
# for perplexity in range(50,70):
#     z = manual_extract(iters = 500, early = 1, perplexity = perplexity, random_state = 123123)
#     temp = pd.DataFrame(z[-1], columns = ['x','y'])
#     temp = temp.merge(tsnedf2, left_index = True, right_index = True)
#     temp['iter'] = perplexity
#     dfs.append(temp)

'''
full animation for a single perplexity value
'''
# z = manual_extract(tsnedf2,  iters = 500, early = 1, perplexity = 50, random_state = 224590)
# for i, frame in enumerate(z):
#     if i%5 == 0:
#         temp = pd.DataFrame(frame, columns = ['x','y'])
#         temp = temp.merge(tsnedf2, left_index = True, right_index = True)
#         temp['iter'] = i
#         dfs.append(temp)
'''
single output, no animation
'''
# temp = pd.DataFrame(z[-1], columns = ['x','y'])
# temp = temp.merge(tsnedf2, left_index = True, right_index = True)
# temp['iter'] = len(z)
# dfs.append(temp)


tsnedf2 = pd.concat(dfs)
#tsnedf2 = tsnedf2.loc[tsnedf2['iter']<310]

#%%
import plotly.express as px
from plotly.offline import plot
fig = px.scatter(
    tsnedf2, x='x', y='y',
    color=tsnedf2['SUFFEREDLOG'],
    height = 800, width = 800,
    animation_frame = 'iter',
    color_continuous_scale='agsunset',
    #title = f'A t-SNE transform of the On Time dataset, [seed = {224590}]',
    hover_name = 'Location', hover_data = {'x':False,
                                      'y':False,}
)

fig.update_traces(marker=dict(size=3))

import plotly.graph_objs as go

# ...    

layout = go.Layout(
    yaxis=dict(
        range=[-40, 40]
    ),
    xaxis=dict(
        range=[-40, 40]
    )
)
fig.update_layout(layout)
#fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 0.1



plot(fig)
