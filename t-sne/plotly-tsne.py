# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:26:28 2021

@author: TFahry
"""

from main import manual_extract
import datetime

#@title On Time and PPM Regressor (press Play to run)
Extras  = False
TSNE = True
import pandas as pd
import numpy as np

_df = pd.read_csv(r"..\ppm-on-time-new-data.csv")
#C:\Users\Tfarhy\OneDrive - Network Rail\Profile\Desktop\ppm-on-time-new-data.csv
#https://raw.githubusercontent.com/C-CA/keras-demos/main/ppm-on-time-new-data.csv
df = _df.copy()
#df = df[['Primary Delay per 100 miles','Footfall','Count of Trains/Timing Points - WTT','Planned','On Time WTT%','PPM%']]

df['Day of Week'] = df.apply(lambda row: datetime.datetime.strptime(row['Date'], '%d/%m/%Y').strftime('%A') ,axis = 1)


#%%
tsnedf2 = df.copy()
tsnedf2 = tsnedf2.loc[tsnedf2['Primary Delay per 100 miles']<4]

dfs = []
z = manual_extract(iters = 50000, early = 1, perplexity = 50, random_state = 224590)

# for i in range(100):
#     random_state = np.random.randint(1,999999)
#     z = manual_extract(iters = 500, early = 1, perplexity = 50, random_state =random_state )
#     temp = pd.DataFrame(z[-1], columns = ['x','y'])
#     temp = temp.merge(tsnedf2, left_index = True, right_index = True)
#     temp['iter'] = random_state
#     dfs.append(temp)
        

# for perplexity in range(50,70):
#     z = manual_extract(iters = 500, early = 1, perplexity = perplexity, random_state = 123123)
#     temp = pd.DataFrame(z[-1], columns = ['x','y'])
#     temp = temp.merge(tsnedf2, left_index = True, right_index = True)
#     temp['iter'] = perplexity
#     dfs.append(temp)

'''
full animation for a single perplexity value
'''
for i, frame in enumerate(z):
    if i%3 == 0:
        temp = pd.DataFrame(frame, columns = ['x','y'])
        temp = temp.merge(tsnedf2, left_index = True, right_index = True)
        temp['iter'] = i
        dfs.append(temp)
'''
single output, no animation
'''
# temp = pd.DataFrame(z[-1], columns = ['x','y'])
# temp = temp.merge(tsnedf2, left_index = True, right_index = True)
# temp['iter'] = len(z)
# dfs.append(temp)


tsnedf2 = pd.concat(dfs)
#tsnedf2 = tsnedf2.loc[tsnedf2['iter']<310]


import plotly.express as px
from plotly.offline import plot
fig = px.scatter(
    tsnedf2, x='x', y='y',
    color=tsnedf2['On Time WTT%'],
    height = 800, width = 800,
    animation_frame = 'iter',
    color_continuous_scale='agsunset',
    title = f'A t-SNE transform of the On Time dataset, [seed = {224590}]',
    hover_name = 'Date', hover_data = {'x':False,
                                      'y':False,
                                      'On Time WTT%':False,
                                      'iter':False,
                                      'Planned' : True,
                                      'Primary Delay per 100 miles':True,
                                      'Footfall': True},

)

#%%
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
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 0.1



plot(fig)
