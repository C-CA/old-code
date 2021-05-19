# -*- coding: utf-8 -*-



"""
TODO

We need to make delay PfPI be columns for each date+location entry = Unstack
We need to group both 901 and 902 Other.
We need to resolve alternately named stations.

"""

import xlwings as xw
app = xw.App(visible=False)

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

import seaborn as sns

import glob

def filesin(directory):
    return glob.glob(directory + "/*.xlsx")

root = 'C:\\Users\\Tfarhy\\OneDrive - Network Rail\\2021.01.21_Keras demos\\Correlation Demo\\'

#%%
footfalldfs = []

for file in filesin(root + 'Footfall'):

    book = app.books.open(file)
    print('Reading '+str(book))
    sheet = book.sheets[0]

    footfalldf = sheet.range('A1').expand().options(pd.DataFrame, header = 1, index = False).value
    footfalldf = footfalldf[['Day','Station','Count In']]
    footfalldf = footfalldf.dropna()
    
    footfalldfs.append(footfalldf)
    
    book.close()

footfalldf = pd.concat(footfalldfs, axis=0, ignore_index=True)

dailyfootfall = footfalldf.groupby(['Station','Day']).sum().reset_index()
dailyfootfall = dailyfootfall.rename(columns = {'Day':'Date','Station':'Geography Description'})

stations = set(dailyfootfall['Geography Description'])

print(footfalldf)
#%%
ppmdfs = []

for file in filesin(root + 'PPM by Day files'):
    
    book = app.books.open(file)
    print('Reading '+str(book))
    sheet = book.sheets['On Time by Location']
    
    ppmdf = sheet.range('A1').expand().options(pd.DataFrame, header = 1, index = False).value
    ppmdf = ppmdf[['Date','Geography Code','Geography Description','Count of Trains/Timing Points - WTT','On Time - WTT']]
    ppmdf = ppmdf.dropna()
    
    ppmdfs.append(ppmdf)
    
    book.close()

ppmdf  = pd.concat(ppmdfs, axis=0, ignore_index=True)

print(ppmdf)



#%%
delaydfs = []

for file in filesin(root + 'Reactionary Delay - Congestion Dashboard'):
    
    book = app.books.open(file)
    print('Reading '+str(book))
    sheet = book.sheets['Reactionary Delay']
    #x='A';len(set(sheet.range(f'{x}2:{x}149809').value))
    
    delaydf = sheet.range('A1').expand().options(pd.DataFrame, header = 1, index = False).value
    delaydf = delaydf[['Date','Reactionary Reason','Incident Location','v_Common End Location','PfPI Minutes']]
    delaydf = delaydf.dropna()
    
    delaydfs.append(delaydf)
    
    book.close()

delaydf = pd.concat(delaydfs, axis=0, ignore_index=True)

delaydf = delaydf.groupby(['Date','v_Common End Location','Reactionary Reason',]).sum().unstack(fill_value = 0)
delaydf = delaydf.reset_index()
delaydf.columns = delaydf.columns.map('_'.join)
delaydf = delaydf.rename(columns = {'v_Common End Location_':'Geography Description','Date_':'Date'})

delaydf = delaydf[delaydf['Geography Description'].isin(stations)]


print(delaydf)

#%%
[app.quit() for app in xw.apps]


#sns.pairplot(delaydf, diag_kind='kde' ,plot_kws={"s": 2}, height=3)
#%%
fv = ppmdf.merge(delaydf, on=['Geography Description','Date'])
fv = fv.groupby(['Date','Geography Description','PfPI Minutes_901 - Lost Path','PfPI Minutes_901 - Other','PfPI Minutes_902 - Late Crew','PfPI Minutes_902 - Late Stock','PfPI Minutes_902 - Other']).sum()

fw = fv.reset_index()
fz = dailyfootfall.merge(fw)
