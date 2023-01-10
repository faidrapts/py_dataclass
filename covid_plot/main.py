import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np


# github repo tsv data for BY, NW, SN, TH
url_by = 'https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-BY.tsv'
url_nw = 'https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-NW.tsv'
url_sn = 'https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-SN.tsv'
url_th = 'https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-TH.tsv'
url_total = 'https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-DE-total.tsv'

df_by = pd.read_csv(url_by, sep='\t')
df_nw = pd.read_csv(url_nw, sep='\t')
df_sn = pd.read_csv(url_sn, sep='\t')
df_th = pd.read_csv(url_th, sep='\t')
df_total = pd.read_csv(url_total, sep='\t')

max_by = max(df_by['Cases_Last_Week_Per_Million'])
arg_by = np.argmax(df_by['Cases_Last_Week_Per_Million'])
max_date_by = df_by['Date'][arg_by]
max_nw = max(df_nw['Cases_Last_Week_Per_Million'])
arg_nw = np.argmax(df_nw['Cases_Last_Week_Per_Million'])
max_date_nw = df_nw['Date'][arg_nw]
max_sn = max(df_sn['Cases_Last_Week_Per_Million'])
arg_sn = np.argmax(df_sn['Cases_Last_Week_Per_Million'])
max_date_sn = df_sn['Date'][arg_sn]
max_th = max(df_th['Cases_Last_Week_Per_Million'])
arg_th = np.argmax(df_th['Cases_Last_Week_Per_Million'])
max_date_th = df_th['Date'][arg_th]

# max in data
maxima = [max_by, max_nw, max_sn, max_th]
args = [arg_by, arg_nw, arg_sn, arg_th]
max_dates = [max_date_by, max_date_nw, max_date_sn, max_date_th]
bundeslaender = ['BY', 'NW', 'SN', 'TH']
maximum = max(maxima)
arg_in_maxima = np.argmax(maxima)

# max in plot
x_max = args[arg_in_maxima]
y_max = maximum
bundesland = bundeslaender[arg_in_maxima]

# plot
fig, ax = plt.subplots()
fig.set_figwidth(14)
fig.set_figheight(8)
ax.plot(df_by['Date'], df_by['Cases_Last_Week_Per_Million'],
        label='BY', c='dodgerblue')
ax.plot(df_nw['Date'], df_nw['Cases_Last_Week_Per_Million'],
        label='NW', c='green')
ax.plot(df_sn['Date'], df_sn['Cases_Last_Week_Per_Million'],
        label='SN', c='firebrick')
ax.plot(df_th['Date'], df_th['Cases_Last_Week_Per_Million'],
        label='TH', c='mediumpurple')
ax.set(xlabel='Date', ylabel='n/(week$\cdot$million)',
       title='7-day incidence/Mio of Covid-19 cases')
ax.grid()
ax.set_facecolor('lavender')
ax.annotate(f'Maximum value is {y_max} in {bundesland} \n on {max_dates[arg_in_maxima]}', xy=(
    x_max, y_max), xytext=(x_max, y_max+1), arrowprops=dict(facecolor='darkblue', shrink=0.005))
num_ticksx = df_by['Date'].iloc[::74].shape[0]
plt.legend(loc='upper left')
plt.xticks(df_by['Date'].iloc[::230])
plt.yscale('log')

# inset plot for whole of Germany bbox_to_anchor = (x0, y0, width, height)
inset_ax = inset_axes(ax, "30%", "30%", loc='lower right',
                      bbox_to_anchor=(-0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)
inset_ax.grid(ax)
inset_ax.set_facecolor('lavender')
inset_ax.plot(df_total['Date'],
              df_total['Cases_Last_Week_Per_Million'], c='darkblue')
inset_ax.set(title='Total Incidence in Germany', yscale='log',
             xticks=df_total['Date'].iloc[::450])

plt.savefig('plot.pdf')
#plt.show()
