import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.error import HTTPError

# for HTTP requests use https://archive.sensor.community/<date>/<date>_bme280_sensor_<sensornumber>.csv

# id = sensor_id, dates = datestrings (list)
def download_bme280(id, dates) -> pd.DataFrame:
    error = 0
    dates = sorted(dates)
    df_all = pd.DataFrame()

    for idx,date in enumerate(dates):
        if date != dates[0]:
            try:
                api_url = f"https://archive.sensor.community/{date}/{date}_bme280_sensor_{id}.csv"
                df_date = pd.read_csv(api_url,index_col=False,sep=';')
                df_all = pd.concat([df_all,df_date],ignore_index=True,verify_integrity=True)
            except HTTPError:
                error += 1
    
        else:
            try:
                api_url = f"https://archive.sensor.community/{date}/{date}_bme280_sensor_{id}.csv"
                df_all = pd.read_csv(api_url,index_col=False,sep=';')
            except HTTPError:
                if len(dates)>0:
                    df_all = pd.DataFrame()
                error += 1
                pass
        
    if df_all.empty:
        raise(FileNotFoundError)

    return df_all

# convert timestamp-column to actual timestamp
def conv_ts(sensordf) -> pd.DataFrame:
    timestamp_conv = pd.to_datetime(sensordf['timestamp'])
    sensordf['timestamp'] = timestamp_conv
    return sensordf


def select_time(sensordf, after, before) -> pd.DataFrame:

    # convert input dataframe 'timestamp' column to datatype timestamp
    sensordf_ts = conv_ts(sensordf)
    
    # convert input strings to timestamp
    after_time = np.datetime64(after)
    before_time = np.datetime64(before)
    
    subselection = sensordf_ts[np.logical_and(sensordf_ts['timestamp']>after_time,sensordf_ts['timestamp']<before_time)]

    return subselection
    

def filter_df(sensordf) -> dict:
    min_pressure = min(sensordf['pressure'])
    min_temperature = min(sensordf['temperature'])

    for idx, pressure in enumerate(sensordf['pressure']):
        if pressure == min_pressure:
            min_pressure_idx = idx

    for idx, temperature in enumerate(sensordf['temperature']):
        if temperature == min_temperature:
            min_temperature_idx = idx
            print(idx)
    
    min_pressure_sensor_id = sensordf['sensor_id'].iloc[min_pressure_idx]
    min_temperature_sensor_id = sensordf['sensor_id'].iloc[min_temperature_idx]

    # parse values to return dict
    rd = {}
    rd["min_T"] = float(min_temperature)
    rd["min_p"] = float(min_pressure)
    rd["min_T_id"] = int(min_temperature_sensor_id)
    rd["min_p_id"] = int(min_pressure_sensor_id)
    
    return(rd)


if __name__ == "__main__":
    df = download_bme280(10135,['2022-01-09'])
    df2 = download_bme280(11036, ['2022-03-02','2022-01-01'])

    # extrema.json bonus exercise
    dates = ['2022-01-28','2022-01-29','2022-01-30','2022-01-31','2022-02-01','2022-02-02','2022-02-03']
    df10881 = conv_ts(download_bme280(10881,dates))
    df11036 = conv_ts(download_bme280(11036,dates))
    df11077 = conv_ts(download_bme280(11077,dates))
    df11114 = conv_ts(download_bme280(11114,dates))
    df_concat = pd.concat([df10881,df11036,df11077,df11114],ignore_index=True)
    df_concat_sub = select_time(df_concat,'2022-01-28 05:13:00','2022-02-03 12:31:00')
    extrema_dict = {}
    extrema_dict = filter_df(df_concat_sub)
    with open('extrema.json','w') as json_out:
        json.dump(extrema_dict,json_out)

    # plot exercise
    df_toplot_1 = download_bme280(10881,['2022-01-01'])
    df_toplot_2 = download_bme280(11036,['2022-01-01'])

    temps_10881 = df_toplot_1['temperature'].tolist()
    temps_11036 = df_toplot_2['temperature'].tolist()
    float_temps_10881 = [float(t) for t in temps_10881]
    float_temps_11036 = [float(t) for t in temps_11036]

    plt.plot(range(len(temps_10881)), float_temps_10881, label='10881', c='purple')
    plt.plot(range(len(temps_11036)), float_temps_11036, label='11036', c='green')
    plt.legend()
    plt.xlabel('Measurement index')
    plt.ylabel('Temperature [C]')
    plt.savefig('./sensors.pdf')
    #plt.show()
