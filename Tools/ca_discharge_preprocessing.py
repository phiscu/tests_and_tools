import pandas as pd
import os

wd = '/home/phillip/Seafile/CLIMWATER/Data/CA-Discharge/chirchik/'
os.chdir(wd)
data = pd.read_csv('discharge_chirchik.csv')

# Gauge files
unique_ids = data['CODE'].unique()
gaps_info = []  # List to store gap information

for unique_id in unique_ids:
    subset = data[data['CODE'] == unique_id][['date', 'value']]
    subset.columns = ['date', 'q_m3s']
    subset['date'] = pd.to_datetime(subset['date'])
    subset['gap'] = subset['date'].diff()
    subset['gap'] = subset['gap'].where(subset['gap'] > pd.Timedelta(days=11), False)
    # First and last date
    first_date = subset['date'].min()
    last_date = subset['date'].max()
    # Total number of gap days
    if not subset[subset['gap'] != False].empty:
        total_gaps = subset[subset['gap'] != False]['gap'].sum().days
    else:
        total_gaps = 0

    gaps_info.append({'CODE': unique_id, 'first_date': first_date,
                      'last_date': last_date, 'gaps': total_gaps})

    subset.to_csv(f'{unique_id}_data.csv', index=False)

# Meta file
id_lat_lon = data[['CODE', 'LAT', 'LON']].drop_duplicates()
gaps_df = pd.DataFrame(gaps_info)
id_lat_lon = id_lat_lon.merge(gaps_df, on='CODE', how='left')

id_lat_lon.to_csv('meta_data.csv', index=False)
