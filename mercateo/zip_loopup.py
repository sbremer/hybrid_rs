import script_chdir
import requests
import numpy as np
import pandas as pd
import pickle


def get_location(zip_code, country):
    zip_code = zip_code + ' ' + country
    r = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address={}&sensor=true&key=AIzaSyABXczQ2GnW8KRW0ZC_i-mfSTLOdAVPvQQ'.format(zip_code))
    d = r.json()

    if d['status'] == 'ZERO_RESULTS':
        print('No location found for:', zip_code)
        lon = float('NaN')
        lat = float('NaN')
    else:
        try:
            loc = d['results'][0]['geometry']['location']
            lon = loc['lng']
            lat = loc['lat']
        except IndexError:
            print('Error for:', zip_code)
            lon = float('NaN')
            lat = float('NaN')

    return lon, lat

# Read in data from csv files
ratings = pd.read_csv('data/mercateo/buy_frequencies.csv')
users = pd.read_csv('data/mercateo/customer_data.csv')
items = pd.read_csv('data/mercateo/supplier_data.csv')

users_locations = pd.DataFrame(columns=['customer_no', 'loc_lon', 'lon_lat'])
users_locations = users_locations.set_index('customer_no')

for user in users.itertuples():
    lon, lat = get_location(user.postal_code, user.country_code)
    users_locations.loc[user.customer_no] = lon, lat


items_locations = pd.DataFrame(columns=['supplier_id', 'loc_lon', 'lon_lat'])
items_locations = items_locations.set_index('supplier_id')

for item in items.itertuples():
    lon, lat = get_location(item.postalcode, item.countrycode)
    items_locations.loc[item.supplier_id] = lon, lat

users_locations.to_csv('data/mercateo/users_locations.csv')
items_locations.to_csv('data/mercateo/items_locations.csv')
