import glob 
import pandas as pd
import json
from functools import reduce
from collections import defaultdict

DEBUG = False

"""
Load the airline names, ids, and percent ontime from the lookup table and create the final object
and working dictionary 
"""
def load_basic_airline_info():

    this_data = pd.read_csv('flight-data/carriers_percent.csv', skipinitialspace=True, low_memory=False)
    airlines = []
    airlines_dict = {}

    for index, row in this_data.iterrows():
        airline_info = {}
        airline_id = row['OP_UNIQUE_CARRIER']
        airline_info['airline_name'] = row['Description']
        airline_info['airline_id'] = airline_id
        airline_info['airline_percent_ontime_arrival'] = int(row['N/A(PCT_ONTIME_ARR)'])
        airlines.append(airline_info)

        airlines_dict[airline_id] = {}
        airlines_dict[airline_id]['flights'] = 0
        airlines_dict[airline_id]['departures_delay'] = 0
        airlines_dict[airline_id]['arrival_delay'] = 0
        airlines_dict[airline_id]['connecting_airports'] = []

    return airlines, airlines_dict



"""
Load the airport names, ids, cities, states, and percent ontime from the lookup table and create 
the final object and working dictionary 
"""
def load_basic_airport_info():
    this_data = pd.read_csv('flight-data/airports.csv', skipinitialspace=True, low_memory=False)
    airports = []
    airports_dict = {}

    for index, row in this_data.iterrows():
        airport_info = {}
        full_airport_name = row['Description'] # looks like "Nashville, TN: Nashville International"
        airport_split_at_colon = full_airport_name.split(': ')
        airport_id = row['ORIGIN']

        airport_info['airport_name'] = airport_split_at_colon[1]
        airport_info['airport_state'] = airport_split_at_colon[0].split(', ')[1]
        airport_info['airport_city'] = airport_split_at_colon[0].split(', ')[0]
        airport_info['airport_id'] = airport_id
        airport_info['airport_percent_ontime_departure'] = int(row['N/A(PCT_ONTIME_DEP)'])
        airports.append(airport_info)

        airports_dict[airport_id] = {}
        airports_dict[airport_id]['departures'] = 0
        airports_dict[airport_id]['departures_delay'] = 0
        airports_dict[airport_id]['arrivals'] = 0
        airports_dict[airport_id]['taxi_in'] = 0
        airports_dict[airport_id]['taxi_out'] = 0
        airports_dict[airport_id]['connecting_airports'] = []
        airports_dict[airport_id]['connecting_airlines'] = []


    return airports, airports_dict



"""
Load a month's data from all the csv files for each year and create a dataframe
"""
def load_month_data(month):

    path = 'flight-data/*/*_' + str(month) + '.csv'
    month_data = glob.glob(path)
    loaded_data = []
    for path in month_data:
        temp_data = []
        for chunk in pd.read_csv(path, skipinitialspace=True, low_memory=False, chunksize=20000):
            temp_data.append(chunk)
        this_data = pd.concat(temp_data, axis= 0)
        loaded_data.append(this_data)

    df = pd.concat(loaded_data)
    del loaded_data
    print('done!\nConsolidating dataframe... ', end=' ')

    df = df[['Reporting_Airline', 'Origin',
             'Dest', 'DepDelay', 'ArrDelay', 'TaxiIn', 'TaxiOut', 'AirTime']]
    print('done!\nDropping null values from dataframe... ', end=' ')

    df.dropna(inplace = True)
    return df



"""
Using the dataframe and working dictionaries, add information from each flight
to the routes, airports, and airlines dicts
"""
def calculate_month_flight_data(month, data, airports, airlines, routes):

    for index, row in data.iterrows():
        flight_origin = row['Origin']
        flight_dest = row['Dest']
        flight_airline = row['Reporting_Airline']
        route_key = flight_origin + '_' + flight_dest

        if route_key in routes:
            routes[route_key]['flights'] += 1
            if flight_airline not in routes[route_key]['airlines']:
                routes[route_key]['airlines'].append(flight_airline)
            routes[route_key]['time'] += row['AirTime']
        else :
            routes[route_key] = {}
            routes[route_key]['flights'] = 1
            routes[route_key]['airlines'] = [flight_airline]
            routes[route_key]['time'] = row['AirTime']
            routes[route_key]['origin'] = flight_origin
            routes[route_key]['destination'] = flight_dest

        try:
            if flight_airline not in airports[flight_origin]['connecting_airlines']:
                airports[flight_origin]['connecting_airlines'].append(flight_airline)
            if flight_dest not in airports[flight_origin]['connecting_airports']:
                airports[flight_origin]['connecting_airports'].append(flight_dest)
            airports[flight_origin]['departures'] += 1
            airports[flight_origin]['departures_delay'] += row['DepDelay']
            airports[flight_origin]['taxi_out'] += row['TaxiOut']
        except KeyError as e:
            if DEBUG:
                print('Skipping missing airport: {}'.format(flight_origin))

        try: 
            airports[flight_dest]['arrivals'] += 1
            airports[flight_dest]['taxi_in'] += row['TaxiIn']
        except KeyError as e:
            if DEBUG:
                print('Skipping missing airport: {}'.format(flight_dest))

        try:
            if flight_dest not in airlines[flight_airline]['connecting_airports']:
                airlines[flight_airline]['connecting_airports'].append(flight_dest)
            if flight_origin not in airlines[flight_airline]['connecting_airports']:
                airlines[flight_airline]['connecting_airports'].append(flight_origin)
            airlines[flight_airline]['flights'] += 1 
            airlines[flight_airline]['departures_delay'] += row['DepDelay']
            airlines[flight_airline]['arrival_delay'] += row['ArrDelay']
        except KeyError as e:
            if DEBUG:
                print('Skipping missing airline: {}'.format(flight_airline))

    return airports, airlines, routes



"""
Average out data made from adding multiple values in the airlines, routes, and airports
working dictionaries
"""
def combine_data(airports, airlines, routes):
    for key in airlines.keys():
        if airlines[key]['flights'] > 0:
            airlines[key]['departures_delay'] = airlines[key]['departures_delay'] / airlines[key]['flights']
            airlines[key]['arrival_delay'] = airlines[key]['arrival_delay'] / airlines[key]['flights']

    for key in airports.keys():
        if airports[key]['departures'] > 0 and airports[key]['departures'] > 0:
            airports[key]['departures_delay'] = airports[key]['departures_delay'] / airports[key]['departures']
            airports[key]['taxi_in'] = airports[key]['taxi_in'] / airports[key]['arrivals']
            airports[key]['taxi_out'] = airports[key]['taxi_out'] / airports[key]['departures']

    for key in routes.keys():
        if routes[key]['flights'] > 0:
            routes[key]['time'] = routes[key]['time'] / routes[key]['flights']

    return airports, airlines, routes



"""
Sort a dictionary of values in ascending or descending order
"""
def sort_dict(x, rev_order):
    return {key: rank for rank, key in enumerate(sorted(x, key=x.get, reverse=rev_order), 1)}



"""
Order data in airports, airlines, and routes dictionaries
"""
def rank_order_data(airports, airlines, routes):
    airports_volume = {}
    airports_dept_delay = {}
    for key in airports.keys():
        airports_volume[key] = airports[key]['departures'] + airports[key]['arrivals']
        airports_dept_delay[key] = airports[key]['departures_delay']

    airlines_volume = {}
    airlines_dept_delay = {}
    airlines_arr_delay = {}
    for key in airlines.keys():
        airlines_volume[key] = airlines[key]['flights']
        airlines_dept_delay[key] = airlines[key]['departures_delay']
        airlines_arr_delay[key] = airlines[key]['arrival_delay']

    route_volume = {}
    for key in routes.keys():
        route_volume[key] = routes[key]['flights']

    sorted_airlines_volume = sort_dict(airlines_volume, True)
    sorted_airlines_dept_delay = sort_dict(airlines_dept_delay, False)
    sorted_airlines_arr_delay = sort_dict(airlines_arr_delay, False)
    sorted_airports_volume = sort_dict(airports_volume, True)
    sorted_airports_dept_delay = sort_dict(airports_dept_delay, False)
    sorted_route_volume = sort_dict(route_volume, True)

    return sorted_airlines_volume, sorted_airlines_dept_delay, sorted_airlines_arr_delay, sorted_airports_volume, sorted_airports_dept_delay,sorted_route_volume



"""
Running the script to create the final JSON objects. Load each of the 12 months worth of 
data and then average out the values. 
"""
print('Loading airport data... ', end=' ')
airports, airports_dict = load_basic_airport_info()
print('done!\nLoading airline data... ', end=' ')
airlines, airlines_dict = load_basic_airline_info()
routes_dict = {}
for i in range (1, 13):
    print('done!\nLoading data for month {}... '.format(i), end=' ')
    data = load_month_data(1)
    print('done!\nCalculating values for month {}... '.format(i), end=' ')
    airports_dict, airlines_dict, routes_dict = calculate_month_flight_data(i, data, airports_dict, airlines_dict, routes_dict)

print('Combining calculated values for all months... ', end=' ')
airports_dict, airlines_dict, routes_dict = combine_data(airports_dict, airlines_dict, routes_dict)   
print('done!\nRank ordering calculated values... ', end=' ')
sorted_airlines_volume, sorted_airlines_dept_delay, sorted_airlines_arr_delay, sorted_airports_volume, sorted_airports_dept_delay, sorted_route_volume = rank_order_data(airports_dict, airlines_dict, routes_dict)
print('done!\nCreating final objects... ', end=' ')

for i in range(len(airports)):
    airport_id = airports[i]['airport_id']
    try:
        airports[i]['airport_taxi_in_time'] = int(airports_dict[airport_id]['taxi_in'])
        airports[i]['airport_taxi_out_time'] = int(airports_dict[airport_id]['taxi_out'])
        airports[i]['airport_departures_per_year'] = int(airports_dict[airport_id]['departures'] / 3)
        airports[i]['airport_arrivals_per_year'] = int(airports_dict[airport_id]['arrivals'] / 3)
        airports[i]['airport_departure_delay'] = int(airports_dict[airport_id]['departures_delay'])
        airports[i]['airport_destinations'] = airports_dict[airport_id]['connecting_airports']
        airports[i]['airport_airlines'] = airports_dict[airport_id]['connecting_airlines']
        airports[i]['airport_flight_volume_rank'] = int(sorted_airports_volume[airport_id])
        airports[i]['airport_ontime_departure_rank'] = int(sorted_airports_dept_delay[airport_id])
    except KeyError as e:
        if DEBUG:
            print('Skipping missing airport: {}'.format(airport_id))

for i in range(len(airlines)):
    airline_id = airlines[i]['airline_id']
    try:
        airlines[i]['airline_flights_per_year'] = int(airlines_dict[airline_id]['flights'] / 3)
        airlines[i]['airline_departure_delay'] = int(airlines_dict[airline_id]['departures_delay'])
        airlines[i]['airline_arrival_delay'] = int(airlines_dict[airline_id]['arrival_delay'])
        airlines[i]['airline_destinations'] = airlines_dict[airline_id]['connecting_airports']
        airlines[i]['airline_ontime_departure_rank'] = int(sorted_airlines_dept_delay[airline_id])
        airlines[i]['airline_ontime_arrival_rank'] = int(sorted_airlines_arr_delay[airline_id])
        airlines[i]['airline_flight_volume_rank'] = int(sorted_airlines_volume[airline_id])
    except KeyError as e:
        if DEBUG:
            print('Skipping missing airline: {}'.format(airline_id))

routes = []
for key in routes_dict.keys():
    route_object = {}
    route_object['route_name'] = key
    route_object['route_origin_airport'] = routes_dict[key]['origin']
    route_object['route_destination_airport'] = routes_dict[key]['destination']
    route_object['route_flights_per_year'] = int(routes_dict[key]['flights'] / 3) 
    route_object['route_time'] = int(routes_dict[key]['time'])
    route_object['route_airlines'] = routes_dict[key]['airlines']
    route_object['route_flight_volume_rank'] = int(sorted_route_volume[key])
    routes.append(route_object)

print('done!\nSaving JSON files... ', end=' ')
data = {}
data['airlines'] = airlines
data['airports'] = airports
data['routes'] = routes
with open('meta/all_data.json', 'w') as fp:
        json.dump(data, fp)


print('done!')