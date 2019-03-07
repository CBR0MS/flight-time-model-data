import os
import json
import requests

with open('meta/all_data.json') as f:
    data = json.load(f)

# don't bother trying this token - it's been disabled ;)
auth_key = {'Authorization': 'Token 933d4a0c48c13d8b759fd47a5cb0deed64314ac6'}
base_url = 'https://api.flygeni.us/'
airlines = data['airlines']
airlines.sort(key=lambda x: x['airline_flight_volume_rank'])
airports = data['airports']
airports.sort(key=lambda x: x['airport_flight_volume_rank'])
routes = data['routes']
routes.sort(key=lambda x: x['route_flight_volume_rank'])

airport_dict = dict()
index = 1
for airport in airports:
    airport_dict[airport['airport_id']] = index
    index += 1

airline_dict = dict()
index = 1
for airline in airlines:
    airline_dict[airline['airline_id']] = index
    index +=1 

def put_airlines(data):
    airlines_url =  base_url + 'airlines/'

    index = 1
    for airline in airlines:
        this_url = airlines_url + (str(index) + '/')
        airline.pop('airline_destinations', None)
        add_data = airline
        response = requests.put(this_url, data=add_data, headers=auth_key)
        print(response.status_code)
        index += 1

def put_airports(data):

    #print(airline_dict)
    airports_url = base_url + 'airports/'
    
    index = 1
    for airport in airports:
        this_url = airports_url + (str(index) + '/?use_db_ids=True')
        airport.pop('airport_destinations', None)
        airlines_list = airport['airport_airlines']
        airport.pop('airport_airlines', None)
        newlist = []
        for airline in airlines_list:
            try:
                id = airline_dict[airline]
                newlist.append(id)
            except KeyError as e:
                pass
        airport['airport_airlines'] = newlist
        add_data = airport
        response = requests.put(this_url, data=add_data, headers=auth_key)
        print(response.status_code)
        #print(add_data)
        index += 1

def update_airports_and_airlines(data):

    airlines_url =  base_url + 'airlines/'

    index = 1
    for airline in airlines:
        this_url = airlines_url + (str(index) + '/?use_db_ids=True')

        airports_list = airline['airline_destinations']
        airline.pop('airline_destinations', None)
        newlist = []
        for airport in airports_list:
            try:
                id = airport_dict[airport]
                newlist.append(id)
            except KeyError as e:
                pass
        add_data = {'airline_destinations': newlist }
       # print(add_data)
        response = requests.request('PATCH', this_url, data=add_data, headers=auth_key)
        print(response.status_code)
        index += 1

    airports_url = base_url + 'airports/'

    index = 1
    for airport in airports:
        this_url = airports_url + (str(index) + '/?use_db_ids=True')

        airports_list = airport['airport_destinations']
        airport.pop('airport_destinations', None)
        newlist = []
        for airport in airports_list:
            try:
                id = airport_dict[airport]
                newlist.append(id)
            except KeyError as e:
                pass
        add_data = {'airport_destinations': newlist }
        #print(add_data)
        response = requests.request('PATCH', this_url, data=add_data, headers=auth_key)
        print(response.status_code)
        index += 1

def put_routes(data):
    routes_url = base_url + 'routes/'

    index = 1
    for route in routes:
        if index > 2033:
            this_url = routes_url + (str(index) + '/?use_db_ids=True')
            
            airlines_list = route['route_airlines']
            route.pop('route_airlines', None)
            newlist = []
            for airline in airlines_list:
                try:
                    id = airline_dict[airline]
                    newlist.append(id)
                except KeyError as e:
                    pass
            route['route_airlines'] = newlist
            try:
                route['route_destination_airport'] = airport_dict[route['route_destination_airport']]
                route['route_origin_airport'] = airport_dict[route['route_origin_airport']]
                add_data = route
                response = requests.put(this_url, data=add_data, headers=auth_key)
                print(response.status_code)
            except KeyError as e:
                pass
        #print(add_data)
        index += 1


# run the functions
put_airlines(data)
put_airports(data)
update_airports_and_airlines(data)
put_routes(data)