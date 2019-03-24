import os
import json
import argparse
import requests
VERBOSE = False 

def print_code(status_code):
    if status_code not in [200, 201]:
        print(status_code)
    elif VERBOSE:
        print(status_code)


path = os.path.normpath(os.path.join(os.getcwd(), 'processing/v2/meta/all_data.json'))
with open(path) as f:
    data = json.load(f)

# don't bother trying this token - it's been disabled ;)
auth_key = {'Authorization': 'Token 462e482d41aeac63ab04171a4fed47e73cf5c5e5'}

#933d4a0c48c13d8b759fd47a5cb0deed64314ac6
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
        response = requests.request('PATCH', this_url, data=add_data, headers=auth_key)
        if response.status_code == 500:
            requests.put(this_url, data=add_data, headers=auth_key)
        print_code(reponse.status_code)

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
        response = requests.request('PATCH', this_url, data=add_data, headers=auth_key)
        if response.status_code == 500:
            requests.put(this_url, data=add_data, headers=auth_key)
        print_code(response.status_code)
        #print(add_data)
        index += 1

def update_airlines(data):

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
        if response.status_code == 500:
            requests.put(this_url, data=add_data, headers=auth_key)
        print_code(response.status_code)
        index += 1

def update_airports(data):
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
        if response.status_code == 500:
            requests.put(this_url, data=add_data, headers=auth_key)
        print_code(response.status_code)
        index += 1

def put_routes(data):
    routes_url = base_url + 'routes/'
    fn_url = base_url + 'flightnumbers/'
    fn_database_id = 1

    index = 1
    for route in routes:
        print('Starting new route...')

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

        print('Adding flight numbers...')
        for flight_number in route['route_flight_numbers']:
            url = fn_url + str(fn_database_id) + '/'
            unique_num = flight_number + '_' + route['route_name']
            data = {'flight_number_unique': unique_num, 'flight_number': flight_number}
            if fn_database_id > 8115:
                response = requests.request('PATCH', url, data=data, headers=auth_key)
                if response.status_code == 500:
                    response = requests.put(url, data=data, headers=auth_key)
                print(response.text)
                print_code(response.status_code)
            fn_database_id += 1
        print('Done adding flight numbers!')

        try:
            route['route_flight_numbers'] = list(map(lambda x: x + '_' + route['route_name'], route['route_flight_numbers']))
            route['route_destination_airport'] = airport_dict[route['route_destination_airport']]
            route['route_origin_airport'] = airport_dict[route['route_origin_airport']]
            add_data = route
            #print(add_data)
            if fn_database_id > 8115:
                response = requests.request('PATCH', this_url, data=add_data, headers=auth_key)
                if response.status_code == 500:
                    response = requests.put(this_url, data=add_data, headers=auth_key)
                print(response.text)
                print_code(response.status_code)
        except KeyError as e:
            pass
        index += 1

parser = argparse.ArgumentParser()
parser.add_argument('--update-airlines', action='store_true', help='Update airline resources in the API')
parser.add_argument('--update-airports', action='store_true', help='Update airport resources in the API')
parser.add_argument('--update-routes', action='store_true', help='Update route resources in the API')
parser.add_argument('--update-all', action='store_true', help='Update all resources in the API')
parser.add_argument('--update-relations-only', action='store_true', help='Update just the ManyToMany relationships between resources in the API')
parser.add_argument('--verbose', action='store_true', help='Print all response codes from API')
args = parser.parse_args()

VERBOSE = args.verbose

# run the functions
if (args.update_airlines or args.update_all) and not args.update_relations_only:
    print('Updating airlines...', end=' ')
    put_airlines(data)
    print('done!')

if (args.update_airports or args.update_all) and not args.update_relations_only:
    print('Updating airports...', end=' ')
    put_airports(data)
    print('done!')

if (args.update_airlines or args.update_all) or args.update_relations_only:
    print('Updating airlines ManyToMany and ForeignKey relationships...', end=' ')
    update_airlines(data) 
    print('done!')

if (args.update_airports or args.update_all) or args.update_relations_only:
    print('Updating airports ManyToMany and ForeignKey relationships...', end=' ')
    update_airports(data)
    print('done!')

if (args.update_routes or args.update_all) or args.update_relations_only:
    print('Updating routes...', end=' ')
    put_routes(data)
    print('done!')