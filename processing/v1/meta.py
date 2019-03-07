import glob 
import pandas as pd
import json

# this_data = pd.read_csv('flight-data/carriers.csv', skipinitialspace=True, low_memory=False)
# airline_codes_to_airlines = this_data.set_index('Code')['Description'].to_dict()
# airlines_to_airline_codes = this_data.set_index('Description')['Code'].to_dict()

this_data = pd.read_csv('flight-data/carriers_percent.csv', skipinitialspace=True, low_memory=False)
airline_codes_to_airlines = this_data.set_index('OP_UNIQUE_CARRIER')['Description'].to_dict()
airlines_to_airline_codes = this_data.set_index('Description')['OP_UNIQUE_CARRIER'].to_dict()
airline_codes_to_punctuality = this_data.set_index('OP_UNIQUE_CARRIER')['N/A(PCT_ONTIME_ARR)'].to_dict()
#airline_codes_to_punctuality = dict()

# for key in airline_codes_to_airlines.keys():
#     if key in punc:
#         airline_codes_to_punctuality[key] = punc[key]
#     else:
#         airline_codes_to_punctuality[key] = None


this_data = pd.read_csv('flight-data/airports.csv', skipinitialspace=True, low_memory=False)
airport_codes_to_airports = this_data.set_index('ORIGIN')['Description'].to_dict()
airports_to_airport_codes = this_data.set_index('Description')['ORIGIN'].to_dict()
airport_codes_to_punctuality = this_data.set_index('ORIGIN')['N/A(PCT_ONTIME_DEP)'].to_dict()

data = {}
data['airline_codes_to_airlines'] = airline_codes_to_airlines
data['airlines_to_airline_codes'] = airlines_to_airline_codes
data['airline_codes_to_punctuality'] = airline_codes_to_punctuality
data['airport_codes_to_airports'] = airport_codes_to_airports
data['airports_to_airport_codes'] = airports_to_airport_codes
data['airport_codes_to_punctuality'] = airport_codes_to_punctuality

with open('meta/lookup_all.json', 'w') as fp:
        json.dump(data, fp)