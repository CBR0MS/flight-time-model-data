from datetime import datetime
startTime = datetime.now()
import json
import glob
import numpy as np 
import sklearn
import pandas as pd
from io import StringIO
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

EPOCHS = 12
airports = dict()
airlines = dict()

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

# load the carriers lookup
this_data = pd.read_csv('flight-data/carriers.csv', skipinitialspace=True, low_memory=False)
airlines_lookup = this_data.set_index('Code')['Description'].to_dict()

# we are going to create twelve different models, one for each month
# of the year. The importing, cleaning, and training all take place
# twelve times and result in twelve different model json files. 
for i in range (1, 13):

    print('Starting month {}'.format(i))
    print('===============================================================\nLoading data for month {}... '.format(i), end=' ')

    path = 'flight-data/*/*_' + str(i) + '.csv'
    month_data = glob.glob(path)
    loaded_data = []
    for path in month_data:
        this_data = pd.read_csv(path, skipinitialspace=True, low_memory=False)
        loaded_data.append(this_data)

    df = pd.concat(loaded_data)
    print('done!\nConsolidating dataframe... ', end=' ')

    df = df[['DayofMonth', 'DayOfWeek', 'Reporting_Airline', 'Origin',
             'Dest', 'DepDelay', 'ArrDelay']]
    print('done!\nDropping null values from dataframe... ', end=' ')

    df.dropna(inplace = True)
    print('done!\nEncoding airport names... ', end=' ')
    
    if i == 1:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df['Origin'])
        zipped = zip(df['Origin'], integer_encoded)
        airports = dict(zipped)

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df['Reporting_Airline'])
        zipped = zip(df['Reporting_Airline'], integer_encoded)
        airlines = dict(zipped)

        airlines = {k:int(v) for k,v in airlines.items()}
        airports = {k:int(v) for k,v in airports.items()}

        with open('meta/location_names.json', 'w') as fp:
            json.dump(airports, fp)

        with open('meta/airline_abbreviations.json', 'w') as fp:
            json.dump(airlines, fp)

        with open('meta/airline_names.json', 'w') as fp:
            json.dump(airlines_lookup, fp)
        
    df['Origin'] = df['Origin'].replace(to_replace=airports, value=None)
    df['Dest'] = df['Dest'].replace(to_replace=airports, value=None)
    print('done!\nEncoding airline names... ', end=' ')

    df['Reporting_Airline'] = df['Reporting_Airline'].replace(to_replace=airlines, value=None)
    print('done!\nRemoving non-encoded values... ', end=' ')

    df['Origin'] = df['Origin'].apply(lambda x: isinstance(x, (int, np.int64)))
    df['Dest'] = df['Dest'].apply(lambda x: isinstance(x, (int, np.int64)))
    df['Reporting_Airline'] = df['Reporting_Airline'].apply(lambda x: isinstance(x, (int, np.int64)))
    print('done!\nDropping delays not in range [-30, 30]... ', end=' ')

    df = df[df['DepDelay'] < 30]  
    df = df[df['DepDelay'] > -30]  
    print('done!\n')

    for j in range(2):
        if (j == 0):
            print('Building departure time model...\n')
            y = df['DepDelay']
            path = 'models/' + str(i) + '_' + 'dep'
        else:
            print('Building arrival time model...\n')
            y = df['ArrDelay']
            path = 'models/' + str(i) + '_' + 'arr'

        print('Splitting data into training and testing sets... ', end=' ')
        X = df.drop(columns=['DepDelay', 'ArrDelay'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print('done!\n')

        print('Building and training model... \n')
        model = build_model()

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.33, verbose=1, 
                  callbacks=[early_stop], use_multiprocessing=True)
        print('\nSaving model... ', end=' ')
        #tf.keras.models.save_model(model, path, overwrite=True, include_optimizer=False)
        tfjs.converters.save_keras_model(model, path)
        print('done!\n')

    print('Finished month {}!\n'.format(i))

print('Finished running in {}'.format(datetime.now() - startTime))