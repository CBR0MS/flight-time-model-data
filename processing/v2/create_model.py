from datetime import datetime
startTime = datetime.now()
import json
import glob
import os
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from sklearn.model_selection import train_test_split
import requests

EPOCHS = 9
CLASSES = 2

"""
Build and return the Keras model ready to fit 
"""
def build_classification_model(X_train):
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.sigmoid, input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.sigmoid), 
    keras.layers.Dense(64, activation=tf.nn.sigmoid),
    keras.layers.Dense(CLASSES, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  return model

"""
get the percent ontime of a particular airline by id
"""
def get_airline_percent_ontime(id, airlines):
    for obj in airlines:
        if obj['airline_id'] == id:
            return obj['airline_percent_ontime_arrival']

"""
get the percent ontime of a particular airport by id
"""       
def get_airport_percent_ontime(id, airports):
    for obj in airports:
        if obj['airport_id'] == id:
            return obj['airport_percent_ontime_departure']

"""
create the classes for classifiying each departure or arrival time as 
ontime or late 
"""
def create_classes(y):
    for i in range(len(y)):
        if y[i] < 10:
            y[i] = 0
        else:
            y[i] = 1
    return y

"""
create the classes and split the data into training and testing 
"""
def prepare_data(X, y):
    y = y.tolist()
    y = create_classes(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)


"""
Run the program to laod data, create and fit model, test, and save model as json
"""
print('Getting airport and airline metadata from FlyGenius API...', end=' ')
r = requests.get('https://api.flygeni.us/airports/?use_details=True')
airports = r.json()

r = requests.get('https://api.flygeni.us/airlines/?use_details=True')
airlines = r.json()
print('done!\nLoading raw flight data from CSV files...', end=' ')

path = os.path.normpath(os.path.join(os.getcwd(), 'data/flight-data/*_2017_*/*.csv'))
all_data = glob.glob(path)

loaded_data = []
for path in all_data:
    this_data = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    loaded_data.append(this_data)

all_df = pd.concat(loaded_data)
print('done!\nCleaning up and consolidating dataframe object...', end=' ')

og_df = all_df[['Month', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'DepDelay', 'ArrDelay']]

og_df = og_df.sample(frac=0.1)
og_df.dropna(inplace = True)
print('done!\nEncoding dataframe rows with metadata...', end=' ')

og_df['Reporting_Airline'] = og_df['Reporting_Airline'].map(lambda x: get_airline_percent_ontime(x, airlines))
og_df['Origin'] = og_df['Origin'].map(lambda x: get_airport_percent_ontime(x, airports))
#og_df['Dest'] = og_df['Dest'].map(lambda x: get_airport_percent_ontime(x, airports))
print('done!\nSplitting data into training and testing sets...', end=' ')

y = og_df['ArrDelay']
X = og_df.drop(columns=['ArrDelay'])
X['DepDelay'] = X['DepDelay'].map(lambda x: 0 if x < 15 else 1 )

X_train, X_test, y_train, y_test = prepare_data(X, y)
print(X_train[:5])
print('done!\nBuilding and fitting model to training data...', end=' ')

model = build_classification_model(X_train)
history = model.fit(X_train, y_train, epochs=EPOCHS)
print('Evaluating model on test data...', end=' ')

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Model accuracy on test data is {:2.2f}%'.format(100* test_acc))

print('Saving model to json... ', end=' ')
path = os.path.normpath(os.path.join(os.getcwd(), 'processing/v2/meta/model'))
tfjs.converters.save_keras_model(model, path)
print('done!')


print('Finished running in {}'.format(datetime.now() - startTime))