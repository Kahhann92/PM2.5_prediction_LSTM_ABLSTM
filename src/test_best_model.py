import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import os

print("Reading data...")
df = pd.read_csv('dataset/processed_data.csv')

df = df.drop(df.columns[0], axis=1)

print("Processing data...")
# initialize
values = df.values
X_length = 10
Y_length = 6
dimensions = 35
totalRows = len(df.index)
X_idx = (X_length*dimensions)
Y_val = np.array(values[:, X_idx:])
Y_val = Y_val.reshape((Y_val.shape[0], Y_length, dimensions))


# load json and create model
json_file = open('model/ABLSTM_best.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ABLSTM_model = model_from_json(loaded_model_json)
# load weights into new model
ABLSTM_model.load_weights('model/ABLSTM_best.h5')
print("Loaded ablstm model from disk")

json_file = open('model/basic_lstm_model_best.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
LSTM_model = model_from_json(loaded_model_json)
# load weights into new model
LSTM_model.load_weights('model/basic_lstm_model_best.h5')
print("Loaded lstm model from disk")

def one_hot_encode(dataIn):
    # remove month and hour data from y data
    # dataIn = dataIn[:,:,:]
    sizeStep =dataIn.shape[0]
    timeStep = dataIn.shape[1]
    dimensionStep = dataIn.shape[2]

    dataIn = dataIn.reshape((sizeStep, timeStep * dimensionStep))

    # seperate to 6 categories
    # dataIn = 0 * ((0 <= dataIn) & (dataIn < 35)).astype(int) \
    dataIn = 1 * ((35 <= dataIn)  & (dataIn < 75)).astype(int) \
        + 2 * ((75 <= dataIn) & (dataIn < 115)).astype(int) \
        + 3 * ((115 <= dataIn) & (dataIn < 150)).astype(int) \
        + 4 * ((150 <= dataIn) & (dataIn < 250)).astype(int) \
        + 5 * (250 <= dataIn).astype(int)

    dataOut = np.zeros((sizeStep, timeStep * dimensionStep * 6))

    # one hot encoding
    for i in range(sizeStep):
        temp = dataIn[i,:]
        temp = to_categorical(temp, num_classes=6)
        dataOut[i,:] = temp.reshape(-1)
    dataOut = dataOut.reshape((sizeStep, timeStep, dimensionStep*6))

    return dataOut

def oneHotDecode(y):

    sizeStep =y.shape[0]
    timeStep = y.shape[1]
    categoryStep = 6
    dimensionStep = int(y.shape[2] / categoryStep)
    
    temp = y.reshape(sizeStep,timeStep,dimensionStep,categoryStep)

    yOut = np.zeros((sizeStep, timeStep, dimensionStep))

    for i in range(sizeStep):
        for j in range(timeStep):
            for k in range(categoryStep):
                yOut[i,j,k] = argmax(temp[i,j,k,:])
    return yOut

def prediction(model, X):
    y = model.predict(X)
    return oneHotDecode(y)


X_, Y_ = values[:, :X_idx], values[:, X_idx:]

X_ = X_.reshape((X_.shape[0], X_length, dimensions))
Y_ = Y_.reshape((Y_.shape[0], Y_length, dimensions))
X = one_hot_encode(X_)
Y = oneHotDecode(one_hot_encode(Y_))

print("Predicting PM2.5 level...")
Y_AB = prediction(ABLSTM_model,X)
Y_LSTM = prediction(LSTM_model,X)



Acc_AB = (Y_AB == Y).astype(int).sum().sum() / Y.size
Acc_LSTM = (Y_LSTM == Y).astype(int).sum().sum() / Y.size



# rmse = sqrt(mean_squared_error(test_y, yhat))
# print('Test RMSE: %.3f' % rmse)

#################### Plot the result #################### 
# plot accuracy

types = ['ABLSTM', 'LSTM']
accuracy = [Acc_AB,Acc_LSTM]
plt.subplots()
# Create bars
plt.barh(types, accuracy)

for index, value in enumerate(accuracy):
    plt.text(value, index, str(value))
 
# Show graphic
plt.savefig('figures/accuracy_LSTM_vs_ABLSTM.png')
print("Accuracy result output to figures")
plt.show()