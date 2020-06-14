import pandas as pd
import numpy as np
from numpy import argmax
from numpy import concatenate
import keras
from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler



############### for your conveninence ############### 

# define parameters
verbose, epochs, batch_size = 2, 5, 16

#####################################################


print("Data preprocessing...")

# 已知
X_length = 10
Y_length = 6
dimensions = 35

reframed = pd.read_csv('dataset/processed_data.csv')
reframed = reframed.drop(reframed.columns[0], axis=1)


# split into train and test sets
values = reframed.values

# seperate data
totalRows = len(reframed.index)
trainPart = int(totalRows * 4 / 5)
train = values[:trainPart, :]
test = values[trainPart:, :]
X_idx = dimensions*X_length

# split into input and outputs
train_X, train_y = train[:, :X_idx], train[:, X_idx:]
test_X, test_y = test[:, :X_idx], test[:, X_idx:]


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], X_length, dimensions))
test_X = test_X.reshape((test_X.shape[0], X_length, dimensions))
print(train_y.shape)
train_y = train_y.reshape((train_X.shape[0], Y_length, dimensions))
test_y = test_y.reshape((test_X.shape[0], Y_length, dimensions))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


def one_hot_encode(dataIn):
    # remove month and hour data from y data
    # dataIn = dataIn[:,:,2:]
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

train_X = one_hot_encode(train_X)
test_X = one_hot_encode(test_X)
train_y = one_hot_encode(train_y)
test_y = one_hot_encode(test_y)

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#################### Define and Fit Model #################### 


n_timesteps, n_features, n_outputs, y_dimension = train_X.shape[1], train_X.shape[2], Y_length, test_y.shape[2]

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(y_dimension, activation='sigmoid')))
model.compile(loss='mse', optimizer='adam')

print(model.summary())
# learning rate configuration
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)

print('Training...')
# fit network
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                    callbacks=[learning_rate_reduction],verbose=verbose,validation_data=(test_X, test_y),shuffle=False)




#################### Test with the model #################### 
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

# make a prediction
yhat = prediction(model, test_X)
y_true = oneHotDecode(test_y)

Acc_test = (yhat == y_true).astype(int).sum().sum() / yhat.size

# make a prediction
yhat1 = prediction(model, train_X)
y_true1 = oneHotDecode(train_y)

Acc_train = (yhat1 == y_true1).astype(int).sum().sum() / y_true1.size



# rmse = sqrt(mean_squared_error(test_y, yhat))
# print('Test RMSE: %.3f' % rmse)

#################### Plot the result #################### 
# plot accuracy

types = ['Train', 'Test']
accuracy = [Acc_test,Acc_train]
plt.subplots()
# Create bars
plt.barh(types, accuracy)

for index, value in enumerate(accuracy):
    plt.text(value, index, str(value))
 
# Show graphic
plt.savefig('figures/accuracy_basic_LSTM.png')
print("Accuracy result output to figures")
plt.show()



plt.subplots()
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig('figures/loss_basic_LSTM.png')

print("Accuracy result output to figures")
plt.show()


# serialize model to JSON
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/basic_lstm_model.h5")
print("Saved model to disk")