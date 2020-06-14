import pandas as pd
import numpy as np
import keras
from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
from numpy import concatenate  # 数组拼接
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler


############### for your conveninence ############### 

# define parameters
verbose, epochs, batch_size = 2, 5, 16

#####################################################




print("Data preprocessing...")
#################### Data Preprocessing #################### 



#change Y from values to categories
def process_Y(dataIn):

    # seperate to 6 categories
    dataIn = ((0 <= dataIn) & (dataIn < 35)).astype(int) \
        + 2 * ((35 <= dataIn)  & (dataIn < 75)).astype(int) \
        + 3 * ((75 <= dataIn) & (dataIn < 115)).astype(int) \
        + 4 * ((115 <= dataIn) & (dataIn < 150)).astype(int) \
        + 5 * ((150 <= dataIn) & (dataIn < 250)).astype(int) \
        + 6 * (250 <= dataIn).astype(int)

    return dataIn


# 已知
X_length = 10
Y_length = 6
dimensions = 37

reframed = pd.read_csv('dataset/processed_data_with_time.csv')
reframed = reframed.drop(reframed.columns[0], axis=1)


# split into train and test sets
values = reframed.values

# normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
scaler = values.max()
scaled = values / scaler

# seperate data
totalRows = len(reframed.index)
trainPart = int(totalRows * 4 / 5)
train = scaled[:trainPart, :]
test = scaled[trainPart:, :]
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

train_y = train_y[:,:,2:]
test_y = test_y[:,:,2:]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#################### Define and Fit Model #################### 

# define parameters
n_timesteps, n_features, n_outputs, y_dimension = train_X.shape[1], train_X.shape[2], Y_length, test_y.shape[2]

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(y_dimension,activation = 'sigmoid')))
model.compile(loss='mse', optimizer='adam')
print(model.summary)
# learning rate configuration
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
plot_model(model, to_file='loss_after_categorize.png', show_shapes=True, show_layer_names=True)
# fit network
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                    callbacks=[learning_rate_reduction],verbose=verbose,validation_data=(test_X, test_y),shuffle=False)






#################### Test with the model #################### 

# make a prediction
yhat = model.predict(test_X)

# inverse scaler
yhat = yhat * scaler
y_true = test_y * scaler

yhat = process_Y(yhat)
y_true = process_Y(y_true)

Acc_test = (yhat == y_true).astype(int).sum().sum() / yhat.size

# make a prediction
yhat1 = model.predict(train_X)
# inverse scaler
yhat1 = yhat1 * scaler
y_true1 = train_y * scaler

yhat1 = process_Y(yhat1)
y_true1 = process_Y(y_true1)

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
# plt.show()
plt.savefig('figures/accuracy_after_categorize.png')
print("Accuracy result output to figures")
plt.show()



plt.subplots()

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig('figures/loss_after_categorize.png')

print("Accuracy result output to figures")
plt.show()