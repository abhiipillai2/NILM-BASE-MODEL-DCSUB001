from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy as np
import pandas as pd
  
#reading and assigning index as timestamp
datasetDcmain001 = pd.read_csv("dcmain001.csv",usecols=[2,3], names=['power', 'time_stamp'] , parse_dates=["time_stamp"], index_col="time_stamp")
print(datasetDcmain001.head())
datasetDcmain001 = datasetDcmain001.resample('10s').mean()

#reading and assigning index as timestamp
datasetDcsub001 = pd.read_csv("dcsub001.csv",usecols=[2,3], names=['power', 'time_stamp'] , parse_dates=["time_stamp"], index_col="time_stamp")

#resampling the data set
datasetDcsub001 = datasetDcsub001.resample('10s').mean()

mainDf = pd.merge(datasetDcmain001, datasetDcsub001, on="time_stamp")
mainDf = mainDf.dropna()
#mainDf = datasetDcmain001.merge(datasetDcsub001,how = 'inner',left_index = True, right_index =True)

print(mainDf.head())

x = mainDf["power_x"].values
y = mainDf["power_y"].values

xx = np.array(x)
yy = np.array(y)

seq_val= 100

x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size = 0.20,shuffle=False)

def sequesnceGenerator(arr,n):
    i=0
    arr1 = []
    temp = []
    while(i < len(arr)):
        if(i%n == 0 and i != 0):
            arr1.append(temp)
            temp = [arr[i]]

        else:
            temp.append(arr[i])
        i+=1
    #arr1.append(temp)
    return arr1

X_TRAIN=np.array(sequesnceGenerator(x_train,seq_val))
Y_TRAIN=np.array(sequesnceGenerator(y_train,seq_val))
X_TEST=np.array(sequesnceGenerator(x_test,seq_val))
Y_TEST=np.array(sequesnceGenerator(y_test,seq_val))


model = Sequential()
model.add(Dense(500, input_dim=seq_val, activation='relu'))
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(seq_val, activation='linear'))

#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.Adam(lr=.01), # SGD is short for stochastic gradient descent
              metrics=["mae"])

#model.fit_generator(dataLoader(xx,yy,5), epochs = 10)
history = model.fit(X_TRAIN, Y_TRAIN, epochs=1000, batch_size=seq_val,validation_split=0.20,validation_data=None,verbose=1)

file_name = "NILM_BASE_MODEL.h5"
model.save(file_name)

predict = model.predict(X_TEST,batch_size=seq_val)
print("---------------------------------------")
print(Y_TEST)


# plot metrics
#pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(predict[67:100])
pyplot.plot(Y_TEST[67:100])
#pyplot.plot(history.history['accuracy'])
# pyplot.plot(history.history['mean_absolute_error'])
#pyplot.plot(history.history['mean_absolute_percentage_error'])
# pyplot.plot(history.history['cosine_proximity'])
pyplot.show()

#score = model.evaluate(X_test, y_test, verbose = 0) 

# print('Test loss:', score[0]) 
# print('Test accuracy:', score[1])