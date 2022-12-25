import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np
import pandas as pd
  
#reading and assigning index as timestamp
datasetDcmain001 = pd.read_csv("dcmain001.csv",usecols=[2,3], names=['power', 'time_stamp'] , parse_dates=["time_stamp"], index_col="time_stamp")

datasetDcmain001 = datasetDcmain001.resample('8s').mean()

#reading and assigning index as timestamp
datasetDcsub001 = pd.read_csv("dcsub001.csv",usecols=[2,3], names=['power', 'time_stamp'] , parse_dates=["time_stamp"], index_col="time_stamp")

#resampling the data set
datasetDcsub001 = datasetDcsub001.resample('8s').mean()

mainDf = pd.merge(datasetDcmain001, datasetDcsub001, on="time_stamp")
mainDf = mainDf.dropna()
#mainDf = datasetDcmain001.merge(datasetDcsub001,how = 'inner',left_index = True, right_index =True)

x = mainDf["power_x"].values
y = mainDf["power_y"].values

xx = np.array(x)
yy = np.array(y)

x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size = 0.20,random_state = 1)

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

X_TRAIN=np.array(sequesnceGenerator(x_train,100))
Y_TRAIN=np.array(sequesnceGenerator(y_train,100))
X_TEST=np.array(sequesnceGenerator(x_test,100))
Y_TEST=np.array(sequesnceGenerator(y_test,100))

# cnn = models.Sequential([
#     layers.Conv1D(filters=64, kernel_size=1, activation='relu',input_shape=(100,1)),
#     layers.Dropout(0.5),

#     layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
#     layers.Conv1D(filters=16, kernel_size=1, activation='relu'),
#     layers.MaxPooling1D(pool_size=1, name="MaxPooling1D"),

#     layers.Flatten(),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(100, activation='linear')
# ])

cnn = models.Sequential([
    layers.Conv1D(filters=34, kernel_size=10, activation='relu',input_shape=(100,1)),
    layers.Dropout(0.5),

    layers.Conv1D(filters=30, kernel_size=8, activation='relu'),
    layers.Conv1D(filters=40, kernel_size=6, activation='relu'),
    layers.Conv1D(filters=50, kernel_size=5, activation='relu'),
    layers.Conv1D(filters=50, kernel_size=5, activation='relu'),
    layers.Conv1D(filters=50, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=1, name="MaxPooling1D"),

    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(100, activation='sigmoid')
])


optimizer = tf.keras.optimizers.RMSprop(0.001)

cnn.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

#model.fit_generator(dataLoader(xx,yy,5), epochs = 10)
history = cnn.fit(X_TRAIN, Y_TRAIN, epochs=20, batch_size=8,validation_split=0.15,validation_data=None,verbose=1)

file_name = "NILM_BASE_MODEL.h5"
cnn.save(file_name)

#predict = model.predict(X_TEST,batch_size=100)
predict = cnn.predict(X_TEST,batch_size=100)
print("---------------------------------------")
print(Y_TEST)


# plot metrics
#pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(predict)
pyplot.plot(Y_TEST)
#pyplot.plot(history.history['accuracy'])
# pyplot.plot(history.history['mean_absolute_error'])
#pyplot.plot(history.history['mean_absolute_percentage_error'])
# pyplot.plot(history.history['cosine_proximity'])
pyplot.show()

#score = model.evaluate(X_test, y_test, verbose = 0) 

# print('Test loss:', score[0]) 
# print('Test accuracy:', score[1])