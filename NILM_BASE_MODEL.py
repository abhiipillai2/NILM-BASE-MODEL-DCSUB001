from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy as np
import pandas as pd

#reading and assigning index as timestamp
datasetDcmain001 = pd.read_csv("dcmain001.csv",usecols=[2,3], names=['power', 'time_stamp'] , parse_dates=["time_stamp"], index_col="time_stamp")
#print(datasetDcmain001.index)

datasetDcmain001.loc[:,'power'] = datasetDcmain001['power'].resample('8s').mean()

#reading and assigning index as timestamp
datasetDcsub001 = pd.read_csv("dcsub001.csv",usecols=[2,3], names=['power', 'time_stamp'] , parse_dates=["time_stamp"], index_col="time_stamp")
#print(datasetDcmain001.index)

datasetDcsub001['power'] = datasetDcsub001['power'].resample('8s').mean()

print(len(datasetDcsub001))
print(len(datasetDcmain001))

#mainDf = pd.merge(datasetDcmain001, datasetDcsub001, on="time_stamp")
mainDf = datasetDcmain001.merge(datasetDcsub001,how = 'inner',left_index = True, right_index =True)
print(mainDf.head())

x = mainDf["power_x"].values
y = mainDf["power_y"].values

xx = np.array(x)
yy = np.array(y)

x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size = 0.20,random_state = 1)


print(mainDf,datasetDcmain001,)
# def sequesnceGenerator(arr,n):
#     i=0
#     arr1 = []
#     temp = []
#     while(i < len(arr)):
#         if(i%n == 0 and i != 0):
#             arr1.append(temp)
#             temp = [arr[i]]
#         else:
#             temp.append(arr[i])
#         i+=1
#     #arr1.append(temp)
#     return arr1

# X_TRAIN=np.array(sequesnceGenerator(x_train,100))
# Y_TRAIN=np.array(sequesnceGenerator(y_train,100))
# X_TEST=np.array(sequesnceGenerator(x_test,100))
# Y_TEST=np.array(sequesnceGenerator(y_test,100))

# model = Sequential()
# model.add(Dense(200, input_dim=100, activation='relu'))
# model.add(Dense(200, input_dim=200, activation='relu'))
# model.add(Dense(200, input_dim=200, activation='relu'))
# model.add(Dense(100, activation='linear'))

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# #model.fit_generator(dataLoader(xx,yy,5), epochs = 10)
# history = model.fit(X_TRAIN, Y_TRAIN, epochs=10, batch_size=100,validation_split=0.15,validation_data=None,verbose=1)

# file_name = "NILM_BASE_MODEL.h5"
# model.save(file_name)

# print(model.predict(X_TEST,batch_size=100))
# print("---------------------------------------")
# print(Y_TEST)


# # plot metrics
# pyplot.plot(history.history['mean_squared_error'])
# #pyplot.plot(history.history['accuracy'])
# # pyplot.plot(history.history['mean_absolute_error'])
# #pyplot.plot(history.history['mean_absolute_percentage_error'])
# # pyplot.plot(history.history['cosine_proximity'])
# pyplot.show()

# #score = model.evaluate(X_test, y_test, verbose = 0) 

# # print('Test loss:', score[0]) 
# # print('Test accuracy:', score[1])