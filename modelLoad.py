import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

seq_val = 10

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

X=np.array(sequesnceGenerator(xx,seq_val))

#model_name = "testModelTelivision.h5"
model_name = "NILM_BASE_MODEL.h5"

loded_model = load_model(model_name)

predicted = loded_model(X)
array = predicted.numpy()
result = array.flatten()

time=[]
for i in range(len(result)):

	time.append(i)

count = 500

# # xx = np.delete(xx, range(count,4344))
yy = np.delete(yy, range(count,246423))
time = np.delete(time, range(count,246420))
result=np.delete(result, range(count,246420))

print(len(yy))
print(len(result))

#plt.plot(time,xx ,label = "mains reading")
plt.plot(time,yy,label = "actual value of refrigerator")
plt.plot(time,result ,label = "model predicted value of refrigerator")
plt.legend()
plt.show()

