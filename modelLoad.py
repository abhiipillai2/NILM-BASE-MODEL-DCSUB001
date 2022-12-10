import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("NILM_001.csv")
print(dataset.head())

x = dataset["dcmain001"].values
y = dataset["dcsub001"].values

xx = np.array(x)
yy = np.array(y)

xx = xx.array.flatten()
yy = xx.array.flatten()

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

X=np.array(sequesnceGenerator(xx,100))

#arr=np.array([[514,301,438,592,559]])
arr=np.array([[372,511,552,359,496,582,550,490]])

#model_name = "testModelTelivision.h5"
model_name = "NILM_BASE_MODEL.h5"

loded_model = load_model(model_name)

predicted = loded_model(X)
array = predicted.numpy()
result = array.flatten()

time=[]
for i in range(len(result)):

	time.append(i)

count = 448300

# # xx = np.delete(xx, range(count,4344))
# yy = np.delete(yy, range(count,448325))
# time = np.delete(time, range(count,448325))
# # result=np.delete(result, range(count,4336))

print(len(time))
print(len(xx))
print(len(yy))
print(len(result))

#plt.plot(time,xx ,label = "mains reading")
plt.plot(time,result ,label = "model predicted value of refrigerator")
plt.plot(time,yy,label = "actual value of refrigerator")
plt.legend()
plt.show()

