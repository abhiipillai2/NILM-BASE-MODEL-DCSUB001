from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy as np
import pandas as pd

dataset = pd.read_csv("NILM_001.csv")
print(dataset.head())

x = dataset["dcmain001"].values
y = dataset["dcsub001"].values

xx = np.array(x)
yy = np.array(y)

x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size = 0.15,random_state = 1)

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

X_TRAIN=np.array(sequesnceGenerator(x_train,5))
Y_TRAIN=np.array(sequesnceGenerator(y_train,5))
X_TEST=np.array(sequesnceGenerator(x_test,5))
Y_TEST=np.array(sequesnceGenerator(y_test,5))

model = Sequential()
model.add(Dense(200, input_dim=5, activation='relu'))
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(5, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

#model.fit_generator(dataLoader(xx,yy,5), epochs = 10)
history = model.fit(X_TRAIN, Y_TRAIN, epochs=10, batch_size=5,validation_split=0.15,validation_data=None,verbose=1)

file_name = "NILM_BASE_MODEL.h5"
model.save(file_name)

print(model.predict(X_TEST,batch_size=5))
print("---------------------------------------")
print(Y_TEST)


# plot metrics
pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['accuracy'])
# pyplot.plot(history.history['mean_absolute_error'])
#pyplot.plot(history.history['mean_absolute_percentage_error'])
# pyplot.plot(history.history['cosine_proximity'])
pyplot.show()

#score = model.evaluate(X_test, y_test, verbose = 0) 

# print('Test loss:', score[0]) 
# print('Test accuracy:', score[1])