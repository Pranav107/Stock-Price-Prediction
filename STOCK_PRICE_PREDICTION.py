#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np


# In[111]:


df = pd.read_csv(r"C:\Users\Manas Ranjan Kar\Downloads\AMZN (1).csv",index_col = 'Date')
df.head()


# In[112]:


df1=df.reset_index()['Close']
df1


# In[113]:


import matplotlib.pyplot as plt
plt.figure(figsize = (14,8))
plt.plot(df1)


# In[114]:


### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 


# In[115]:


df1


# In[116]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[117]:


print(df1)


# In[118]:


##splitting dataset into train and test split


# In[119]:


training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[120]:


training_size,test_size


# In[121]:


train_data


# In[122]:


test_data


# In[123]:


import numpy
# convert an array of values into a dataset matrix

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[124]:


time_step = 150
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[125]:


print(X_train.shape), print(y_train.shape)


# In[126]:


print(X_test.shape), print(ytest.shape)


# In[127]:


# reshape input to be [samples, time steps, features] which is required for LSTM

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[128]:


## Create the Stacked LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[129]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(150,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[130]:


model.summary()


# In[131]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=150,batch_size=64,verbose=1)


# In[132]:


import tensorflow as tf


# In[133]:


tf.__version__


# In[134]:


### Lets Do the prediction and check performance metrics

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[135]:


##Transformback to original form

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[136]:


### Calculate RMSE performance metrics

import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[137]:


### Test Data RMSE

math.sqrt(mean_squared_error(ytest,test_predict))


# In[138]:


### Plotting 

# shift train predictions for plotting

plt.figure(figsize = (16,8))

look_back=150
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting

testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[139]:


len(test_data)


# In[140]:


x_input=test_data[411:].reshape(1,-1)
x_input.shape


# In[141]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[142]:


temp_input


# In[143]:


# demonstrate prediction for next 10 days

from numpy import array

lst_output=[]
n_steps=150
i=0
while(i<80):
    
    if(len(temp_input)>150):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[144]:


day_new=np.arange(1,151)
day_pred=np.arange(151,231)


# In[145]:


import matplotlib.pyplot as plt


# In[146]:


len(df1)


# In[155]:


plt.figure(figsize = (16,8))
plt.plot(day_new,scaler.inverse_transform(df1[1718:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[154]:


df3=df1.tolist()
df3.extend(lst_output)
plt.figure(figsize = (16,8))
plt.plot(df3[1500:])


# In[151]:


df3=scaler.inverse_transform(df3).tolist()


# In[156]:


plt.figure(figsize = (16,8))
plt.plot(df3)


# In[ ]:





# In[ ]:




