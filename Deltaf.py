#!/usr/bin/env python
# coding: utf-8

# In[225]:


import tensorflow as tf
from scipy import stats
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_array as check_arrays
from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
import statistics as st
from numpy.random import seed
from tensorflow import set_random_seed
from keras import backend as k
from keras import regularizers
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[226]:


seed(1)

set_random_seed(2)


# In[227]:


'''c=[]
k=[]
for i in range(0,2439):
    c.append(i)
for i in range(0,487):
    k.append(i)

'''


train_x=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\train_x.csv",header=None)
train_y=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\train_y.csv",header=None)
test_x=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\test_x.csv",header=None)
test_y=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\test_y.csv",header=None)

values_trx=train_x.values
values_tesx=test_x.values
print(values_trx.shape)
print(values_tesx.shape)
values_x=np.append(values_trx,values_tesx,axis=0)
values_try=train_y.values
print(values_try.shape)
values_tesy=test_y.values
print(values_tesy.shape)
values_y=np.append(values_try,values_tesy,axis=0)
values_y=values_y.reshape(-1,1)
print (values_x.shape)
print (values_y.shape)


# In[228]:


data=np.append(values_x,values_y,axis=1)
print(data.shape)


# In[229]:


len(values_tesy)


# In[259]:


input_=5
hidden=20
epoch=5000
sgd = SGD(lr = 0.01, momentum = 0.0, decay = 0.0, nesterov=False)
model=Sequential()
model.add(Dense(hidden,input_dim=input_,activation='sigmoid',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse',optimizer='sgd')
#model.compile(loss='rmse',optimizer='adam')
model.summary()


# In[260]:


history=model.fit(train_x,train_y,batch_size=len(train_y),epochs=epoch,validation_split=0.15,verbose=1,shuffle=False)


# In[261]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()


# In[262]:


model.predict(test_x)

#print(test_y.dtype)


# In[263]:


print(test_y)


# In[264]:


weights_ij, biases_ij = model.layers[0].get_weights()
weights_j, biases_j = model.layers[1].get_weights()
#weights_ij.reshape(5,10)
weights_ij.reshape(input_,hidden)
print(weights_j)


# In[265]:


weights_ij=np.float64(weights_ij)
biases_ij=np.float64(biases_ij)
weights_j=np.float64(weights_j)
biases_j=np.float64(biases_j)
print(weights_j)


# In[266]:


print(weights_ij.dtype)


# In[267]:




def f_generator(datam):
    F=[]
    for k in range(len(datam)):

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))


        phi=[]
        on_phi=[]
        for i in range(0,hidden):
            sum1=0
            for j in range(0,input_):
                sum1=sum1+((weights_ij[j][i]*datam[k][j]))
            z=sigmoid(sum1+biases_ij[i])
            on_phi.append(z)
            phi.append(z*(1-z))
            sum1=0
        #print (on_phi)

        f=[]
        for j in range(0,input_):
            W_ij=[]
            for i in range(0,hidden):
                W_ij.append(phi[i]*weights_j[i])
            for i in range(0,hidden):
                f.append(W_ij[i]*datam[k][j])

        for j in range(0,hidden):
            f.append(weights_j[j]*phi[j])

        #print ((f))

        for i in range(0,hidden):
            f.append(on_phi[i])
        #print(len(f))    

        f.append(1)

        f=np.asarray(f,dtype='float64')
        f.reshape(-1)
        #print(f.shape)

        F.append(f)
    
    F=np.asarray(F)
    return(F)

train_F=f_generator(train_x)
test_F=f_generator(test_x)
print(train_F)
print(test_F)
    


# In[268]:


lambda_=0.0001
I=np.identity(len(tran_F))
I_=lambda_*I
print(I_)


# In[269]:


tran_F=(np.transpose(train_F))
mult=np.dot(tran_F,train_F)
q=np.linalg.inv(mult+I_)

print(mult.shape)
const1=np.dot((np.dot(q,mult)),q)
print(const1.shape)


# In[270]:


t_val=stats.t.ppf(0.95,(len(test_y)-len(tran_F)))
print(t_val)


# In[271]:


output=model.predict(test_x)


# In[272]:


print(len(test_y))


# In[ ]:





# In[273]:


sum_=0
for i in range(len(test_y)):
    x=train_y[i]-output[i]
    sum_=sum_+(x*x)
var=sum_/len(test_y)
const2=(math.sqrt(var))*t_val
print(const2)


# In[ ]:





# In[274]:


S=[]
for i in range(len(test_y)):
    S.append(np.dot(np.dot(np.transpose(test_F[i]),const1),test_F[i]))
S=np.asarray(S)
S=S.reshape(-1,1)
print(S.shape)
bb=[]
for i in range(len(test_y)):
    print(i)
    hh=math.sqrt(1+S[i])
    print(hh)
    bb.append(hh)
#z=const2*math.sqrt(S[1])


# In[275]:


error=[]
#error.append(math.sqrt(5))
for i in range(len(test_y)):
    #print(len(test_y))
    error.append(const2*bb[i])
    #error.append(const2*(math.sqrt(S[i])))
#print(len(error))
#error=np.asarray(error,dtype='float64')
print(error)


# In[ ]:





# result=[]

# In[276]:


output=np.float64(output)
#print(output)


# In[277]:


plt.plot(output, label='Prediction')
plt.plot(test_y, label='True Value')
plt.legend()
plt.show()


# In[278]:


print(len(error))


# In[280]:


result=[]
for i in range(len(test_x)):
    if(output[i]-error[i]<0):
        result.append([0,output[i]+error[i]])
    else:
        result.append([output[i]-error[i],output[i]+error[i]])
#result=np.asarray(result)
#result.reshape[-1,2]
#result=np.asarray(result)
#print(result)


# In[281]:


df=pd.DataFrame(result,columns=['Lower','Upper'])
df['actual']=test_y
    


# In[282]:


#print(df)  #Prediction Interval printed


# In[283]:


plt.plot(df['Lower'], label='Lower Limit')
plt.plot(df['Upper'], label='Upper Limit')
plt.plot(df['actual'], label='True Value')
plt.legend()
plt.show()


# In[284]:


sum_=0
sum2=0
for i in range(len(test_y)):
    if(((result[i][1])>=test_y[i]) & ((result[i][0])<test_y[i])):
        sum_=sum_+1
    else:
        sum_=sum_+0
    sum2=sum2+abs(result[i][1]-result[i][0])
    
PICP=sum_/len(test_y)
PINAW=sum2/len(test_y)
print(PICP)
print(PINAW)


# In[285]:


alpha=0.05
mu=1-alpha
eta=15
gamma=0
if (PICP<mu):
    gamma=1
else:
    gamma=0


# In[ ]:





# In[ ]:





# In[286]:


print(mu)


# In[287]:


CWC = PINAW + (gamma*len(test_y)*math.exp(-eta*(PICP-mu)))


# In[288]:


print(CWC)


# In[ ]:





# In[ ]:




