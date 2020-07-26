#!/usr/bin/env python
# coding: utf-8

# In[68]:


import tensorflow as tf
from scipy import stats
import numpy as np
import pandas as pd
from keras.utils import plot_model
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_array as check_arrays
from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential,Model
from keras.layers.merge import concatenate
from keras.layers import Dense
from keras.layers import LSTM,GRU,Input
import statistics as st
from sklearn.utils import resample
from keras import backend as k
from keras import regularizers
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import math
import random 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:



train_x_=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\train_x.csv",header=None)
train_y_=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\train_y.csv",header=None)
test_x_=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\test_x.csv",header=None)
test_y_=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\test_y.csv",header=None)
print(train_x.shape)
print(train_y.shape)
train_=np.append(train_x,train_y,axis=1)
test_=np.append(test_x,test_y,axis=1)


# In[ ]:





# In[70]:


B=10
def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]
D=partition(train_,B)
D_=partition(test_,B)


# In[71]:


print(len(D_[9]))


# In[72]:


input_=5
hidden=20
epoch=5000
sgd = SGD(lr = 0.2, momentum = 0.0, decay = 0.0, nesterov=False)


# In[73]:


def func(dat,t,tes):
    y_hat=[]
    y_hat_tes=[]
    for i in range(0,B):
        tr = dat[i]
        tr_x,tr_y=tr[ : , : -1],tr[ : , -1]
        
        model=Sequential()
        model.add(Dense(hidden,input_dim=input_,activation='sigmoid',kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dense(1,activation='linear'))
        model.compile(loss='mse',optimizer='sgd')
        
        model.fit(tr_x,tr_y,batch_size=len(tr_y),epochs=epoch,verbose=1,shuffle=False)
        y_hat.append(model.predict(t))
        y_hat_tes.append(model.predict(tes))
        model.reset_states()
        
        
    sum_y=y_hat[0]
    sum_y_tes=y_hat_tes[0]
    for i in range(1,B):
        sum_y+=y_hat[i]
        sum_y_tes+=y_hat_tes[i]
    sum_y=sum_y/B
    sum_y_tes=sum_y_tes/B
    y_hat_=sum_y

    print(sum_y)   
    sigma=[]
    sigma_tes=[]
    for i in range(0,B):
        x=y_hat[i]-y_hat_
        x=x**2
        sigma.append(x)
        x_tes=y_hat_tes[i]-sum_y_tes
        x_tes=x_tes**2
        sigma_tes.append(x_tes)
    sigma_y_squared=sigma[0]
    sigma_tes_sq=sigma_tes[0]
    for i in range(1,B):
        sigma_y_squared+=sigma[i]
        sigma_tes_sq+=sigma_tes[i]
    sigma_y_squared=sigma_y_squared/(B-1)
    sigma_tes_sq=sigma_tes_sq/(B-1)
    return (y_hat_,sigma_y_squared,sum_y_tes,sigma_tes_sq)

        


# In[74]:


#print(y_hat[9]+y_hat[8])
y_hat_tr,sigma_train,y_hat_test,sigma_test=func(D,train_x_,test_x_)

#y_hat_test=func(D_,test_x_)


# In[75]:



print((sigma_test))


# In[ ]:





# In[76]:



print(len(train_x_))


# In[77]:


def z_(dt,y_h,sig):
    dt=np.asarray(dt)
    dt=dt.reshape(-1,1)
    z=(dt-y_h)**2-sig
    for i in range(len(dt)):
        z[i]=max(z[i],0)
    return (z)
r_tr=z_(train_y_,y_hat_tr,sigma_train)
#r_tes=z_(test_y_,y_h_tes,sig_sq_tes)
#print(r_tes)


# In[78]:


def custom_loss(y_true, y_pred):
    
    loss=tf.reduce_mean((y_true/y_pred)+tf.math.log(y_pred))
    
    return loss


# In[79]:


model2=Sequential()
model2.add(Dense(hidden,input_dim=input_,activation='sigmoid'))
model2.add(Dense(1,activation='linear'))
model2.compile(loss=custom_loss,optimizer='sgd')

model2.summary()
history=model2.fit(train_x_,r_tr,batch_size=len(r_tr),epochs=epoch,verbose=1,shuffle=False)


# In[80]:


sig=model2.predict(test_x_)
for i in range(len(sig)):
    sig[i]=math.sqrt(sig[i])
print(sig)


# In[ ]:





# In[81]:


lower=[]
upper=[]
result=[]
y_ha=y_hat_test
for i in range(len(test_y_)):
    
    lower.append(y_ha[i]-(1.64*sig[i]))
    if(lower[i]<0):
        lower[i]=0
    upper.append(y_ha[i]+(1.64*sig[i]))
for i in range(len(y_ha)):
    
    result.append([lower[i],upper[i]])


# In[82]:


result=np.asarray(result)
result=result.reshape(-1,2)
print(result.shape)
test_y_=np.asarray(test_y_)
print((test_y_.shape))


# In[83]:



sum1=0
sum2=0
for i in range(len(test_y_)):
   
    if(((result[i][1])>test_y_[i])  & ((result[i][0])<=test_y_[i])):
        sum1=sum1+1
    else:
        sum1=sum1+0
    sum2=sum2+abs(result[i][1]-result[i][0])
    
PICP=sum1/len(test_y_)
PINAW=sum2/len(test_y_)
print (PICP)
print(PINAW)


# In[84]:


alpha=0.05
mu=1-alpha
eta=50
gamma=0
if (PICP<mu):
    gamma=1
else:
    gamma=0


# In[85]:


result=np.asarray(result)
print(result)


# In[86]:


CWC = PINAW + (gamma*math.exp(-eta*(PICP-mu)))
print(CWC)


# In[87]:


plt.plot(lower, label='lower_prediction')
plt.plot(upper, label='high_prediction')
plt.plot(test_y_, label='True')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




