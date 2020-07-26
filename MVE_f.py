#!/usr/bin/env python
# coding: utf-8

# In[256]:


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
from keras.layers import Activation, Dense
import keras.activations
from keras.layers import Input
import statistics as st
from sklearn.utils import resample
from keras import backend as k
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import math
import random 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[257]:




train_x_=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\train_x.csv",header=None)
train_y_=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\train_y.csv",header=None)
test_x_=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\test_x.csv",header=None)
test_y_=pd.read_csv("C:\\Users\\Parichya\\Downloads\\norm_kollam_daily\\test_y.csv",header=None)

print(train_x_.shape)
print(train_y_.shape)
print(test_x_.shape)
print(test_y_.shape)


# In[258]:



whole_x=np.append(train_x_,test_x_,axis=0)
whole_y=np.append(train_y_,test_y_,axis=0)
data=np.append(whole_x,whole_y,axis=1)
train_=np.append(train_x_,train_y_,axis=1)
train_x_=np.asarray(train_x_)
train_y_=np.asarray(train_y_)
test_x_=np.asarray(test_x_)
test_y_=np.asarray(test_y_)
print(data.shape)


# In[259]:


#data=np.append(values_x,values_y,axis=1)
print(data.shape)


# In[260]:


def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


# In[261]:


D1,D2,D3=partition(train_,3)


# In[262]:


print(D3.shape)


# In[263]:


train_yhat=D1

test_yhat=D2

train_x_yhat,train_y_yhat=train_yhat[ : , : -1],train_yhat[ : , -1]

test_x_yhat,test_y_yhat=test_yhat[:,:-1],test_yhat[:,-1]
print(test_y_yhat.shape)


# In[272]:


hidden=20
input_=5
epoch=5000



sgd = SGD(lr = 0.1, momentum = 0.0, decay = 0.0, nesterov=False)
model=Sequential()
model.add(Dense(hidden,input_dim=input_,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
#model.add(Dense(1))
#model.add(Activation("sigmoid"))
#model.add(Activation("elu"))
model.compile(loss='mse',optimizer='sgd')

model.summary()
history=model.fit(train_x_yhat,train_y_yhat,batch_size=len(train_y_yhat),epochs=epoch,validation_split=0,verbose=1,shuffle=False)


# In[273]:


y_hat=model.predict(test_x_yhat)


# In[274]:


print(y_hat.shape)


# In[275]:


train_var=D2
test_var=D2
train_x_var,train_y_var=train_var[ : , : -1],train_var[ : , -1]

#print(train_y_var.shape)

true=np.asarray(train_y_var)
true=true.reshape(-1,1)
print((true.shape))


y=np.asarray(y_hat)

print(y.shape)

#z=[(i - j)**2 for i, j in zip(true.transpose(), np.array(y))]


z=(true-np.array(y))**2
print((np.array(z)).shape)

z=tf.convert_to_tensor(z,dtype='float32',name=None,preferred_dtype=None)
print(z)


# In[268]:


h=np.asarray([[1],[2],[3]])
print(h.shape)
h=tf.convert_to_tensor(h,dtype='float32',name=None,preferred_dtype=None)
q=np.asarray([[4]])
q=tf.convert_to_tensor(q,dtype='float32',name=None,preferred_dtype=None)
l=h/q
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(l))


# In[276]:


def custom_loss(y_true, y_pred):
    
    loss=0.5*tf.reduce_mean((z/y_pred)+tf.log(y_pred))
    #loss=0.5*((z/y_pred)+tf.math.log(y_pred))
    #loss=0.5*((np.array(z)/y_pred)+ math.log(y_pred))
    return loss


# In[277]:


cus=custom_loss(5.0,4.0)

print(cus)

#tf.Print((cus),'float32')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(cus))


# In[279]:


model2=Sequential()
model2.add(Dense(hidden,input_dim=input_,activation='sigmoid'))
model2.add(Dense(1,activation='exponential'))
model2.compile(loss=custom_loss,optimizer='sgd')

model2.summary()
history=model2.fit(train_x_var,train_y_var,batch_size=(len(train_y_var)),epochs=epoch,validation_split=0.,verbose=1,shuffle=False)


# In[278]:


#model2.fit(train_x_var,train_y_var,batch_size=100,epochs=2000,validation_split=0.15,verbose=1,shuffle=False)


# In[280]:


train=D3
train_x,train_y=train[ : , : -1],train[ : , -1]
true_=np.asarray(train_y)
print(len(train_x))


# In[281]:


def custom_loss1(y_true, y_pred):
    out_y=y_pred[:,:-1]
    
    out_var=y_pred[:,-1]
    
   
    #print(out_var.shape)
    #out_y=tf.convert_to_tensor(out_y,dtype='float32',name=None,preferred_dtype=None)
    
    #o=model.predict(train_x)
    
    #print(len(out_y))
    
    #y_=np.asarray(out_y)
    z_=(y_true-out_y)**2    
    
    #z_=tf.convert_to_tensor(z_,dtype='float32',name=None,preferred_dtype=None)
    #print(y_pred.shape)
    #out_var=tf.convert_to_tensor(out_var,dtype='float32',name=None,preferred_dtype=None)
    
    #loss_=(z_/out_var)+tf.math.log(out_var)
    loss=0.5*tf.reduce_mean((z_/out_var)+tf.log(out_var))
    
    return loss


# In[282]:


cus=custom_loss1(tf.convert_to_tensor([[7.0],[8.0]],dtype='float32',name=None,preferred_dtype=None), tf.convert_to_tensor([[8.0,9.0],[10.0,12.0]],dtype='float32',name=None,preferred_dtype=None))

#tf.Print((cus),'float32')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(cus))


# In[283]:


inp = Input(shape=(input_,))
hidden_layer1 = Dense(hidden, activation='sigmoid')(inp)
hidden_layer2 = Dense(hidden, activation='sigmoid')(inp)
output1 = Dense(1, activation='sigmoid')(hidden_layer1)
output2 = Dense(1, activation='exponential')(hidden_layer2)
merge = concatenate([output1, output2])


# In[284]:


model = Model(inputs=inp, outputs=merge)
model.compile(optimizer='sgd',loss=custom_loss1)
model.summary()
model.fit(train_x,train_y,batch_size=len(train_y),epochs=epoch,validation_split=0.15)


# In[285]:


o_=model.predict(test_x_)
print(o_)


# In[286]:


print(len(test_y_))
c=[]
c.append([1,2])
print(c[0])


# In[287]:


lower=[]
upper=[]
result=[]
for i in range(len(o_)):
    lower.append(o_[i][0]-(1.64*math.sqrt(o_[i][1])))
    upper.append(o_[i][0]+(1.64*math.sqrt(o_[i][1])))
for i in range(len(o_)):
    if(lower[i]<0):
        lower[i]=0
    result.append([lower[i],upper[i]])


# In[288]:


#print((result[568][0]))


# In[289]:



sum_=0
sum2=0
for i in range(len(test_y_)):
    
    if(((result[i][1])<test_y_[i]) | ((result[i][0])>test_y_[i])):
        sum_=sum_+0
    else:
        sum_=sum_+1
    sum2=sum2+abs(result[i][1]-result[i][0])
    
PICP=sum_/len(test_y_)
PINAW=sum2/len(test_y_)
print (PICP)
print(PINAW)


# In[290]:


alpha=0.1
mu=1-alpha
eta=15
gamma=0
if (PICP<mu):
    gamma=1
else:
    gamma=0


# In[291]:


CWC = PINAW + len(test_y_)*(gamma*math.exp(-eta*(PICP-mu)))
print(PICP)
print(PINAW)
print(CWC)


# In[292]:


v=(gamma*math.exp(-eta*(PICP-mu)))
print(v)


# In[293]:


plt.plot(result, label='prediction')
plt.plot(test_y_, label='True')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




