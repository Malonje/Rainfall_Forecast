#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_array as check_arrays
from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras import regularizers
from sklearn.utils import resample
from keras.optimizers import SGD
import math,random
from sklearn.model_selection import KFold

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_x=pd.read_csv("/Users/swagotoroy/Documents/INDICES/kollam/daily/5_features/train_x_5.csv",names = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10'])
train_y=pd.read_csv("/Users/swagotoroy/Documents/INDICES/kollam/daily/5_features/train_y.csv",header=None)
test_x=pd.read_csv("/Users/swagotoroy/Documents/INDICES/kollam/daily/5_features/test_x_5.csv",names = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10'])
test_y=pd.read_csv("/Users/swagotoroy/Documents/INDICES/kollam/daily/5_features/test_y.csv",header=None)

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


# In[4]:


print (data.shape)

values=data

#data=data.astype('float32')
#scaler=MinMaxScaler()
#scaled_data=normalize(data)
mean=np.mean(values,axis=0)
std=np.std(values,axis=0)

#print(values)

#norm_values=[]
#np.asarray(norm_values)
#for i in range(0,6):
#    for j in range(0,1708):
#        values[j][i]=(values[j][i]-mean[i])/std[i]
        
#print(np.amax(values,axis=0))

scale_data=values
whole_x,whole_y=scale_data[:,:-1],scale_data[:,-1]
tra=scale_data[:math.ceil(0.75*len(scale_data))]
tra_x,tra_y=tra[:,:-1],tra[:,-1]
te=scale_data[math.ceil(0.75*len(scale_data)):]
tes_x,tes_y=te[:,:-1],te[:,-1]

print(mean)
print(std)


# In[21]:


print (data.shape)

values=data

#data=data.astype('float32')
#scaler=MinMaxScaler()
#scaled_data=normalize(data)
mean=np.mean(values,axis=0)
std=np.std(values,axis=0)

#print(values)

#norm_values=[]
#np.asarray(norm_values)
#for i in range(0,6):
#    for j in range(0,1708):
#        values[j][i]=(values[j][i]-mean[i])/std[i]
        
#print(np.amax(values,axis=0))
scale_data=values
whole_x,whole_y=scale_data[:,:-1],scale_data[:,-1]
tra=scale_data[:math.ceil(0.75*len(scale_data))]
tra_x,tra_y=tra[:,:-1],tra[:,-1]
te=scale_data[math.ceil(0.75*len(scale_data)):]
tes_x,tes_y=te[:,:-1],te[:,-1]

print(mean)
print(std)


# In[22]:


def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
def nn(weights_ij,weights_j,tr):
    o=[]
    for j in range(len(tr)):

        xx=np.dot(weights_ij[:,:-1],tr[j])
        xx=xx+weights_ij[:,-1]
        xx=np.asarray(xx)
        #print(o)
        for i in range(len(xx)):
            xx[i]=sigmoid(xx[i])
        #print(o)

        x1=np.dot(weights_j[:,:-1],xx)
        x1=x1+weights_j[:,-1]
        o.append(x1)
    o=np.asarray(o)
    return (o)



    
def cwc(ou,truth):
    sum1=0
    sum2=0
    #print(ou)
    for i in range(len(ou)):
        if((max(ou[i])<truth[i]) | (min(ou[i])>truth[i])):
            sum1=sum1+0
        else:
            sum1=sum1+1
        sum2=sum2+abs(ou[i][1]-ou[i][0])
    global_max=max(ou[:,:-1])
    global_min=min(ou[:,-1])
    r=global_max-global_min
    PICP=sum1/len(truth)
    PINAW=sum2/len(truth)
    PINAW=PINAW/r

    alpha=0.05
    mu=1-alpha
    eta=50
    gamma=0
    if(PICP>=mu):
        gamma=0
    else:
        gamma=1

    k=(gamma*math.exp(-1*eta*(PICP-mu)))
    CWC = PINAW + k
    
    return (CWC,PICP,PINAW)


# In[23]:




weights_hidden_opt=[]
for i in range(0,220):
    temp=np.random.uniform(-1,1)
    weights_hidden_opt.append(temp)
weights_hidden_opt=np.asarray(weights_hidden_opt)
weights_hidden_opt=weights_hidden_opt.reshape(20,11)
weights_opt=[]
for i in range(0,42):
    temp=np.random.uniform(-1,1)
    weights_opt.append(temp)
weights_opt=np.asarray(weights_opt)
weights_opt=weights_opt.reshape(2,21)

print(weights_opt)


# In[12]:


def cross_out(x_train,y_train,x_test,y_test,weights_hidden_opt,weights_opt):
    
    T_opt=5
    CWC_low=0.0001
    T_low=0.00005
    c=0
    k=1
    
    
    o_opt=nn(weights_hidden_opt,weights_opt,x_train)
    
    
    
    

    #CWC_optl=cwc(o_opt,y_train)
    CWC_opt,PICP_opt,PINAW_opt=cwc(o_opt,y_train)
    
    
    CWC_prev=CWC_opt
    PICP_prev=PICP_opt
    PINAW_prev=PINAW_opt
    weights_hidden_new=np.empty_like (weights_hidden_opt)
    weights_new=np.empty_like (weights_opt)
    
    for i in range(0,5000):
        print(i)
        T_new=0.9*T_opt
        weights_hidden_new[:] = weights_hidden_opt
        weights_new[:] = weights_opt

        pos=random.sample(range(0,260), 1)
        pos=np.asarray(pos)
        if(pos<=219):
            pos_x=int(pos/11)
            pos_y=pos-(11*pos_x)
            temp=np.random.uniform(0,1)
            if(temp>0.5):
                v=np.random.uniform(0,0.01)
                if((weights_hidden_new[pos_x][pos_y]+v)<-1):
                    weights_hidden_new[pos_x][pos_y]=-1
                if((weights_hidden_new[pos_x][pos_y]+v)>1):
                    weights_hidden_new[pos_x][pos_y]=1
                else:
                     weights_hidden_new[pos_x][pos_y]+=v
            else:
                v=np.random.uniform(0,0.01)
                if((weights_hidden_new[pos_x][pos_y]-v)<-1):
                    weights_hidden_new[pos_x][pos_y]=-1
                if((weights_hidden_new[pos_x][pos_y]-v)>1):
                    weights_hidden_new[pos_x][pos_y]=1
                else:
                     weights_hidden_new[pos_x][pos_y]-=v

        else:
            pos=pos-219
            pos_x=int(pos/21)
            pos_y=pos-(21*pos_x)
            temp=np.random.uniform(0,1)
            if(temp>0.5):
                v=np.random.uniform(0,0.01)
                if((weights_new[pos_x][pos_y]+v)<-1):
                    weights_new[pos_x][pos_y]=-1
                if((weights_new[pos_x][pos_y]+v)>1):
                    weights_new[pos_x][pos_y]=1
                else:
                     weights_new[pos_x][pos_y]+=v


            else:
                v=np.random.uniform(0,0.01)
                if((weights_new[pos_x][pos_y]-v)<-1):
                    weights_new[pos_x][pos_y]=-1
                if((weights_new[pos_x][pos_y]-v)>1):
                    weights_new[pos_x][pos_y]=1
                else:
                     weights_new[pos_x][pos_y]-=v




        out=nn(weights_hidden_new,weights_new,x_train)
        #print(o)
        #CWC_newl=cwc(out,y_train)
        CWC_new,PICP_new,PINAW_new=cwc(out,y_train)
        CWC_curr=CWC_new
        PICP_curr=PICP_new
        PINAW_curr=PINAW_new
        #print(CWC_new)
        

        if(CWC_new<CWC_opt):
            #print(22)
            CWC_opt=CWC_new
            PICP_opt=PICP_new
            PINAW_opt=PINAW_new
            weights_hidden_opt=weights_hidden_new
            weights_opt=weights_new
        else :
            #print(1)
            r=np.random.uniform(0,1)
            z=math.exp(-(CWC_new-CWC_opt)/(k*T_new))
            #print(CWC_new)
            #print(CWC_opt)
            if(r>=z):
                CWC_opt=CWC_new
                weights_hidden_opt=weights_hidden_new
                weights_opt=weights_new
            

        #print(CWC_new)        
        T_opt=T_new
        CWC_prev=CWC_curr
        
    return(weights_hidden_opt,weights_opt)


# In[24]:


kf = KFold(n_splits=5)  
kf.get_n_splits(tra_x) 
print(kf)
cwc_list=[]
weighthid_f=[]
weights_f=[]
for train_index, test_index in kf.split(tra_x):
    X_train, X_test = tra_x[train_index], tra_x[test_index]
    Y_train, Y_test = tra_y[train_index], tra_y[test_index]
    weights_hidden_opt,weights_opt=(cross_out(X_train,Y_train,X_test,Y_test,weights_hidden_opt,weights_opt))
    weighthid_f.append(weights_hidden_opt)
    weights_f.append(weights_opt)
    
     
    
    


# In[25]:


weighthid_f=np.asarray(weighthid_f)

print(weights_f)
print(weights_opt)


# In[26]:


final_o=nn(weights_hidden_opt,weights_opt,tes_x)
cwcf,picpf,pinawf=cwc(final_o,tes_y)
print(cwcf)
print(picpf)
print(pinawf)


# In[16]:


plt.plot(final_o, label='prediction')
plt.plot(tes_y, label='True')
plt.legend()
plt.show()


# In[ ]:




    


# In[12]:



plt.plot(weighthid_f[0], label='hidden-layer')
plt.plot(weighthid_f[1], label='hidden-layer')
plt.plot(weighthid_f[2], label='hidden-layer')
plt.plot(weighthid_f[3], label='hidden-layer')
plt.plot(weighthid_f[4], label='hidden-layer')
#plt.plot(test_y, label='True Value')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[359]:





# In[ ]:





# In[ ]:





# In[362]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




