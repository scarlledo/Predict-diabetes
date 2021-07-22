#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data= pd.read_csv('data.csv')


# In[3]:


dir(pd)


# In[4]:


data.isnull().sum()


# In[5]:


data=data.dropna()


# In[6]:


data.shape 


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


sns.pairplot(data)


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


for i in data.columns:
    print (i)


# In[11]:


x=data.drop('Outcome',axis=1).values
y=data['Outcome'].values


# In[12]:


x.shape,y.shape


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True)


# In[14]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


log=LogisticRegression()


# In[17]:


log.fit(x_train,y_train)


# In[18]:


pred = log.predict(x_test)


# In[19]:


pred


# In[20]:


from sklearn.metrics import accuracy_score


# In[21]:


accuracy_score(y_test,pred)


# In[22]:


plt.plot(pred,'--r')
plt.plot(y_test,'-b')
plt.title('Predicted vs Actual')


# In[23]:


name = input('Enter your name :: ')
preg = input('Enter number of preg :: ')
glu = input('Enter your glo :: ')
bp = input('Enter you bp :: ')
st = input('Enter your st :: ')
ins = input('Enter your ins :: ')
bmi = input('Enter your bmi :: ')
dpf = input('Enter your dpf :: ')
age = input('Enter your age :: ')


# In[24]:


import numpy as np


# In[28]:


person=pd.DataFrame([[preg, glu, bp, st, ins, bmi, dpf, age]])


# In[29]:


ot = log.predict(person)
if ot == 1:
    print ("{} have".format(name))
else:
    print ("{} do not have".format(name))


# In[ ]:




