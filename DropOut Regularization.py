#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("sonar_dataset.csv", header = None)
df.head()


# In[3]:


df.shape


# In[4]:


df.isna().sum()


# In[5]:


df.columns


# In[6]:


df[60].value_counts()


# In[8]:


X = df.drop(60, axis="columns")
y = df[60]
y.head()


# In[9]:


y = pd.get_dummies(y , drop_first=True)
y.sample(5)


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state =1)


# In[11]:


X_train.shape


# In[12]:


import tensorflow as tf
from tensorflow import keras


# model = keras.Sequential([
# 

# In[15]:


model = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation = "relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(15, activation= "relu"),
    keras.layers.Dense(1, activation= "sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer ="adam", metrics= ["accuracy"])

model.fit(X_train, y_train, epochs = 100, batch_size=8)


# In[16]:


model.evaluate(X_test, y_test)


# In[17]:


model = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation = "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation= "relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation= "sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer ="adam", metrics= ["accuracy"])

model.fit(X_train, y_train, epochs = 100, batch_size=8)


# In[18]:


model.evaluate(X_test,y_test)


# In[ ]:




