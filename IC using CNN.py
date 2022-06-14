#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models


# In[2]:


datasets.cifar10.load_data()


# In[3]:


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train.shape


# In[6]:


plt.imshow(X_train[0])


# In[7]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[8]:


X_train = X_train/255
y_train = y_train/255


# In[14]:


cnn = models.Sequential([
    #cnn
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters=32, kernel_size =(3,3), activation ='relu'),
    layers.MaxPooling2D((2,2)),
    
    #dense
    layers.Flatten(),
    layers.Dense(62, activation = "relu"),
    layers.Dense(10, activation = "softmax")
])


# In[15]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[16]:


cnn.fit(X_train, y_train, epochs=10)


# In[17]:


cnn.evaluate(X_test, y_test)


# In[ ]:




