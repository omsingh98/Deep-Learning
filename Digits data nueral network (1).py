#!/usr/bin/env python
# coding: utf-8

# In[19]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[3]:


len(X_train)


# In[4]:


X_train[0].shape


# In[5]:


X_train[0]


# In[6]:


plt.matshow(X_train[0])


# In[7]:


X_train.shape


# In[8]:


X_train = X_train/255
X_test = X_test/255


# In[9]:


X_test[0]


# In[10]:


X_train_flattened = X_train.reshape(len(X_train),28*28)
X_train_flattened.shape
X_test_flattened = X_test.reshape(len(X_test),28*28)
X_test_flattened.shape


# In[11]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"])

model.fit(X_train_flattened, y_train, epochs = 10)


# In[12]:


model.evaluate(X_test_flattened, y_test)


# In[13]:


plt.matshow(X_test[0])


# In[14]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[15]:


np.argmax(y_predicted[0])


# In[16]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,),activation="relu"),
    keras.layers.Dense(10, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"])

model.fit(X_train_flattened, y_train, epochs = 5)


# In[17]:


model.evaluate(X_test_flattened, y_test)


# In[21]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28,28)),
    keras.layers.Dense(100, input_shape=(784,),activation="tanh"),
    keras.layers.Dense(10, activation="sigmoid")
])

tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)
model.compile(
    optimizer="adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"])

model.fit(X_train, y_train, epochs = 5)


# In[22]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# In[ ]:




