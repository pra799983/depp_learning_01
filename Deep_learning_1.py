#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[6]:


get_ipython().system('pip install tensorflow')


# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[3]:


(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()


# In[4]:


len(x_train)


# In[5]:


len(x_test)


# In[6]:


x_train[0].shape


# In[12]:


x_train[0]


# In[18]:


x_train = x_train/255
x_test = x_test/255


# In[19]:


x_train[0]


# In[20]:


plt.matshow(x_train[5])


# In[21]:


plt.matshow(x_train[2])


# In[9]:


plt.matshow(x_train[3])


# In[22]:


plt.matshow(x_train[7
                   ])


# In[26]:


y_train[2]


# In[24]:


y_train[:5]


# In[25]:


y_train[:]


# In[27]:


x_train.shape


# In[51]:


x_train_flattened = x_train.reshape(len(x_train),28*28)
x_train_flattened.shape


# In[52]:


x_test_flattened = x_test.reshape(len(x_test),28*28)
x_test_flattened.shape


# In[53]:


x_train_flattened[0]


# In[54]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,) , activation = 'sigmoid')
])
model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])
model.fit(x_train_flattened,y_train,epochs = 5)


# model.evaluate(x_test_flattened, y_test)

# In[55]:


model.evaluate(x_test_flattened, y_test)


# In[56]:


plt.matshow(x_test[0])


# In[57]:


y_predicted = model.predict(x_test_flattened)
y_predicted[0]


# In[58]:


np.argmax(y_predicted[0])


# In[41]:


np.argmax(y_predicted[1])


# In[59]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[60]:


y_predicted_labels[:5]


# In[61]:


cm = tf.math.confusion_matrix(labels = y_test,predictions = y_predicted_labels)
cm


# In[62]:


import seaborn as sn
plt.figure(figsize= (10,7))
sn.heatmap(cm,annot = True,fmt = 'd')
plt.xlabel('predicted')
plt.ylabel('Truth')


# # Using hidden layer 

# In[64]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)


# In[66]:


y_predicted = model.predict(x_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # Using  Flatten layer so that we don't have to call .reshape on input dataset

# In[68]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)


# In[70]:


model.evaluate(x_test,y_test)


# In[ ]:




