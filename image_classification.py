#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


tf.__version__


# In[3]:


#Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[4]:


#Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[5]:


#Building the CNN
#Initialising the CNN
cnn = tf.keras.models.Sequential()


# In[6]:


#Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# In[7]:


#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[8]:


#Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[9]:


#Flattening
cnn.add(tf.keras.layers.Flatten())


# In[10]:


#Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[11]:


#Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[12]:


#Training the CNN
#Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[13]:


#Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# In[17]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
print(prediction)


# In[24]:


from IPython.display import Image
Image('dataset/single_prediction/cat_or_dog_1.jpg')

