#!/usr/bin/env python
# coding: utf-8

# # Image Data Augmentation with Keras
# 
# ![Horizontal Flip](assets/horizontal_flip.jpg)

# # Task 1: Import Libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt

print('Using TensorFlow', tf.__version__)


# # Task 2: Rotation

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40
)


# In[ ]:


image_path = 'images/train/cat/cat.jpg'

plt.imshow(plt.imread(image_path));


# In[ ]:


x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));


# # Task 3: Width and Height Shifts

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=[-40, -20, 0, 20, 40],
    height_shift_range=[-50,50]
)


# In[ ]:


x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));


# # Task 4: Brightness

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    brightness_range=(0., 2.)
)

x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));


# # Task 5: Shear Transformation

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    shear_range=45
)

x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));


# # Task 6: Zoom

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=0.5
)

x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));


# # Task 7: Channel Shift

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    channel_shift_range=100
)

x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));


# # Task 8: Flips

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
)

x, y = next(generator.flow_from_directory('images', batch_size=1))
plt.imshow(x[0].astype('uint8'));


# # Task 9: Normalization
# 
# ### Featurewise

# In[ ]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

generator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

generator.fit(x_train)


# In[ ]:


x, y = next(generator.flow(x_train, y_train, batch_size=1))
print(x.mean(), x.std(), y)
print(x_train.mean())


# ### Samplewise

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True
)

x, y = next(generator.flow(x_train, y_train, batch_size=1))
print(x.mean(), x.std(), y)


# # Task 10: Rescale and Preprocessing Function

# In[ ]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rescale=1.
)


# In[ ]:


x, y = next(generator.flow(x_train, y_train, batch_size=1))


# In[ ]:


print(x.mean(), x.std(), y)


# # Task 11: Using in Model Training

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_shape=(32, 32, 3), pooling='avg'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


_ = model.fit(
    generator.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=10, epochs=1
)


# In[ ]:




