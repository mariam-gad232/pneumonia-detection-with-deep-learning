#!/usr/bin/env python
# coding: utf-8

# In[24]:


import warnings
warnings.filterwarnings('ignore')


# In[25]:


from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten


# In[26]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[27]:


from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[51]:


train_path = 'data\chest_xray\train'
valid_path = 'data\chest_xray\test'


# In[52]:


vgg = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)


# In[53]:


vgg.trainable = False
x=Flatten()(vgg.output)
x=Dense(64,activation='relu')(x)
x=Dense(32,activation='relu')(x)
outputs=Dense(2,activation='softmax')(x)
model=Model(inputs=vgg.input,outputs=outputs)
model.summary()


# In[54]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[55]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[56]:


path = 'data/chest_xray/train'
print(f"Directory path: {path}")


# In[57]:


path = 'data/chest_xray/test'
print(f"Directory path: {path}")


# In[58]:


#preparing the images before feeding them into the model

train_datagen = ImageDataGenerator(rescale = 1./255, #make pixel values smaller
                                   shear_range = 0.2,#tilts the image by 0.2 degrees    
                                   zoom_range = 0.2, #zooms the image by 20% in and out
                                   horizontal_flip = True) #flips the image horizontally left to right

test_datagen = ImageDataGenerator(rescale = 1./255)



#we tell the model where to get the images and their size
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('data/chest_xray/train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')




test_set = test_datagen.flow_from_directory('data/chest_xray/test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')


# In[60]:


r=model.fit(
    training_set,
    validation_data=test_set,
    epochs=1, #how many times the model will see the entire dataset
    steps_per_epoch=len(training_set), #how many batches the model will see in each epoch , here it is the whole lenght of the trainig set , since we will only go through it once
    validation_steps=len(test_set)
)


# In[61]:


import tensorflow as tf
from keras.models import load_model

model.save('chest_xray.h5')


# In[63]:


model=load_model('chest_xray.h5')


# In[67]:


img=image.load_img(r'D:\self development\ML projects\pneumonia detection\data\chest_xray\val\NORMAL\NORMAL2-IM-1427-0001.jpeg',target_size=(224,224))
x=image.img_to_array(img)


# In[68]:


x=np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
result=int(classes[0][0])


# In[69]:


if result==0:
    print("Person is Affected By PNEUMONIA")
else:
    print("Result is Normal")


# In[ ]:




