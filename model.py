import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Conv2D,GlobalAveragePooling2D,Flatten
from keras import layers,models
from keras.models import Sequential,Model
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

train_path = "hotdog-nothotdog/hotdog-nothotdog/train"
test_path = "hotdog-nothotdog/hotdog-nothotdog/test"

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.09)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path, target_size=(300,300), batch_size=32, class_mode='binary',subset='training')
test_generator = test_datagen.flow_from_directory(test_path, target_size=(300,300), batch_size=32, class_mode='binary')
val_generator = train_datagen.flow_from_directory(train_path, target_size=(300,300), batch_size=32, class_mode='binary', subset='validation')

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss = "binary_crossentropy", optimizer = 'adam',
               metrics = ['accuracy'])

history = model.fit(train_generator, epochs=5, validation_data=val_generator)

model.save('hotdog.h5')
