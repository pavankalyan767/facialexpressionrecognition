#import all the required libraries 

import matplotlib.pyplot as plt 
import numpy as np  
import pandas as pd
import seaborn as sns
import os
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from keras.models import Model,Sequential
from keras.optimizers import Adam,SGD,RMSprop

#defining the picture size and the path which contains the dataset we have used fer 2013 dataset here 
picture_size = 48
folder_path = "/kaggle/input/face-expression-recognition-dataset/images/"
# now this below code is to display images of a dataset here we are displaying angry 
expression = 'angry'
plt.style.use('dark_background')
plt.figure(figsize=(12,12))
for i in range(1,10,1):
    plt.subplot(3,3,i)
    img = load_img(folder_path+"train/"+expression+"/"+
                  os.listdir(folder_path+"train/"+expression)[i],target_size=(picture_size,picture_size))
    plt.imshow(img)
plt.show()

#here we define the batch size then train set validation set 
batch_size = 64
datagen_train = ImageDataGenerator()
datagen_val = ImageDataGenerator()
train_set = datagen_train.flow_from_directory(folder_path+"train",
                                             target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size = batch_size,
                                              class_mode = "categorical",
                                              shuffle=True
                                             )
test_set = datagen_val.flow_from_directory(folder_path+"validation",
                                          target_size=(picture_size,picture_size),
                                          color_mode = "grayscale",
                                          batch_size=batch_size,
                                          class_mode="categorical",
                                          shuffle=False)


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# here the neural network ie sequential model is used so first we define the shape 
# Define the number of classes
num_classes = 7

# Define the input shape for the images
input_shape = (48, 48, 1)  # Grayscale images have one channel

# Create the model
model = Sequential()

# Add a convolutional layer with 32 filters, each of size 3x3, and ReLU activation
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# Add a max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another convolutional layer with 64 filters, each of size 3x3, and ReLU activation
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Add another max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output to feed into the fully connected layers
model.add(Flatten())

# Add a fully connected layer with 128 units and ReLU activation
model.add(Dense(128, activation='relu'))

# Add the final output layer with 7 units (one for each class) and softmax activation
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to see the architecture
model.summary()

#here we have written code for the model to comeback or stop if the accuracy is becoming lower than the previous saved ones 
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
checkpoint = ModelCheckpoint("./model.h5",monitor="val_acc",verbose=1,save_best_only=True,mode='max')
early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1,
                              restore_best_weights=True
                              )
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2,
                                       patience=3,
                                       verbose=1,
                                       min_delta=0.0001)
callbacks_list = [early_stopping,checkpoint,reduce_learningrate]
epochs = 500
history = model.fit(x=train_set,
                              steps_per_epoch=train_set.n//train_set.batch_size,
                              epochs=epochs,
                              validation_data=test_set,
                              validation_steps=test_set.n//test_set.batch_size,
                              callbacks=callbacks_list
                             )

plt.style.use('dark_background')
# this code over here is written to plot values of acuracy and loss
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.suptitle('Optimizer :Adam',fontsize=15)
plt.ylabel('Loss',fontsize=20)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.ylabel('Accuracy',fontsize=20)
plt.plot(history.history['accuracy'],label='Training Accuracy')
