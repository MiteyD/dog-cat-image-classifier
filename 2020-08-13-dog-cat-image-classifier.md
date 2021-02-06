---
title: "Dog Cat Image Classifier"
date: 2020-08-13
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Dog Cat Image Classifier

I used TensorFlow 2.0 and Keras in this project to create a convolutional neural network that correctly classifies images of cats and dogs with at least 63% accuracy.

## Project Insructions

*Note: You are currently reading this using Google Colaboratory which is a cloud-hosted version of Jupyter Notebook. This is a document containng both text cells for documentation and runnable code cells. If you are unfamiliar with Jupyter Notebook, watch this 3-minute introduction before starting this challenge: https://www.youtube.com/watch?v=inN8seMm7UI*



---


For this challenge, you will complete the code below to classify images of dogs and cats. You will use Tensorflow 2.0 and Keras to create a convolutional neural network that correctly classifies images of cats and dogs at least 63% of the time. (Extra credit if you get it to 70% accuracy!)

Some of the code is given to you but some code you must fill in to complete this challenge. Read the instruction in each text cell so you will know what you have to do in each code cell.

The first code cell imports the required libraries. The second code cell downloads the data and sets key variables. The third cell is the first place you will write your own code.

The structure of the dataset files that are downloaded looks like this (You will notice that the test directory has no subdirectories and the images are not labeled):
```
cats_and_dogs
|__ train:
    |______ cats: [cat.0.jpg, cat.1.jpg ...]
    |______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
    |______ cats: [cat.2000.jpg, cat.2001.jpg ...]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
```

You can tweak epochs and batch size if you like, but it is not required.


```python
try:
  # This command only in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
```


```python
URL = 'https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
```

    Downloading data from https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip
    70705152/70702765 [==============================] - 54s 1us/step
    

Now it is your turn! Set each of the variables below correctly. (They should no longer equal `None`.)

Create image generators for each of the three image data sets (train, validation, test). Use `ImageDataGenerator` to read / decode the images and convert them into floating point tensors. Use the `rescale` argument (and no other arguments for now) to rescale the tensors from values between 0 and 255 to values between 0 and 1.

For the `*_data_gen` variables, use the `flow_from_directory` method. Pass in the batch size, directory, target size (`(IMG_HEIGHT, IMG_WIDTH)`), class mode, and anything else required. `test_data_gen` will be the trickiest one. For `test_data_gen`, make sure to pass in `shuffle=False` to the `flow_from_directory` method. This will make sure the final predictions stay is in the order that our test expects. For `test_data_gen` it will also be helpful to observe the directory structure.


After you run the code, the output should look like this:
```
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Found 50 images belonging to 1 classes.
```


```python
train_image_generator = ImageDataGenerator(rescale = 1./255)
validation_image_generator = ImageDataGenerator(rescale = 1./255)
test_image_generator = ImageDataGenerator(rescale = 1./255)

train_data_gen = train_image_generator.flow_from_directory(train_dir,
                                                           target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                           batch_size = batch_size,
                                                           class_mode = 'binary')
val_data_gen = validation_image_generator.flow_from_directory(validation_dir,
                                                              target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                              batch_size = batch_size,
                                                              class_mode = 'binary')
test_data_gen = test_image_generator.flow_from_directory(PATH,
                                                         classes = ['test'],
                                                         target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                         batch_size = batch_size,
                                                         class_mode = None,
                                                         shuffle = False)
```

    Found 2000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.
    Found 50 images belonging to 1 classes.
    

The `plotImages` function will be used a few times to plot images. It takes an array of images and a probabilities list, although the probabilities list is optional. This code is given to you. If you created the `train_data_gen` variable correctly, then running the cell below will plot five random training images.


```python
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])

```


![png](Dog-Cat-Image-Classifier_files/Dog-Cat-Image-Classifier_7_0.png)


Recreate the `train_image_generator` using `ImageDataGenerator`. 

Since there are a small number of training examples there is a risk of overfitting. One way to fix this problem is by creating more training data from existing training examples by using random transformations.

Add 4-6 random transformations as arguments to `ImageDataGenerator`. Make sure to rescale the same as before.



```python
train_image_generator = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)

```

You don't have to do anything for the next cell. `train_data_gen` is created just like before but with the new `train_image_generator`. Then, a single image is plotted five different times using different variations.


```python
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
```

    Found 2000 images belonging to 2 classes.
    


![png](Dog-Cat-Image-Classifier_files/Dog-Cat-Image-Classifier_11_1.png)


In the cell below, create a model for the neural network that outputs class probabilities. It should use the Keras Sequential model. It will probably involve a stack of Conv2D and MaxPooling2D layers and then a fully connected layer on top that is activated by a ReLU activation function.

Compile the model passing the arguments to set the optimizer and loss. Also pass in `metrics=['accuracy']` to view training and validation accuracy for each training epoch.


```python
model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", input_shape=[IMG_HEIGHT, IMG_WIDTH, 3]))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# Adding a second convolutional layer
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# Adding a third convolutional layer
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
model.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
model.add(Dense(units=128, activation='relu'))

# Step 5 - Output Layer
model.add(Dense(units=1, activation='sigmoid'))

# Training the CNN
# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])







model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_8 (Conv2D)            (None, 150, 150, 64)      1792      
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 75, 75, 64)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 75, 75, 64)        36928     
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 37, 37, 64)        0         
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 37, 37, 64)        36928     
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 18, 18, 64)        0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 20736)             0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 128)               2654336   
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 2,730,113
    Trainable params: 2,730,113
    Non-trainable params: 0
    _________________________________________________________________
    

Use the `fit` method on your `model` to train the network. Make sure to pass in arguments for `x`, `steps_per_epoch`, `epochs`, `validation_data`, and `validation_steps`.


```python
history = model.fit(train_data_gen,
                    steps_per_epoch = len(train_data_gen),
                    epochs = epochs,
                    validation_data = val_data_gen,
                    validation_steps = len(val_data_gen))
```

    Epoch 1/15
    16/16 [==============================] - 127s 8s/step - loss: 0.7230 - accuracy: 0.4980 - val_loss: 0.6928 - val_accuracy: 0.5080
    Epoch 2/15
    16/16 [==============================] - 125s 8s/step - loss: 0.6918 - accuracy: 0.5090 - val_loss: 0.6913 - val_accuracy: 0.5130
    Epoch 3/15
    16/16 [==============================] - 125s 8s/step - loss: 0.6781 - accuracy: 0.5665 - val_loss: 0.6876 - val_accuracy: 0.5570
    Epoch 4/15
    16/16 [==============================] - 128s 8s/step - loss: 0.6551 - accuracy: 0.6315 - val_loss: 0.6646 - val_accuracy: 0.6150
    Epoch 5/15
    16/16 [==============================] - 125s 8s/step - loss: 0.6540 - accuracy: 0.6395 - val_loss: 0.6713 - val_accuracy: 0.6360
    Epoch 6/15
    16/16 [==============================] - 125s 8s/step - loss: 0.6207 - accuracy: 0.6615 - val_loss: 0.6032 - val_accuracy: 0.6770
    Epoch 7/15
    16/16 [==============================] - 125s 8s/step - loss: 0.6116 - accuracy: 0.6620 - val_loss: 0.6054 - val_accuracy: 0.6550
    Epoch 8/15
    16/16 [==============================] - 128s 8s/step - loss: 0.5889 - accuracy: 0.6955 - val_loss: 0.5985 - val_accuracy: 0.6660
    Epoch 9/15
    16/16 [==============================] - 126s 8s/step - loss: 0.5392 - accuracy: 0.7350 - val_loss: 0.5748 - val_accuracy: 0.7000
    Epoch 10/15
    16/16 [==============================] - 125s 8s/step - loss: 0.5438 - accuracy: 0.7210 - val_loss: 0.5708 - val_accuracy: 0.6860
    Epoch 11/15
    16/16 [==============================] - 125s 8s/step - loss: 0.5258 - accuracy: 0.7460 - val_loss: 0.5605 - val_accuracy: 0.7090
    Epoch 12/15
    16/16 [==============================] - 125s 8s/step - loss: 0.5012 - accuracy: 0.7490 - val_loss: 0.6336 - val_accuracy: 0.6810
    Epoch 13/15
    16/16 [==============================] - 128s 8s/step - loss: 0.4766 - accuracy: 0.7645 - val_loss: 0.5475 - val_accuracy: 0.7220
    Epoch 14/15
    16/16 [==============================] - 127s 8s/step - loss: 0.4766 - accuracy: 0.7725 - val_loss: 0.5307 - val_accuracy: 0.7220
    Epoch 15/15
    16/16 [==============================] - 126s 8s/step - loss: 0.4610 - accuracy: 0.7780 - val_loss: 0.5689 - val_accuracy: 0.7080
    

Run the next cell to visualize the accuracy and loss of the model.


```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```


![png](Dog-Cat-Image-Classifier_files/Dog-Cat-Image-Classifier_17_0.png)


Now it is time to use your model to predict whether a brand new image is a cat or a dog.

In this final cell, get the probability that each test image (from `test_data_gen`) is a dog or a cat. `probabilities` should be a list of integers. 

Call the `plotImages` function and pass in the test images and the probabilities corresponding to each test image.

After your run the cell, you should see all 50 test images with a label showing the percentage sure that the image is a cat or a dog. The accuracy will correspond to the accuracy shown in the graph above (after running the previous cell). More training images could lead to a higher accuracy.


```python
import math

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


y_pred = model.predict(test_data_gen)

probabilities = [round_half_up(value) for value in y_pred]

def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()

sample_training_images= next(test_data_gen)
plotImages(sample_training_images[:50])
```


![png](Dog-Cat-Image-Classifier_files/Dog-Cat-Image-Classifier_19_0.png)


Run this final cell to see if you passed the challenge or if you need to keep trying.


```python
answers =  [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 
            0, 0, 0, 0, 0, 0]

correct = 0

for probability, answer in zip(probabilities, answers):
  if round(probability) == answer:
    correct +=1

percentage_identified = (correct / len(answers))

passed_challenge = percentage_identified > 0.63

print(f"Your model correctly identified {round(percentage_identified, 2)}% of the images of cats and dogs.")

if passed_challenge:
  print("You passed the challenge!")
else:
  print("You haven't passed yet. Your model should identify at least 63% of the images. Keep trying. You will get it!")
```

    Your model correctly identified 0.76% of the images of cats and dogs.
    You passed the challenge!
    
