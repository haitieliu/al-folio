---
layout: page
title: Traffic Sign Classification
description: CNN Traffic Sign Classification
img: /assets/img/GTSRB/output_10_0.png
importance: 1
category: work
---
- Built and trained a Convolutional Neural Network (CNN) using TensorFlow to classify 43 types of traffic signs, using over 60,000 images from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
- Applied advanced techniques such as max pooling and dropout in neural network architecture to enhance training efficiency.
- Utilized OpenCV for high-quality image preprocessing and data augmentation, significantly increasing the model's robustness.
- Experimented with various neural network architectures and parameters, optimizing for accuracy, and achieving a 98% accuracy rate in traffic sign classification.


## Importing Required Libraries


```python
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
np.random.seed(42)

from matplotlib import style
style.use('fivethirtyeight')
```

## Assigning Path for Dataset


```python
data_dir = '../input/gtsrb-german-traffic-sign'
train_path = '../input/gtsrb-german-traffic-sign/Train'
test_path = '../input/gtsrb-german-traffic-sign/'

# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3
```

## Finding Total Classes


```python
NUM_CATEGORIES = len(os.listdir(train_path))
NUM_CATEGORIES
```




    43




```python
# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
```

## Visualizing The Dataset


```python
folders = os.listdir(train_path)

train_number = []
class_num = []

for folder in folders:
    train_files = os.listdir(train_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])
    
# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [ list(tuple) for tuple in  tuples]

# Plotting the number of images in each class
plt.figure(figsize=(21,10))  
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.show()
```


    
![png](/assets/img/GTSRB/output_9_0.png)


    



```python
# Visualizing 25 random images from test data
import random
from matplotlib.image import imread

test = pd.read_csv(data_dir + '/Test.csv')
imgs = test["Path"].values

plt.figure(figsize=(25,25))

for i in range(1,26):
    plt.subplot(5,5,i)
    random_img_path = data_dir + '/' + random.choice(imgs)
    rand_img = imread(random_img_path)
    plt.imshow(rand_img)
    plt.grid(b=None)
    plt.xlabel(rand_img.shape[1], fontsize = 20)#width of image
    plt.ylabel(rand_img.shape[0], fontsize = 20)#height of image
```


    
![png](/assets/img/GTSRB/output_10_0.png)

    


## Collecting the Training Data


```python
image_data = []
image_labels = []

for i in range(NUM_CATEGORIES):
    path = data_dir + '/Train/' + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)

# Changing the list to numpy array
image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(image_data.shape, image_labels.shape)
```

    (39209, 30, 30, 3) (39209,)
    

## Shuffling the training data


```python
shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]
```

## Splitting the data into train and validation set


```python
X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255 
X_val = X_val/255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)
```

    X_train.shape (27446, 30, 30, 3)
    X_valid.shape (11763, 30, 30, 3)
    y_train.shape (27446,)
    y_valid.shape (11763,)
    

## One hot encoding the labels


```python
y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)

print(y_train.shape)
print(y_val.shape)
```

    (27446, 43)
    (11763, 43)
    

## Making the model


```python
model = keras.models.Sequential([    
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(43, activation='softmax')
])
```


```python
lr = 0.001
epochs = 30

opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

## Augmenting the data and training the model


```python
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val))
```

    Epoch 1/30
    858/858 [==============================] - 16s 18ms/step - loss: 1.1347 - accuracy: 0.6976 - val_loss: 0.1190 - val_accuracy: 0.9612
    Epoch 2/30
    858/858 [==============================] - 16s 18ms/step - loss: 0.1845 - accuracy: 0.9443 - val_loss: 0.0397 - val_accuracy: 0.9870
    Epoch 3/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.1081 - accuracy: 0.9674 - val_loss: 0.0325 - val_accuracy: 0.9901
    Epoch 4/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0790 - accuracy: 0.9756 - val_loss: 0.0231 - val_accuracy: 0.9929
    Epoch 5/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0715 - accuracy: 0.9776 - val_loss: 0.0218 - val_accuracy: 0.9931
    Epoch 6/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0586 - accuracy: 0.9813 - val_loss: 0.0076 - val_accuracy: 0.9984
    Epoch 7/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0504 - accuracy: 0.9850 - val_loss: 0.0090 - val_accuracy: 0.9974
    Epoch 8/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0417 - accuracy: 0.9868 - val_loss: 0.0387 - val_accuracy: 0.9878
    Epoch 9/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0400 - accuracy: 0.9873 - val_loss: 0.0079 - val_accuracy: 0.9980
    Epoch 10/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0322 - accuracy: 0.9902 - val_loss: 0.0082 - val_accuracy: 0.9975
    Epoch 11/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0362 - accuracy: 0.9891 - val_loss: 0.0098 - val_accuracy: 0.9969
    Epoch 12/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0354 - accuracy: 0.9891 - val_loss: 0.0058 - val_accuracy: 0.9986
    Epoch 13/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0316 - accuracy: 0.9906 - val_loss: 0.0061 - val_accuracy: 0.9981
    Epoch 14/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0224 - accuracy: 0.9932 - val_loss: 0.0043 - val_accuracy: 0.9983
    Epoch 15/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0208 - accuracy: 0.9934 - val_loss: 0.0046 - val_accuracy: 0.9990
    Epoch 16/30
    858/858 [==============================] - 15s 17ms/step - loss: 0.0187 - accuracy: 0.9939 - val_loss: 0.0044 - val_accuracy: 0.9986
    Epoch 17/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0233 - accuracy: 0.9931 - val_loss: 0.0046 - val_accuracy: 0.9986
    Epoch 18/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0183 - accuracy: 0.9942 - val_loss: 0.0052 - val_accuracy: 0.9989
    Epoch 19/30
    858/858 [==============================] - 16s 18ms/step - loss: 0.0186 - accuracy: 0.9943 - val_loss: 0.0049 - val_accuracy: 0.9986
    Epoch 20/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0140 - accuracy: 0.9957 - val_loss: 0.0041 - val_accuracy: 0.9992
    Epoch 21/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0139 - accuracy: 0.9949 - val_loss: 0.0034 - val_accuracy: 0.9991
    Epoch 22/30
    858/858 [==============================] - 16s 18ms/step - loss: 0.0138 - accuracy: 0.9959 - val_loss: 0.0173 - val_accuracy: 0.9957
    Epoch 23/30
    858/858 [==============================] - 16s 18ms/step - loss: 0.0131 - accuracy: 0.9960 - val_loss: 0.0027 - val_accuracy: 0.9992
    Epoch 24/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0121 - accuracy: 0.9962 - val_loss: 0.0028 - val_accuracy: 0.9994
    Epoch 25/30
    858/858 [==============================] - 16s 18ms/step - loss: 0.0129 - accuracy: 0.9957 - val_loss: 0.0025 - val_accuracy: 0.9991
    Epoch 26/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0103 - accuracy: 0.9968 - val_loss: 0.0027 - val_accuracy: 0.9993
    Epoch 27/30
    858/858 [==============================] - 16s 18ms/step - loss: 0.0092 - accuracy: 0.9972 - val_loss: 0.0037 - val_accuracy: 0.9991
    Epoch 28/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0132 - accuracy: 0.9962 - val_loss: 0.0022 - val_accuracy: 0.9996
    Epoch 29/30
    858/858 [==============================] - 16s 18ms/step - loss: 0.0103 - accuracy: 0.9968 - val_loss: 0.0016 - val_accuracy: 0.9996
    Epoch 30/30
    858/858 [==============================] - 15s 18ms/step - loss: 0.0070 - accuracy: 0.9976 - val_loss: 0.0018 - val_accuracy: 0.9994
    

## Evaluating the model


```python
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
```


    
![png](/assets/img/GTSRB/output_25_0.png)

    


## Loading the test data and running the predictions


```python
test = pd.read_csv(data_dir + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data =[]

for img in imgs:
    try:
        image = cv2.imread(data_dir + '/' +img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test/255

pred = model.predict_classes(X_test)

#Accuracy with the test data
print('Test Data accuracy: ',accuracy_score(labels, pred)*100)
```

    Test Data accuracy:  98.53523357086303
    

## Visualizing the confusion matrix


```python
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, pred)
```


```python
import seaborn as sns
df_cm = pd.DataFrame(cf, index = classes,  columns = classes)
plt.figure(figsize = (20,20))
sns.heatmap(df_cm, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7e81f8f93290>




    
![png](/assets/img/GTSRB/output_30_1.png)

    


## Classification report


```python
from sklearn.metrics import classification_report

print(classification_report(labels, pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        60
               1       0.99      1.00      0.99       720
               2       1.00      1.00      1.00       750
               3       0.99      0.99      0.99       450
               4       1.00      0.99      1.00       660
               5       0.98      1.00      0.99       630
               6       1.00      0.95      0.97       150
               7       1.00      1.00      1.00       450
               8       1.00      0.99      0.99       450
               9       0.99      1.00      0.99       480
              10       1.00      1.00      1.00       660
              11       0.96      0.96      0.96       420
              12       1.00      0.97      0.98       690
              13       0.99      1.00      0.99       720
              14       1.00      1.00      1.00       270
              15       0.90      1.00      0.95       210
              16       1.00      1.00      1.00       150
              17       1.00      1.00      1.00       360
              18       1.00      0.99      0.99       390
              19       1.00      1.00      1.00        60
              20       0.93      1.00      0.96        90
              21       0.96      0.98      0.97        90
              22       1.00      0.77      0.87       120
              23       0.96      1.00      0.98       150
              24       1.00      0.98      0.99        90
              25       0.94      0.99      0.97       480
              26       0.99      1.00      0.99       180
              27       0.88      0.50      0.64        60
              28       0.97      0.98      0.97       150
              29       0.95      1.00      0.97        90
              30       0.85      0.92      0.88       150
              31       1.00      0.99      0.99       270
              32       0.95      1.00      0.98        60
              33       1.00      1.00      1.00       210
              34       1.00      1.00      1.00       120
              35       0.99      1.00      1.00       390
              36       0.98      1.00      0.99       120
              37       1.00      1.00      1.00        60
              38       1.00      0.99      0.99       690
              39       0.98      0.99      0.98        90
              40       0.96      0.97      0.96        90
              41       1.00      0.95      0.97        60
              42       1.00      0.97      0.98        90
    
        accuracy                           0.99     12630
       macro avg       0.98      0.97      0.97     12630
    weighted avg       0.99      0.99      0.98     12630
    
    

## Predictions on Test Data


```python
plt.figure(figsize=(25, 25))
start_index = 0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    prediction = pred[start_index + i]
    actual = labels[start_index + i]
    col = 'g' if prediction == actual else 'r'

    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color=col)
    plt.imshow(X_test[start_index + i])

plt.show()

```


    
![png](/assets/img/GTSRB/output_34_0.png)

    

