import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D



lines = []

# read the csv file created from the training data
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
augmented_images = []
augmented_measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = 'data\\IMG\\' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# augment the data by flipping the images to take care of clockwise and anitclokwise movemements
for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-0.58)

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

model = Sequential()
#crop the top layer of the images so that the training focuses only on relevant portions of the image
model.add(Cropping2D(cropping=((60,10), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0), input_shape=(160,320,3)))
#convolution layer 1
model.add(Convolution2D(6, 5, 5,input_shape=(160,320,3)))
model.add(MaxPooling2D())
#convolution layer 2
model.add(Convolution2D(6, 5, 5,input_shape=(160,320,3)))
model.add(MaxPooling2D())
model.add(Activation('relu'))
# use dropout to eliminate overfitting and improve the validation loss
model.add(Dropout(.3))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mse')
#splitting the training data into training and validation sets
model.fit(X_train, Y_train,validation_split = 0.2, shuffle= True, nb_epoch=7)
model.save('model.h5')