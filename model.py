from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, BatchNormalization, Cropping2D
from contextlib import redirect_stdout

import argparse
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

correction_factor = 0.2
base_path = './data'
image_path = base_path + '/IMG/'
driving_log_path = base_path + '/driving_log.csv'

epochs = 1
batch_size = 32

data = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for line in reader:
        data.append(line)

training_samples, validation_samples = train_test_split(data, test_size=0.2)

def data_generator(samples, batch_size=128):
    num_samples = len(samples)

    while True:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images, steering_angles = [], []

            for batch_sample in batch_samples:
                augmented_images, augmented_angles = process_batch(batch_sample)
                images.extend(augmented_images)
                steering_angles.extend(augmented_angles)

            X_train, y_train = np.array(images), np.array(steering_angles)
            yield shuffle(X_train, y_train)

def process_batch(batch_sample):
    steering_angle = np.float32(batch_sample[3])
    images, steering_angles = [], []

    for image_path_index in range(3):
        image_name = batch_sample[image_path_index].split('/')[-1]

        image = cv2.imread(image_path + image_name)
        rgb_image = bgr2rgb(image)
        # cropped = cropimg(rgb_image)
        # resized = resize(cropped)

        images.append(rgb_image)

        if image_path_index == 1:
            steering_angles.append(steering_angle + correction_factor)
        elif image_path_index == 2:
            steering_angles.append(steering_angle - correction_factor)
        else:
            steering_angles.append(steering_angle)

        if image_path_index == 0:
            flipped_center_image = flipimg(rgb_image)
            images.append(flipped_center_image)
            steering_angles.append(-steering_angle)

    return images, steering_angles

def train_generator(training_samples, batch_size=128):
    return data_generator(samples=training_samples, batch_size=batch_size)

def validation_generator(validation_samples, batch_size=128):
    return data_generator(samples=validation_samples, batch_size=batch_size)

#=======================================================================================================================
# helper functions
#=======================================================================================================================
def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def flipimg(image):
    return cv2.flip(image, 1)

def cropimg(image):
    cropped = image[60:130, :]
    return cropped

def resize(image, shape=(160, 70)):
    return cv2.resize(image, shape)
#=======================================================================================================================

def model(loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model

model = model()

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
model.fit_generator(generator=train_generator(training_samples), validation_data=validation_generator(validation_samples),
                    epochs=epochs, steps_per_epoch=len(training_samples) * 10 // batch_size,
                    validation_steps=len(validation_samples) // batch_size)
model.save('model.h5')