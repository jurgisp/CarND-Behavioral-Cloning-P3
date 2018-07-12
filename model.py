import pickle
import random
import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, Lambda, Flatten, Conv2D, MaxPooling2D, Cropping2D
from keras.models import Sequential, Model, load_model

def read_log_data(paths, steering_correction=None, flip=False):
    images = []
    measurements = []
    
    for path in paths:
        print('Loading data from', path)
        log = pd.read_csv(path, header=None)
        log_images_center = log[0].tolist()
        log_images_left = log[1].tolist()
        log_images_right = log[2].tolist()
        log_steering = log[3].tolist()

        image_dir = '/'.join(path.split('/')[:-1]) + '/IMG/'

        for i in range(len(log_steering)):
            img_center = np.asarray(Image.open(image_dir + log_images_center[i].replace('\\', '/').split('/')[-1]))
            steering = log_steering[i]
            if img_center is None:
                continue

            images.append(img_center)
            measurements.append(steering)

            if flip:
                images.append(np.fliplr(img_center))
                measurements.append(-steering)

            if steering_correction is not None:
                img_left = np.asarray(Image.open(image_dir + log_images_left[i].replace('\\', '/').split('/')[-1]))
                img_right = np.asarray(Image.open(image_dir + log_images_right[i].replace('\\', '/').split('/')[-1]))
                if img_left is None or img_right is None:
                    continue

                images.extend([img_left, img_right])
                measurements.extend([steering+steering_correction, steering-steering_correction])

                if flip:
                    images.extend([np.fliplr(img_left), np.fliplr(img_right)])
                    measurements.extend([-steering-steering_correction, -steering+steering_correction])

    X = np.array(images)
    y = np.array(measurements)
    return (X, y)

def create_lenet():
    inputs = Input(shape=(160,320,3))
    x = Cropping2D(cropping=((60,30), (0,0)))(inputs)
    x = Lambda(lambda x: 2 * (x / 255.0 - 0.5))(x)

    x = Conv2D(6, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    y = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer='Adam', loss='mse')
    return model

def create_mobilenet():
    from keras.applications.mobilenet import MobileNet, preprocess_input

    inputs = Input(shape=(160,320,3))
    inputs = Cropping2D(cropping=((60,30), (0,0)))(inputs)
    inputs = Lambda(preprocess_input)(inputs)

    base_model = MobileNet(input_tensor=inputs, input_shape=(70,320,3), weights=None, include_top=False, pooling='avg')
    x = base_model.output
    y = Dense(1, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=y)

    model.compile(optimizer='Adam', loss='mse')
    return model

if __name__ == '__main__':

    # Load data

    (X_train, y_train) = read_log_data(['data/driving_log_laps_1.csv'], steering_correction=0.1, flip=True)
    (X_valid, y_valid) = read_log_data(['data/driving_log_laps_2.csv'])
    print(X_train.shape, y_train.shape) # (41892, 160, 320, 3) (41892,)
    print(X_valid.shape, y_valid.shape) # (3695, 160, 320, 3) (3695,)

    # Build model

    model = create_mobilenet()

    # Train

    EPOCHS = 10
    model.fit(X_train, y_train, 
        epochs=EPOCHS,
        batch_size=64,
        validation_data=(X_valid, y_valid),
        shuffle=True,
        verbose=2)

    # Save

    model.save('model.h5')
