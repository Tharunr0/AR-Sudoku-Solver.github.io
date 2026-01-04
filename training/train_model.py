import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

print("Downloading and processing MNIST data...")

# 1. Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Reshape for CNN (28x28 pixels, 1 channel/grayscale)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# 3. Normalize pixel values (0-1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 4. One-hot encoding for labels (e.g., '5' becomes [0,0,0,0,0,1,0,0,0,0])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 5. Build the Model (CNN)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 6. Compile and Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training model... this might take a minute.")
model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(x_test, y_test))

# 7. Save the model
model.save('digit_model.h5')
print("Success! 'digit_model.h5' has been saved.")