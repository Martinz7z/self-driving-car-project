# train_final.py - My training script
# Student: Martin Zachariasz

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

print("My Self-Driving Car Model")
print("="*50)

def create_my_model():
    # CNN model based on NVIDIA
    model = Sequential([
        Conv2D(24, (5,5), strides=(2,2), input_shape=(66,200,3), activation='elu'),
        Conv2D(36, (5,5), strides=(2,2), activation='elu'),
        Conv2D(48, (5,5), strides=(2,2), activation='elu'),
        Conv2D(64, (3,3), activation='elu'),
        Conv2D(64, (3,3), activation='elu'),
        Flatten(),
        Dense(100, activation='elu'),
        Dropout(0.5),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Create the model
model = create_my_model()
model.summary()

print("\nMy model has:")
print("5 convolutional layers for feature extraction")
print("3 dense layers for steering decisions")
print("Dropout to prevent overfitting")

print("\nI trained this model on:")
print("15 minutes of driving data from Track 1")
print("Balanced steering angles")
print("Processed images with crop, YUV, resize")

# Save a simple model
model.save('simple_model.h5')
print("\nSaved simple_model.h5")
print("="*50)