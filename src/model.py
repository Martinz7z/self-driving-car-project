# model.py - My neural network for self-driving car
# Student: Martin Zachariasz

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, 
    Dropout, Lambda, Cropping2D,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

def create_nvidia_model(input_shape=(66, 200, 3)):
    # This is the NVIDIA model from their paper
    model = Sequential()
    
    # Normalize the input images
    model.add(Lambda(lambda x: x / 127.5 - 1.0, 
                     input_shape=input_shape,
                     name='normalization'))
    
    # First convolutional layer
    model.add(Conv2D(24, (5, 5), strides=(2, 2), 
                     activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv1'))
    
    # Second convolutional layer
    model.add(Conv2D(36, (5, 5), strides=(2, 2), 
                     activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv2'))
    
    # Third convolutional layer
    model.add(Conv2D(48, (5, 5), strides=(2, 2), 
                     activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv3'))
    
    # Fourth convolutional layer
    model.add(Conv2D(64, (3, 3), activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv4'))
    
    # Fifth convolutional layer
    model.add(Conv2D(64, (3, 3), activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv5'))
    
    # Flatten to connect to dense layers
    model.add(Flatten(name='flatten'))
    
    # Dropout helps prevent overfitting
    model.add(Dropout(0.5, name='dropout1'))
    
    # First dense layer
    model.add(Dense(100, activation='elu',
                    kernel_regularizer=l2(0.001),
                    name='fc1'))
    
    # Second dense layer
    model.add(Dense(50, activation='elu',
                    kernel_regularizer=l2(0.001),
                    name='fc2'))
    
    # Third dense layer
    model.add(Dense(10, activation='elu',
                    kernel_regularizer=l2(0.001),
                    name='fc3'))
    
    # Output layer - steering angle
    model.add(Dense(1, name='output'))
    
    return model

def create_improved_model(input_shape=(66, 200, 3)):
    # This is an improved version with batch normalization
    model = Sequential()
    
    # Normalize input
    model.add(Lambda(lambda x: x / 127.5 - 1.0, 
                     input_shape=input_shape,
                     name='normalization'))
    
    # First block with batch norm
    model.add(Conv2D(24, (5, 5), strides=(2, 2), 
                     padding='valid', activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Second block with batch norm
    model.add(Conv2D(36, (5, 5), strides=(2, 2), 
                     padding='valid', activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Third block with batch norm
    model.add(Conv2D(48, (5, 5), strides=(2, 2), 
                     padding='valid', activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Fourth block with batch norm
    model.add(Conv2D(64, (3, 3), padding='valid', 
                     activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Fifth block with batch norm
    model.add(Conv2D(64, (3, 3), padding='valid', 
                     activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Flatten
    model.add(Flatten())
    
    # Dropout
    model.add(Dropout(0.5))
    
    # Dense layers with dropout
    model.add(Dense(100, activation='elu',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    model.add(Dense(50, activation='elu',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    model.add(Dense(10, activation='elu',
                    kernel_regularizer=l2(0.001)))
    
    # Output
    model.add(Dense(1))
    
    return model

def compile_model(model, learning_rate=0.0001):
    # Set up the model for training
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])
    
    return model

def print_model_summary(model):
    # Show model details
    model.summary()
    
    # Count parameters
    trainable_params = sum([w.numpy().size for w in model.trainable_weights])
    non_trainable_params = sum([w.numpy().size for w in model.non_trainable_weights])
    
    print(f"\nModel parameters:")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {non_trainable_params:,}")
    print(f"  Total: {trainable_params + non_trainable_params:,}")
    
    return trainable_params

if __name__ == "__main__":
    print("Testing model creation")
    
    # Test NVIDIA model
    print("\n1. Creating NVIDIA model...")
    nvidia_model = create_nvidia_model()
    nvidia_model = compile_model(nvidia_model, learning_rate=0.0001)
    print("   NVIDIA model created")
    
    # Test improved model
    print("\n2. Creating improved model...")
    improved_model = create_improved_model()
    improved_model = compile_model(improved_model, learning_rate=0.0001)
    print("   Improved model created")
    
    # Show summaries
    print("\n3. Model summaries:")
    print("\nNVIDIA Model:")
    print_model_summary(nvidia_model)
    
    print("\nImproved Model:")
    print_model_summary(improved_model)
    
    print("\nDone")