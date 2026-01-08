"""
CNN Model for Self-Driving Car
Based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
"""

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
    """
    Creates the NVIDIA CNN model architecture
    
    Reference: 
    Bojarski, M., et al. (2016). End to End Learning for Self-Driving Cars
    
    Architecture:
    - Input: 66x200x3 (YUV image)
    - Normalization layer
    - 5 Convolutional layers
    - 4 Fully connected layers
    - Output: Steering angle
    """
    model = Sequential()
    
    # Input normalization
    model.add(Lambda(lambda x: x / 127.5 - 1.0, 
                     input_shape=input_shape,
                     name='normalization'))
    
    # Convolutional layers (as per NVIDIA paper)
    # Layer 1: 24 filters, 5x5 kernel, 2x2 stride
    model.add(Conv2D(24, (5, 5), strides=(2, 2), 
                     activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv1'))
    
    # Layer 2: 36 filters, 5x5 kernel, 2x2 stride
    model.add(Conv2D(36, (5, 5), strides=(2, 2), 
                     activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv2'))
    
    # Layer 3: 48 filters, 5x5 kernel, 2x2 stride
    model.add(Conv2D(48, (5, 5), strides=(2, 2), 
                     activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv3'))
    
    # Layer 4: 64 filters, 3x3 kernel
    model.add(Conv2D(64, (3, 3), activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv4'))
    
    # Layer 5: 64 filters, 3x3 kernel
    model.add(Conv2D(64, (3, 3), activation='elu',
                     kernel_regularizer=l2(0.001),
                     name='conv5'))
    
    # Flatten for fully connected layers
    model.add(Flatten(name='flatten'))
    
    # Dropout to prevent overfitting
    model.add(Dropout(0.5, name='dropout1'))
    
    # Fully connected layers
    # Layer 6: 100 neurons
    model.add(Dense(100, activation='elu',
                    kernel_regularizer=l2(0.001),
                    name='fc1'))
    
    # Layer 7: 50 neurons
    model.add(Dense(50, activation='elu',
                    kernel_regularizer=l2(0.001),
                    name='fc2'))
    
    # Layer 8: 10 neurons
    model.add(Dense(10, activation='elu',
                    kernel_regularizer=l2(0.001),
                    name='fc3'))
    
    # Output layer: 1 neuron (steering angle)
    model.add(Dense(1, name='output'))
    
    return model

def create_improved_model(input_shape=(66, 200, 3)):
    """
    Enhanced version with additional improvements:
    1. Batch normalization for faster training
    2. Additional dropout layers
    3. ELU activation for better performance
    4. L2 regularization to prevent overfitting
    """
    model = Sequential()
    
    # Input normalization
    model.add(Lambda(lambda x: x / 127.5 - 1.0, 
                     input_shape=input_shape,
                     name='normalization'))
    
    # Convolutional Block 1
    model.add(Conv2D(24, (5, 5), strides=(2, 2), 
                     padding='valid', activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Convolutional Block 2
    model.add(Conv2D(36, (5, 5), strides=(2, 2), 
                     padding='valid', activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Convolutional Block 3
    model.add(Conv2D(48, (5, 5), strides=(2, 2), 
                     padding='valid', activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Convolutional Block 4
    model.add(Conv2D(64, (3, 3), padding='valid', 
                     activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Convolutional Block 5
    model.add(Conv2D(64, (3, 3), padding='valid', 
                     activation='elu',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    
    # Flatten
    model.add(Flatten())
    
    # Dropout
    model.add(Dropout(0.5))
    
    # Fully connected layers
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
    """
    Compile the model with appropriate optimizer and loss
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae'])  # Mean Absolute Error
    
    return model

def print_model_summary(model):
    """Print model summary and return parameter count"""
    model.summary()
    
    # Calculate total parameters
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"\nModel Statistics:")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"  Total parameters: {trainable_params + non_trainable_params:,}")
    
    return trainable_params

if __name__ == "__main__":
    print("=== Testing Model Creation ===\n")
    
    # Test creating the NVIDIA model
    print("1. Creating NVIDIA model...")
    nvidia_model = create_nvidia_model()
    nvidia_model = compile_model(nvidia_model, learning_rate=0.0001)
    print("   NVIDIA model created successfully")
    
    print("\n2. Creating improved model...")
    improved_model = create_improved_model()
    improved_model = compile_model(improved_model, learning_rate=0.0001)
    print("   Improved model created successfully")
    
    print("\n3. Model summaries:")
    print("\nNVIDIA Model:")
    print_model_summary(nvidia_model)
    
    print("\nImproved Model:")
    print_model_summary(improved_model)
    
    print("\n=== Test Complete ===")