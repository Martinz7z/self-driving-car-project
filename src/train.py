"""
Training script - first attempt
Trying to get the model to train on our data
"""

import tensorflow as tf
from model import create_nvidia_model, compile_model
from data_preprocessing import DataPreprocessor
import numpy as np
import os

def check_environment():
    """Check if we can run TensorFlow"""
    print("Checking TensorFlow setup...")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
def load_training_data():
    """Try to load some data for training"""
    print("\nLoading training data...")
    
    preprocessor = DataPreprocessor('data/sample_data/')
    data = preprocessor.load_data(include_side_cameras=False)
    
    if data is None or len(data) == 0:
        print("ERROR: No data found!")
        print("Need to collect data from Udacity simulator first.")
        print("Run the simulator in Training mode and save to data/training/")
        return None
    
    print(f"Loaded {len(data)} samples")
    return data

def create_simple_model():
    """Create and compile a simple model"""
    print("\nCreating model...")
    
    # Try creating the NVIDIA model
    try:
        model = create_nvidia_model(input_shape=(66, 200, 3))
        model = compile_model(model, learning_rate=0.001)
        print("Model created successfully!")
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def main():
    """Main training function"""
    print("=" * 50)
    print("Self-Driving Car - Training Setup")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Try to load data
    data = load_training_data()
    if data is None:
        return
    
    # Create model
    model = create_simple_model()
    if model is None:
        return
    
    print("\n" + "=" * 50)
    print("READY FOR TRAINING!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Collect real data using Udacity simulator")
    print("2. Update data paths in this script")
    print("3. Implement proper data generator")
    print("4. Run actual training")
    
    return model

if __name__ == "__main__":
    main()