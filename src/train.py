"""
Training script - now with actual training!
First attempt to train the model on our data
"""

import tensorflow as tf
from model import create_nvidia_model, compile_model, print_model_summary
from data_generator import SimpleDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

def setup_training():
    """Setup and test training environment"""
    print("=" * 60)
    print("Self-Driving Car - Training Setup")
    print("=" * 60)
    
    # Check TensorFlow
    print(f"\nTensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("GPU not available - training on CPU (will be slow)")
    
    return len(gpus) > 0

def create_and_compile_model():
    """Create and compile the model"""
    print("\n" + "-" * 60)
    print("Creating model...")
    
    try:
        # Create model (66x200x3 input from preprocessing)
        model = create_nvidia_model(input_shape=(66, 200, 3))
        
        # Compile with Adam optimizer
        model = compile_model(model, learning_rate=0.001)
        
        print("Model created and compiled successfully!")
        print_model_summary(model)
        
        return model
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def prepare_data():
    """Prepare data for training"""
    print("\n" + "-" * 60)
    print("Preparing data...")
    
    try:
        # Create data generator - use small batch for testing
        generator = SimpleDataGenerator(batch_size=16)  # Smaller for testing
        
        if len(generator.image_paths) == 0:
            print("ERROR: No data available!")
            print("Make sure you have data in data/training/")
            return None
        
        print(f"Data prepared: {len(generator.image_paths)} samples")
        print(f"Batch size: 16")
        
        # Calculate steps per epoch
        steps_per_epoch = len(generator.image_paths) // 16
        if steps_per_epoch == 0:
            steps_per_epoch = 1  # At least 1 step
        
        print(f"Steps per epoch: {steps_per_epoch}")
        
        return generator, steps_per_epoch
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_model(model, generator, steps_per_epoch, epochs=3):
    """Train the model for a few epochs"""
    print("\n" + "-" * 60)
    print(f"Starting training for {epochs} epochs...")
    print("(This might take a while - first run will be slow)")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs('../models', exist_ok=True)
        
        # Create callbacks
        callbacks = [
            # Early stopping if loss doesn't improve
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=2,
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                '../models/model_checkpoint.h5',
                monitor='loss',
                save_best_only=True,
                verbose=1
            ),
            # Reduce learning rate if plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=1,
                verbose=1
            )
        ]
        
        # Train the model
        print("\nTraining started...")
        print("Epoch 1 will be slow as data loads for first time")
        
        history = model.fit(
            generator.generate_batch(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1  # Show progress bar
        )
        
        print("\nTraining completed!")
        return history
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_model(model, history):
    """Save the trained model and training history"""
    print("\n" + "-" * 60)
    print("Saving model...")
    
    try:
        # Save model
        model.save('../models/trained_model.h5')
        print("✓ Model saved to: ../models/trained_model.h5")
        
        # Save model architecture as JSON
        model_json = model.to_json()
        with open('../models/model_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        print("✓ Model architecture saved")
        
        # Save training history
        if history is not None:
            # Convert history to dict
            history_dict = {
                'loss': history.history.get('loss', []),
                'mae': history.history.get('mae', [])
            }
            
            # Save as numpy file
            np.save('../models/training_history.npy', history_dict)
            print("✓ Training history saved")
            
            # Plot training history
            if len(history_dict['loss']) > 0:
                plot_training_history(history)
        
        print("Save complete!")
        
    except Exception as e:
        print(f"Error saving model: {e}")

def plot_training_history(history):
    """Plot training loss and metrics"""
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        if 'loss' in history.history:
            plt.plot(history.history['loss'], label='Training Loss', marker='o')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        if 'mae' in history.history:
            plt.plot(history.history['mae'], label='Training MAE', marker='s')
            plt.title('Mean Absolute Error')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig('../models/training_history.png', dpi=100)
        print("✓ Training plot saved to: ../models/training_history.png")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"Error plotting history: {e}")

def quick_test_prediction(model, generator):
    """Do a quick test prediction to see if model works"""
    print("\n" + "-" * 60)
    print("Testing model prediction...")
    
    try:
        # Get a sample
        sample_img, sample_steering = generator.get_sample(0)
        
        if sample_img is not None:
            # Add batch dimension
            test_input = np.expand_dims(sample_img, axis=0)
            
            # Predict
            prediction = model.predict(test_input, verbose=0)
            
            print(f"Sample steering (actual): {sample_steering:.4f}")
            print(f"Model prediction: {prediction[0][0]:.4f}")
            print(f"Difference: {abs(sample_steering - prediction[0][0]):.4f}")
            
            return True
        else:
            print("Could not get sample for testing")
            return False
            
    except Exception as e:
        print(f"Error in prediction test: {e}")
        return False

def main():
    """Main training function"""
    print("\n" + "=" * 60)
    print("SELF-DRIVING CAR - TRAINING START")
    print("=" * 60)
    
    # Setup
    print("\n[1/5] Setting up training environment...")
    has_gpu = setup_training()
    
    # Create model
    print("\n[2/5] Creating model...")
    model = create_and_compile_model()
    if model is None:
        print("Failed to create model. Exiting.")
        return
    
    # Prepare data
    print("\n[3/5] Preparing data...")
    data_prep = prepare_data()
    if data_prep is None:
        print("Failed to prepare data. Exiting.")
        return
    
    generator, steps_per_epoch = data_prep
    
    # Quick prediction test
    print("\n[4/5] Quick prediction test...")
    quick_test_prediction(model, generator)
    
    # Ask for confirmation
    print("\n" + "=" * 60)
    print("READY TO TRAIN")
    print("=" * 60)
    print(f"Will train for 3 epochs")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {steps_per_epoch * 3}")
    print("\nNote: First epoch will be slow as images load for first time")
    
    response = input("\nStart training? (y/n): ")
    
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Train
    print("\n[5/5] Training model...")
    history = train_model(model, generator, steps_per_epoch, epochs=3)
    
    # Save results
    if history is not None:
        save_model(model, history)
        
        # Show final metrics
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        if 'loss' in history.history and len(history.history['loss']) > 0:
            print(f"Final loss: {history.history['loss'][-1]:.4f}")
        
        if 'mae' in history.history and len(history.history['mae']) > 0:
            print(f"Final MAE: {history.history['mae'][-1]:.4f}")
        
        print("\nNext steps:")
        print("1. Test model in autonomous mode")
        print("2. Collect more diverse training data")
        print("3. Train for more epochs if results are good")
    
    else:
        print("\n" + "=" * 60)
        print("TRAINING FAILED OR STOPPED EARLY")
        print("=" * 60)
        print("Check error messages above.")
        print("Common issues:")
        print("  - Not enough memory")
        print("  - Data loading errors")
        print("  - Image preprocessing issues")

if __name__ == "__main__":
    main()