"""
Data Preprocessing for Self-Driving Car Project
Enhanced version with data loading, balancing, and augmentation
"""

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class DataPreprocessor:
    """Handles loading and preprocessing of driving data"""
    
    def __init__(self, data_path='data/training/'):
        self.data_path = data_path
        self.log_file = os.path.join(data_path, 'driving_log.csv')
        
    def load_data(self, include_side_cameras=True, correction=0.2):
        """
        Load driving log data
        Args:
            include_side_cameras: Whether to use left/right camera images
            correction: Steering correction for side cameras
        """
        try:
            # Load CSV with appropriate column names
            columns = ['center', 'left', 'right', 'steering', 
                      'throttle', 'brake', 'speed']
            df = pd.read_csv(self.log_file, names=columns)
            
            print(f"Successfully loaded {len(df)} samples")
            
            # If using side cameras, add adjusted steering for left/right images
            if include_side_cameras:
                center_images = df[['center', 'steering']].copy()
                center_images['source'] = 'center'
                
                left_images = df[['left', 'steering']].copy()
                left_images.columns = ['center', 'steering']
                left_images['steering'] += correction  # Adjust for left camera
                left_images['source'] = 'left'
                
                right_images = df[['right', 'steering']].copy()
                right_images.columns = ['center', 'steering']
                right_images['steering'] -= correction  # Adjust for right camera
                right_images['source'] = 'right'
                
                # Combine all images
                all_images = pd.concat([center_images, left_images, right_images], 
                                      ignore_index=True)
                print(f"Total samples with side cameras: {len(all_images)}")
                return all_images
            
            return df
            
        except FileNotFoundError:
            print(f"Error: Could not find {self.log_file}")
            print("Please collect data using the Udacity simulator first")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def balance_data(self, df, bins=25, max_samples_per_bin=400):
        """
        Balance the dataset to reduce bias toward zero steering
        """
        if df is None or len(df) == 0:
            return df
            
        print(f"Original data size: {len(df)}")
        
        # Create histogram of steering angles
        hist, bin_edges = np.histogram(df['steering'], bins=bins)
        
        # Print distribution
        print("Steering angle distribution:")
        for i in range(len(hist)):
            print(f"  Bin {i+1}: {bin_edges[i]:.3f} to {bin_edges[i+1]:.3f}: {hist[i]} samples")
        
        # Remove samples from over-represented bins
        balanced_indices = []
        
        for i in range(len(bin_edges)-1):
            # Get indices for current bin
            bin_mask = (df['steering'] >= bin_edges[i]) & (df['steering'] < bin_edges[i+1])
            bin_indices = df[bin_mask].index.tolist()
            
            # If too many samples, randomly select max_samples_per_bin
            if len(bin_indices) > max_samples_per_bin:
                np.random.shuffle(bin_indices)
                balanced_indices.extend(bin_indices[:max_samples_per_bin])
            else:
                balanced_indices.extend(bin_indices)
        
        balanced_df = df.loc[balanced_indices]
        print(f"Balanced data size: {len(balanced_df)}")
        
        return balanced_df
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for the model
        Steps based on NVIDIA paper:
        1. Crop (remove sky and car hood)
        2. Resize to 66x200
        3. Convert to YUV color space
        4. Normalize
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Crop (60px from top, 25px from bottom)
            # Original: 160x320, Cropped: 75x320
            image = image[60:-25, :]
            
            # Resize to NVIDIA model input size (66x200)
            image = cv2.resize(image, (200, 66))
            
            # Convert to YUV (as per NVIDIA paper)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            
            # Normalize to [-0.5, 0.5]
            image = image / 255.0 - 0.5
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def augment_image(self, image, steering_angle):
        """
        Data augmentation to increase dataset diversity
        """
        augmented_images = []
        augmented_angles = []
        
        # Original image
        augmented_images.append(image)
        augmented_angles.append(steering_angle)
        
        # Flip image horizontally (mirror)
        flipped_image = cv2.flip(image, 1)
        flipped_angle = -steering_angle
        augmented_images.append(flipped_image)
        augmented_angles.append(flipped_angle)
        
        # Adjust brightness (random)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        brightness = np.random.uniform(0.5, 1.5)
        hsv[:,:,2] = hsv[:,:,2] * brightness
        bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        augmented_images.append(bright_image)
        augmented_angles.append(steering_angle)
        
        return augmented_images, augmented_angles
    
    def plot_steering_distribution(self, df, title="Steering Angle Distribution"):
        """
        Plot histogram of steering angles
        """
        plt.figure(figsize=(10, 6))
        plt.hist(df['steering'], bins=50, edgecolor='black', alpha=0.7)
        plt.title(title)
        plt.xlabel('Steering Angle')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print statistics
        print(f"Steering Statistics:")
        print(f"  Mean: {df['steering'].mean():.4f}")
        print(f"  Std: {df['steering'].std():.4f}")
        print(f"  Min: {df['steering'].min():.4f}")
        print(f"  Max: {df['steering'].max():.4f}")
        print(f"  Zero steering samples: {len(df[abs(df['steering']) < 0.01])}")

# Test function
def test_preprocessing():
    """Test the preprocessing functions"""
    print("Testing data preprocessing module...")
    
    # Create a sample image for testing
    test_image = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
    cv2.imwrite('test_image.jpg', test_image)
    
    # Test preprocessing
    preprocessor = DataPreprocessor()
    processed = preprocessor.preprocess_image('test_image.jpg')
    
    if processed is not None:
        print(f"Original shape: {test_image.shape}")
        print(f"Processed shape: {processed.shape}")
        print("Preprocessing test passed!")
    
    # Clean up
    if os.path.exists('test_image.jpg'):
        os.remove('test_image.jpg')

if __name__ == "__main__":
    test_preprocessing()