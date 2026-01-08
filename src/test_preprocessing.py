"""
Test script for data preprocessing
Run this to verify your preprocessing works
"""

import sys
import os
sys.path.append('.')

from data_preprocessing import DataPreprocessor
import numpy as np

def test_basic_functionality():
    """Test basic preprocessing functions"""
    print("=== Testing Data Preprocessing ===\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor('data/sample_data/')
    
    # Load sample data
    print("1. Loading sample data...")
    data = preprocessor.load_data(include_side_cameras=False)
    
    if data is not None:
        print(f"   Loaded {len(data)} samples")
        print(f"   Columns: {list(data.columns)}")
        
        # Show first few rows
        print("\n   First 3 samples:")
        print(data.head(3))
        
        # Test balancing
        print("\n2. Testing data balancing...")
        balanced = preprocessor.balance_data(data, bins=5, max_samples_per_bin=2)
        if balanced is not None:
            print(f"   Balanced from {len(data)} to {len(balanced)} samples")
        
        # Test image preprocessing
        print("\n3. Testing image preprocessing...")
        if len(data) > 0:
            # Create a dummy image for testing
            test_img = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite('test_dummy.jpg', test_img)
            
            processed = preprocessor.preprocess_image('test_dummy.jpg')
            if processed is not None:
                print(f"   Original shape: {test_img.shape}")
                print(f"   Processed shape: {processed.shape}")
                print(f"   Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
            
            # Clean up
            if os.path.exists('test_dummy.jpg'):
                os.remove('test_dummy.jpg')
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_basic_functionality()