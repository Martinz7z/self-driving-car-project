# drive_track1.py - Self-driving car for Track 1
# Student: Martin Zachariasz
# Smart Technologies CA2

import socketio 
import eventlet
# Needed for Python 3.12 to work with socketio
eventlet.monkey_patch()

from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import time

sio = socketio.Server()
app = Flask(__name__)

# For tracking how long we've been driving
frame_count = 0
start_time = None
recovery_mode = 0

# Process images same as during training
def process_image(img):
    # Crop out sky and car hood
    img = img[60:135, :, :]
    
    # Convert to YUV color space (like NVIDIA paper)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Add blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Resize to what the model expects
    img = cv2.resize(img, (200, 66))
    
    # Normalize pixels to 0-1 range
    img = img / 255.0
    
    return img

# Simple check if we can see road
def check_if_on_road(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    height, width = gray.shape
    
    # Look at bottom center where road should be
    road_area = gray[height-50:height-10, width//2-30:width//2+30]
    
    avg_brightness = np.mean(road_area)
    
    # Road brightness is usually in this range
    return 50 < avg_brightness < 150

@sio.on('telemetry')
def telemetry(sid, data):
    global frame_count, start_time, recovery_mode
    
    if start_time is None:
        start_time = time.time()
        print("\nStarting autonomous driving on Track 1")
    
    try:
        # Get image from simulator
        img_bytes = base64.b64decode(data['image'])
        image = Image.open(BytesIO(img_bytes))
        img_array = np.asarray(image)
        
        speed = float(data['speed'])
        
        # Check if we can see road
        on_road = check_if_on_road(img_array)
        
        # Process image for the model
        processed = process_image(img_array)
        img_batch = np.array([processed])
        
        # Get steering prediction from model
        steering = float(model.predict(img_batch, verbose=0)[0][0])
        
        # Testing showed model has right bias, so adjust
        steering = steering - 0.15
        
        # If off road or stuck, try to recover
        if not on_road or speed < 1.5:
            recovery_mode += 1
            print(f"Attempting recovery {recovery_mode}")
            
            # Alternate left and right to try get back
            if recovery_mode % 20 < 10:
                steering = -0.4  # Turn left
            else:
                steering = 0.4   # Turn right
            throttle = 0.2
        else:
            recovery_mode = 0
            # Normal driving
            if speed > 15:
                throttle = 0.1
            elif speed < 5:
                throttle = 0.25
            else:
                throttle = 0.15
        
        # Don't let steering get too extreme
        steering = max(-0.8, min(0.8, steering))
        
        # Send controls to simulator
        sio.emit('steer', data={
            'steering_angle': str(steering),
            'throttle': str(throttle)
        })
        
        frame_count += 1
        
        # Show status every 15 frames
        if frame_count % 15 == 0:
            elapsed = time.time() - start_time
            print(f"Frame: {frame_count} | Time: {elapsed:.1f}s")
            print(f"Speed: {speed:.1f} | Steering: {steering:.3f}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error: {e}")
        # Send safe defaults if error
        sio.emit('steer', data={
            'steering_angle': '0.0',
            'throttle': '0.1'
        })

@sio.on('connect')
def connect(sid, environ):
    global frame_count, start_time, recovery_mode
    print("Simulator connected")
    frame_count = 0
    start_time = None
    recovery_mode = 0
    
    # Start with slow forward motion
    sio.emit('steer', data={
        'steering_angle': '0.0',
        'throttle': '0.15'
    })

if __name__ == "__main__":
    print("Self-Driving Car - Track 1")
    print("Student: Martin Zachariasz")
    print("Smart Technologies CA2")
    print("="*50)
    
    # Load the trained model
    print("\nLoading model...")
    try:
        model = load_model("../models/trained_model.h5", compile=False)
        print("Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating simple model for testing")
        
        # Create basic model if loading fails
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, Flatten, Dense
        
        model = Sequential([
            Conv2D(24, (5,5), strides=(2,2), input_shape=(66,200,3), activation='elu'),
            Conv2D(36, (5,5), strides=(2,2), activation='elu'),
            Conv2D(48, (5,5), strides=(2,2), activation='elu'),
            Conv2D(64, (3,3), activation='elu'),
            Conv2D(64, (3,3), activation='elu'),
            Flatten(),
            Dense(100, activation='elu'),
            Dense(50, activation='elu'),
            Dense(10, activation='elu'),
            Dense(1)
        ])
        
        print("Simple model created")
    
    print("\nServer starting on port 4567")
    print("Open Udacity Simulator -> Track 1 -> Autonomous")
    print("="*50)
    print("Waiting for connection...")
    
    # Start server
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)