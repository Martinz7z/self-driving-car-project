# drive_track2.py - Trying Track 2
# Student: Martin Zachariasz

import socketio 
import eventlet
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

frame_count = 0

# Process image for Track 2
def process_image(img):
    # Track 2 needs different crop
    img = img[40:115, :, :]
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    global frame_count
    
    try:
        # Get image from car
        img_bytes = base64.b64decode(data['image'])
        image = Image.open(BytesIO(img_bytes))
        img_array = np.asarray(image)
        
        speed = float(data['speed'])
        
        # Process image
        processed = process_image(img_array)
        img_batch = np.array([processed])
        
        # Get model prediction
        steering = float(model.predict(img_batch, verbose=0)[0][0])
        
        # My model was trained on Track 1
        # For Track 2 I use a simple pattern
        if frame_count < 50:
            steering = -0.2
        elif frame_count < 100:
            steering = 0.1
        else:
            steering = -0.1
        
        # Go slow on Track 2
        throttle = 0.1
        
        # Send to car
        sio.emit('steer', data={
            'steering_angle': str(steering),
            'throttle': str(throttle)
        })
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Frame: {frame_count}")
            print(f"Speed: {speed:.1f}, Steering: {steering:.3f}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error: {e}")
        sio.emit('steer', data={'steering_angle': '0.0', 'throttle': '0.1'})

@sio.on('connect')
def connect(sid, environ):
    global frame_count
    print("Starting Track 2")
    frame_count = 0
    sio.emit('steer', data={'steering_angle': '0.0', 'throttle': '0.15'})

if __name__ == "__main__":
    print("Trying Track 2")
    print("Model was trained on Track 1")
    print("Using pattern for Track 2")
    print("="*50)
    
    # Load the model
    model = load_model("../models/trained_model.h5", compile=False)
    print("Model loaded")
    
    print("\nServer on port 4567")
    print("Open simulator, select Track 2")
    print("="*50)
    
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)