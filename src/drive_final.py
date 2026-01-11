"""
Autonomous driving - simplified working version
"""

import socketio
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import time

# Create SocketIO server
sio = socketio.Server()

# Track connection state
connected = False

def preprocess(image):
    """Preprocess image like training"""
    # Crop sky and car hood
    image = image[60:-25, :, :]
    
    # Resize to model input size
    image = cv2.resize(image, (200, 66))
    
    # Convert to YUV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    # Normalize
    image = image / 255.0 - 0.5
    
    return image

@sio.event
def connect(sid, environ):
    global connected
    connected = True
    print("=" * 50)
    print("✓ CONNECTED to simulator!")
    print("=" * 50)
    print("\nCar will start driving...")
    print("Press Ctrl+C to stop\n")

@sio.event
def disconnect(sid):
    global connected
    connected = False
    print("Disconnected from simulator")

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        try:
            # Get image
            img_str = data["image"]
            img = Image.open(BytesIO(base64.b64decode(img_str)))
            img = np.array(img)
            
            # Preprocess
            img_processed = preprocess(img)
            img_processed = np.array([img_processed])
            
            # Predict steering
            steering = float(model.predict(img_processed, verbose=0)[0][0])
            
            # Get speed
            speed = float(data.get("speed", 0))
            
            # Calculate throttle
            throttle = 0.2  # Base speed
            
            # Slow down on turns
            if abs(steering) > 0.5:
                throttle = 0.05
            elif abs(steering) > 0.3:
                throttle = 0.1
            elif abs(steering) > 0.1:
                throttle = 0.15
            
            # Don't go too fast
            if speed > 20:
                throttle = 0.1
            
            # Send control
            send_control(steering, throttle)
            
            # Print occasionally
            if np.random.random() < 0.05:  # 5% chance
                print(f"Steering: {steering:6.3f} | Throttle: {throttle:4.2f} | Speed: {speed:5.1f}")
                
        except Exception as e:
            print(f"Error: {e}")
            send_control(0, 0)

def send_control(steering, throttle):
    """Send control to simulator"""
    sio.emit("steer", data={
        "steering_angle": str(steering),
        "throttle": str(throttle)
    })

def main():
    print("Loading trained model...")
    try:
        global model
        model = load_model('../models/trained_model.h5')
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Model load error: {e}")
        return
    
    print("\n" + "=" * 50)
    print("Waiting for simulator connection...")
    print("=" * 50)
    print("\nIn Udacity simulator:")
    print("1. Select Track 1")
    print("2. Click 'Autonomous Mode'")
    print("3. Wait for connection")
    print("\nServer will run on port 4567")
    print("Press Ctrl+C to exit\n")
    
    # Start the server
    try:
        # Create a simple WSGI app
        import wsgi_ref
        app = wsgi_ref.WSGIApp(sio)
        
        # Start server (this is simplified - might need adjustment)
        from wsgiref.simple_server import make_server
        server = make_server('localhost', 4567, app)
        print("Server started on http://localhost:4567")
        server.serve_forever()
        
    except ImportError:
        # Alternative: use eventlet if available
        try:
            import eventlet
            import eventlet.wsgi
            app = socketio.WSGIApp(sio)
            eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
        except ImportError:
            print("\n" + "=" * 50)
            print("SERVER SETUP ERROR")
            print("=" * 50)
            print("\nNeed to install eventlet OR use Udacity's drive.py")
            print("\nTry:")
            print("1. pip install eventlet==0.30.2")
            print("2. Or use the official Udacity drive.py")
            print("\nFor now, let's focus on improving the model.")

if __name__ == "__main__":
    main()