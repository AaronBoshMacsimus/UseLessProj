<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />

# Lightweight Geometric Shape Classifier 🎯

## Basic Details

### Team Name: BOSHMACSIS

### Team Members
- Team Lead: AARON BOSH MACSIMUS - COLLEGE OF ENGINEERING THALASSERY

### Project Description
This project powers a self-balancing robot with a camera to detect and classify geometric shapes (e.g., pillars as cylinders) in its path, calculate their positions (x,y coordinates), and send the data to a web page. The web page renders a replica of the scene using pre-classified 3D models from a model folder, perfectly placing shapes like cylinders where they belong in the real world!

### The Problem (that doesn't exist)
The world is a chaotic mess of shapes lurking in every corner, confusing robots on their merry way! Pillars, boxes, and pizza slices are out there, plotting to derail navigation with their geometric trickery. Nobody asked for a robot to map these shapes onto a web page, but we decided it’s a crisis worth solving!

### The Solution (that nobody asked for)
We’ve unleashed a wobbly self-balancing robot armed with a camera and a brainy Raspberry Pi Zero 2 W to sniff out shapes like pillars (cylinders, obviously), pinpoint their (x,y) positions, and beam this info to a web page. Instead of boring video frames, the web page conjures a dazzling digital replica of the scene, pulling 3D models from a folder to place cylinders and cubes exactly where the robot sees them. Who needs reality when you’ve got a virtual shape party?

## Technical Details

### Technologies/Components Used
**For Software:**
- **Languages used**: Python 3.7+
- **Frameworks used**: Flask (for web scene rendering)
- **Libraries used**: OpenCV (`opencv-python-headless`), NumPy, scikit-learn, psutil, picamera, requests
- **Tools used**: Git, VS Code/PyCharm, draw.io (for diagrams)

**For Hardware:**
- **Main components**:
  - Raspberry Pi Zero 2 W (for classification and position detection)
  - Raspberry Pi Camera Module v2 (8MP, for scene capture)
  - Self-balancing robot chassis with dual DC motors
  - L298N motor driver
  - MPU-6050 IMU (6-axis for balance control)
  - 3.7V LiPo battery pack
- **Specifications**:
  - Raspberry Pi Zero 2 W: 512MB RAM, 1GHz quad-core CPU
  - Camera: 8MP, 640x480 resolution at 32 FPS
  - Motors: 3-6V DC, 100-300 RPM
  - IMU: Accelerometer + gyroscope
- **Tools required**: Screwdriver, soldering iron (optional), SSH client for Pi setup

### Implementation

**For Software:**

#### Installation
```bash
pip install opencv-python-headless numpy scikit-learn psutil picamera[array] requests flask
git clone https://github.com/<your-username>/lightweight-shape-classifier.git
cd lightweight-shape-classifier
```

#### Run
1. **Training the Model**:
   ```bash
   python geoImg2.py
   ```
   - Enter dataset path (e.g., `D:\AAron\downloaded_dataset`) to train the classifier on up to 50 images per class.
   - Builds a Decision Tree model for shape classification.

2. **Real-Time Robot Operation**:
   - Modify `geoImg2.py` to include the real-time capture and position detection loop:
     ```python
     from picamera import PiCamera
     from picamera.array import PiRGBArray
     import time
     import requests
     import cv2
     import numpy as np

     def send_to_webpage(shape, confidence, x, y):
         data = {'shape': shape, 'confidence': confidence, 'x': x, 'y': y, 'timestamp': time.time()}
         try:
             requests.post('http://<web-server-ip>:5000/update', json=data, timeout=1)
         except requests.RequestException as e:
             print(f"Web transmission error: {e}")

     camera = PiCamera()
     camera.resolution = (640, 480)
     camera.framerate = 32
     raw_capture = PiRGBArray(camera, size=(640, 480))
     time.sleep(0.1)

     for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
         image = frame.array
         prediction, confidence = classifier.predict(image)
         if prediction:
             shape_type = "3D" if prediction in classifier.shape_classes_3d else "2D"
             # Calculate object position (centroid of largest contour)
             gray, edges = classifier.preprocess_image_lightweight(image)
             contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             x, y = -1, -1
             if contours:
                 largest_contour = max(contours, key=cv2.contourArea)
                 moments = cv2.moments(largest_contour)
                 if moments['m00'] != 0:
                     x = int(moments['m10'] / moments['m00'])
                     y = int(moments['m01'] / moments['m00'])
             send_to_webpage(prediction.upper() + f" ({shape_type})", confidence, x, y)
             print(f"Classified: {prediction.upper()} ({shape_type}) at ({x}, {y}) with confidence {confidence:.3f}")
         raw_capture.truncate(0)
     ```
   - Run:
     ```bash
     python geoImg2.py --robot
     ```

3. **Web Server for Scene Rendering**:
   - Create a Flask server (`flask_server.py`) to receive and render the scene:
     ```python
     from flask import Flask, request, jsonify, render_template
     app = Flask(__name__)
     scene_data = []

     @app.route('/update', methods=['POST'])
     def update():
         data = request.json
         scene_data.append(data)
         print(f"Received: {data['shape']} at ({data['x']}, {data['y']}) with confidence {data['confidence']}")
         return jsonify(success=True)

     @app.route('/')
     def render_scene():
         # Render scene using model folder (e.g., /models/cylinder.obj)
         return render_template('scene.html', scene_data=scene_data)

     if __name__ == '__main__':
         app.run(host='0.0.0.0', port=5000)
     ```
   - Create a simple `templates/scene.html` for rendering (work in progress):
     ```html
     <!DOCTYPE html>
     <html>
     <head>
         <title>Shape Scene Replica</title>
     </head>
     <body>
         <h1>Robot Scene Replica</h1>
         <div id="scene">
             {% for item in scene_data %}
                 <p>{{ item.shape }} at ({{ item.x }}, {{ item.y }}) with confidence {{ item.confidence }}</p>
                 <!-- Placeholder for 3D model rendering (e.g., Three.js) -->
             {% endfor %}
         </div>
     </body>
     </html>
     ```
   - Run:
     ```bash
     python flask_server.py
     ```
   - Note: Scene rendering with 3D models is ongoing, using a model folder (`/models`) with `.obj` files for shapes (e.g., `cylinder.obj`).

### Project Documentation

**For Software:**

#### Screenshots
![Training Output](https://github.com/AaronBoshMacsimus/UseLessProj/blob/main/trainingoutput.png)
*Shows the training process with dataset loading and accuracy (e.g., 0.990 for 900 images).*

![Hardware Setup](https://github.com/AaronBoshMacsimus/UseLessProj/blob/main/snapshot.jpg)
*Shows the physical setup with Arduino/microcontroller, breadboard connections, and LED components for the project.*

![Initial State](https://github.com/AaronBoshMacsimus/UseLessProj/blob/main/reactwebbeforechangeinrequest.jpg)
*Displays the web interface showing yellow circle and purple rectangle in their starting positions before collision detection.*

![Animation State](https://github.com/AaronBoshMacsimus/UseLessProj/blob/main/reactwebafterchangeinrequest.jpg)
*Shows the animated state with purple rectangle moved to top-left and yellow circle repositioned during the interaction sequence.*

#### Diagrams
![Workflow](https://github.com/<your-username>/UseLessProj/blob/main/reactwebafterchangeinrequest.jpg)
*Illustrates the pipeline: robot camera capture -> shape classification -> position detection -> web scene rendering with model folder.*

**For Hardware:**

#### Schematic & Circuit
![Circuit](https://github.com/AaronBoshMacsimus/UseLessProj/blob/main/UselessWrokImg.jpg)
*Shows connections: Raspberry Pi Zero 2 W to Camera Module, L298N motor driver to DC motors, MPU-6050 IMU, and battery pack.*

### Project Demo

#### Video
[Demo Video Link](https://github.com/<your-username>/UseLessProj/bolb/main/raspberry pi ai demo.mp4)
*Demonstrates the robot moving, detecting a pillar, classifying it as a cylinder, calculating its (x,y) position, and rendering a cylinder model on the web page at the correct position (rendering in progress).*

#### Additional Demos
- [Live Web Scene Replica](http://<web-server-ip>:5000): Real-time rendering of the scene with positioned models (work in progress).
- [Feature Visualization Notebook](https://github.com/<your-username>/UseLessProj/feature_viz.ipynb): Jupyter notebook for analyzing shape and position features.

### Project Status
The core classification model and position detection are complete, enabling the robot to identify shapes (e.g., pillars as cylinders) and their (x,y) coordinates. The web transmission of classification and position data is fully implemented. **Positioning and scene rendering with 3D models on the web page is ongoing**, with the Flask server set up to receive data and a placeholder HTML template for future model rendering (e.g., using Three.js).

## Team Contributions
- AARON BOSH MACSIMUS: Developed shape classification model, implemented position detection, and set up web data transmission. Built self-balancing robot chassis, integrated camera and IMU, and implemented PID balance control.
and Developed Flask server, worked on scene rendering (ongoing), and curated dataset with geometric models.

---

Made with ❤️ at TinkerHub Useless Projects  
![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)  
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)
