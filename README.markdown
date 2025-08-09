<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />
Lightweight Geometric Shape Classifier 🎯


Basic Details
Team Name: BoshMacsis
Team Members

Team Lead: Aaron Bosh Macsimus - College Of Engineering Thalassery

Project Description
A lightweight machine learning system to classify 2D and 3D geometric shapes like circles, cylinders, and cubes using OpenCV and minimal resources, optimized for Raspberry Pi Zero 2 W.
The Problem (that doesn't exist)
The world is plagued by rogue shapes sneaking into photos, confusing everyone with their geometric audacity—circles pretending to be spheres, cylinders masquerading as pillars!
The Solution (that nobody asked for)
Unleash a Raspberry Pi-powered shape sheriff to detect and classify these sneaky shapes with laser-focused contour analysis and a sprinkle of machine learning magic, saving the day one polygon at a time!
Technical Details
Technologies/Components Used
For Software:

Languages used: Python 3.7+
Frameworks used: None
Libraries used: OpenCV, NumPy, scikit-learn, psutil
Tools used: Git, Raspberry Pi Zero 2 W
For Hardware:
Main components: Raspberry Pi Zero 2 W, 2 BO MOTORS ,camera module (optional for real-time capture),GY91 SENSOR, 2 RECHARGABLE BATTERY, MOTOR DRIVER
Specifications: 512MB RAM, ARM architecture
Tools required: MicroSD card, USB power supply

Implementation
For Software:
Installation
pip install opencv-python-headless numpy scikit-learn psutil
git clone https://github.com/<your-username>/lightweight-shape-classifier.git
cd lightweight-shape-classifier

Run
python geoImg2.py

For testing memory usage:
python geoImg2.py --test-memory

Project Documentation
For Software:
Screenshots
![Screenshot1](Add screenshot of dataset loading here)Shows the script loading the dataset with 50 images per class for 18 shapes.![Screenshot2](Add screenshot of prediction output here)Displays the prediction result for a cylinder image with confidence score.![Screenshot3](Add screenshot of feature visualization here)Visualizes grayscale image and edge detection for feature extraction debugging.
Diagrams
![Workflow](Add your workflow/architecture diagram here)Illustrates the pipeline: image preprocessing, feature extraction, classifier training, and prediction.
For Hardware:
Schematic & Circuit
![Circuit](Add your circuit diagram here)Shows connections for Raspberry Pi Zero 2 W with optional camera module and power supply.![Schematic](Add your schematic diagram here)Details the Raspberry Pi Zero 2 W setup for running the classifier.
Build Photos
![Components](Add photo of your components here)Components: Raspberry Pi Zero 2 W, MicroSD card, USB power cable.![Build](Add photos of build process here)Steps: Flash MicroSD with Raspberry Pi OS, install dependencies, transfer dataset and script.![Final](Add photo of final product here)Final setup: Raspberry Pi Zero 2 W running the shape classifier with a connected display or SSH interface.
Project Demo
Video
[Add your demo video link here]Demonstrates the classifier identifying shapes like cylinders and cubes from sample images in real-time.
Additional Demos
[Add any extra demo materials/links, e.g., sample dataset images or feature visualization outputs]
Team Contributions

AARON BOSH MACSIMUS: ALL DONE BY ME

Made with ❤️ at TinkerHub Useless Projects
![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)
