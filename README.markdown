<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />
## 📝 Project Description
A **lightweight machine learning system** that runs on a Raspberry Pi Zero 2 W to classify **2D and 3D geometric shapes** (circles, cylinders, cubes, etc.) using OpenCV.  
The classifier is paired with a **Vite + React site** that lets you **interactively move and visualize shapes** in real time.

---

## 😂 The Problem (that doesn’t exist)
Geometric imposters are everywhere—cylinders pretending to be pillars, circles acting like spheres. Chaos! Confusion! Geometrical identity crises!

---

## 💡 The Solution (that nobody asked for)
A **shape sheriff** powered by Raspberry Pi and machine learning that detects and classifies these rogue forms.  
And because we can, a **React-based control panel** lets you move shapes around the screen like a bored deity rearranging the universe.

---

## 🛠 Technologies / Components Used

### **Software**
- **Languages:** Python 3.7+, JavaScript (ES6+)
- **Frameworks:** None (Python side), Vite + React (Frontend)
- **Libraries:**  
  - Python: `opencv-python-headless`, `numpy`, `scikit-learn`, `psutil`  
  - React: `react`, `socket.io-client`, `vite`
- **Tools:** Git, Raspberry Pi Zero 2 W, Node.js

### **Hardware**
- Raspberry Pi Zero 2 W (512MB RAM, ARM architecture)
- 2 BO Motors + Motor Driver
- GY91 Sensor
- Optional Camera Module
- 2 Rechargeable Batteries

---

## ⚙ Implementation

### **1. Raspberry Pi Shape Classifier**
**Pipeline:**
1. **Image Preprocessing** – Convert to grayscale, resize, normalize
2. **Feature Extraction** – Contour analysis, edge detection
3. **Classification** – Train a `scikit-learn` model on shape datasets
4. **Prediction** – Output shape name + confidence score

**Install & Run:**
```bash
pip install opencv-python-headless numpy scikit-learn psutil
git clone https://github.com/<your-username>/lightweight-shape-classifier.git
cd lightweight-shape-classifier
python geoImg2.py
Run
python geoImg2.py

For testing memory usage:
python geoImg2.py --test-memory `
```
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
