# Lightweight Geometric Shape Classifier

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-green.svg)
![Raspberry Pi](https://img.shields.io/badge/raspberry_pi-zero_2_w-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

The **Lightweight Geometric Shape Classifier** is a Python-based machine learning project designed to classify images of 2D and 3D geometric shapes, such as circles, squares, cylinders (e.g., pillars), and spheres, with minimal computational resources. Optimized for the Raspberry Pi Zero 2 W (512MB RAM), it uses OpenCV for image processing and a lightweight classifier (k-NN or Decision Tree) to achieve efficient shape classification. The project is ideal for embedded systems and edge devices, with a focus on low memory usage and fast inference.

### Features
- Supports classification of 18 geometric shapes:
  - **2D Shapes**: Circle, Square, Rectangle, Triangle, Pentagon, Hexagon, Octagon, Oval, Parallelogram, Rhombus
  - **3D Shapes**: Cylinder, Sphere, Cube, Cone, Pyramid, Cuboid, Rectangular Prism, Triangular Prism
- Lightweight feature extraction using contour analysis, Hough transforms, and ellipse fitting
- Memory-efficient design with garbage collection and small image sizes (128x128 pixels)
- Optional k-NN or Decision Tree classifier for improved accuracy
- Interactive prediction loop for real-time testing
- Memory usage monitoring for Raspberry Pi compatibility

## Installation

### Prerequisites
- Python 3.7 or higher
- Raspberry Pi Zero 2 W (or compatible device) for deployment
- Git (to clone the repository)

### Dependencies
Install the required Python packages:
```bash
pip install opencv-python numpy scikit-learn psutil
```

On a Raspberry Pi, ensure compatibility with the ARM architecture:
```bash
pip install opencv-python-headless numpy scikit-learn psutil
```

### Clone the Repository
```bash
git clone https://github.com/<your-username>/lightweight-shape-classifier.git
cd lightweight-shape-classifier
```

## Usage

### Dataset Preparation
The classifier requires a dataset organized in the following structure:
```
dataset/
├── circle/
│   ├── image1.jpg
│   └── ...
├── square/
├── rectangle/
├── triangle/
├── cylinder/
│   ├── pillar1.jpg
│   ├── soda_can.jpg
│   └── ...
├── sphere/
├── cube/
├── cone/
├── pentagon/
├── hexagon/
├── octagon/
├── oval/
├── parallelogram/
├── rhombus/
├── pyramid/
├── cuboid/
├── rectangular_prism/
├── triangular_prism/
```
- Each subdirectory should contain images (`.png`, `.jpg`, `.jpeg`, `.bmp`) of the corresponding shape.
- Aim for at least 50 images per class for robust training (up to 50 images per class are loaded by default).
- For 3D shapes like `cylinder` (e.g., pillars, soda cans), include images with varying angles, lighting, and backgrounds.

### Training the Model
1. Run the script:
   ```bash
   python geoImg2.py
   ```
2. If no model exists (`lightweight_shape_model.pkl`), the script prompts for the dataset path:
   ```
   Enter dataset path: /path/to/your/dataset
   ```
3. The script loads up to 50 images per class, extracts 18 features (e.g., area, circularity, Hough circles, lines), trains a Decision Tree classifier (default), and saves the model.
4. Example output:
   ```
   ==================================================
   LIGHTWEIGHT SHAPE CLASSIFIER FOR RASPBERRY PI
   ==================================================
   Optimized for Raspberry Pi Zero 2 W (512MB RAM)
   No existing model found. Training new model...
   Enter dataset path: D:\AAron\downloaded_dataset
   Loading dataset (memory-efficient mode)...
   Processing circle...
   Loaded 50 images for circle
   Processing cylinder...
   Loaded 50 images for cylinder
   ...
   Dataset loaded: 900 images, 18 classes
   Training lightweight tree classifier...
   Training completed! Accuracy: 0.850
   Model saved: lightweight_shape_model.pkl (10.80 KB)
   Model memory usage: 10.8 KB
   ```

### Making Predictions
1. After training or loading an existing model, enter the path to an image for classification:
   ```
   Enter image path (or 'quit' to exit): D:\AAron\downloaded_dataset\cylinder\pillar1.jpg
   ```
2. Example output:
   ```
   Prediction: CYLINDER (3D)
   Confidence: 0.800
   ```
3. Type `quit` to exit the prediction loop.

### Testing Memory Usage
To verify the script’s compatibility with the Raspberry Pi Zero 2 W:
```bash
python geoImg2.py --test-memory
```
Example output:
```
Memory Usage Test
------------------------------
Initial memory: 50.2 MB
After training: 55.4 MB
Model memory: 10.8 KB
Total increase: 5.2 MB
✅ Memory usage suitable for Raspberry Pi Zero 2 W
```

## Dataset Requirements
- **Image Formats**: `.png`, `.jpg`, `.jpeg`, `.bmp`
- **Quantity**: At least 50 images per class recommended for good accuracy
- **Diversity**: Include images with different angles, lighting, and backgrounds
- **Sources**:
  - Collect images from open datasets (e.g., Open Images, ImageNet)
  - Take photos of real objects (e.g., pillars for cylinders, dice for cubes)
  - Use synthetic images generated with tools like Blender for 3D shapes

## Troubleshooting
- **Error: "Feature count mismatch"**:
  - Delete the existing model (`lightweight_shape_model.pkl`) and retrain with the current code:
    ```bash
    rm lightweight_shape_model.pkl
    python geoImg2.py
    ```
- **Low Accuracy**:
  - Increase the number of images per class (e.g., 50-100).
  - Verify feature quality using the visualization function (add to `main_raspberry_pi`):
    ```python
    import matplotlib.pyplot as plt
    def visualize_features(classifier, image_path):
        gray, edges = classifier.preprocess_image_lightweight(image_path)
        if gray is None:
            print("Failed to load image")
            return
        features = classifier.extract_minimal_features(gray, edges)
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale Image')
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        plt.show()
        print("Features:", features)
        print("Feature names: [area, perimeter, circularity, aspect_ratio, extent, solidity, vertices, w, h, cx, cy, moment_ratio, mean_intensity, std_intensity, edge_density, circle_count, line_count, ellipse_ratio]")
    visualize_features(classifier, "D:\\AAron\\downloaded_dataset\\cylinder\\pillar1.jpg")
    ```
- **Invalid Image Path**:
  - Ensure the path points to a valid image file, not a directory (e.g., `D:\AAron\downloaded_dataset\cylinder\pillar1.jpg`).
- **No Cylinder Classification**:
  - Ensure the dataset includes a `cylinder` subdirectory with images of pillars, soda cans, etc., and retrain the model.

## Deployment on Raspberry Pi Zero 2 W
1. Transfer the script and dataset to the Raspberry Pi:
   ```bash
   scp geoImg2.py pi@<pi-ip>:/home/pi/
   scp -r /path/to/dataset pi@<pi-ip>:/home/pi/dataset
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python-headless numpy scikit-learn psutil
   ```
3. Run the script:
   ```bash
   python geoImg2.py
   ```
4. Test memory usage:
   ```bash
   python geoImg2.py --test-memory
   ```

## Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [OpenCV](https://opencv.org/) for image processing
- Optimized for the Raspberry Pi Zero 2 W
- Inspired by shape detection tutorials from [PyImageSearch](https://pyimagesearch.com/) and Stack Overflow