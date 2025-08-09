import cv2
import numpy as np
import os
import pickle
import json
from collections import defaultdict
import gc  # Garbage collection for memory management
import warnings

warnings.filterwarnings('ignore')


class LightweightShapeClassifier:
    """
    Ultra-lightweight geometric shape classifier optimized for Raspberry Pi Zero 2 W
    Memory optimized with minimal dependencies and efficient algorithms
    """

    def __init__(self):
        self.model_params = None
        self.feature_stats = None  # For feature normalization
        self.label_map = {}
        self.reverse_label_map = {}

        # Reduced shape classes for better performance
        self.shape_classes_2d = ['circle', 'square', 'rectangle', 'triangle']
        self.shape_classes_3d = ['cylinder', 'sphere', 'cube', 'cone']
        self.all_classes = self.shape_classes_2d + self.shape_classes_3d

        # Initialize label mappings
        for i, class_name in enumerate(self.all_classes):
            self.label_map[class_name] = i
            self.reverse_label_map[i] = class_name

    def preprocess_image_lightweight(self, image_path, target_size=(128, 128)):
        """Lightweight image preprocessing"""
        try:
            # Read image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = image_path

            if image is None:
                return None

            # Resize to smaller dimensions to save memory
            image = cv2.resize(image, target_size)

            # Convert to grayscale immediately to save memory
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Simple blur (lighter than Gaussian)
            blurred = cv2.blur(gray, (3, 3))

            # Edge detection with reduced parameters
            edges = cv2.Canny(blurred, 30, 100)

            return gray, edges

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def extract_minimal_features(self, gray, edges):
        """Extract minimal but effective features for classification"""
        features = []

        try:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return np.zeros(15)  # Return minimal feature vector

            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Basic geometric features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            if perimeter == 0:
                return np.zeros(15)

            # Key shape descriptors
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 1

            # Extent and solidity (memory efficient)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0

            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Approximate polygon (vertex count)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            vertices = len(approx)

            # Simple moments (only first few)
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                moment_ratio = moments['m20'] / moments['m00'] if moments['m00'] != 0 else 0
            else:
                cx, cy, moment_ratio = 0, 0, 0

            # Compile minimal feature set
            features = [
                area / 10000.0,  # Normalize area
                perimeter / 1000.0,  # Normalize perimeter
                circularity,
                aspect_ratio,
                extent,
                solidity,
                min(vertices, 10) / 10.0,  # Normalize vertices
                w / 128.0,  # Normalized width
                h / 128.0,  # Normalized height
                cx / 128.0,  # Normalized centroid
                cy / 128.0,
                moment_ratio,
                np.mean(gray) / 255.0,  # Average intensity
                np.std(gray) / 255.0,  # Intensity variation
                np.sum(edges > 0) / (128 * 128)  # Edge density
            ]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(15)

    def extract_features(self, image_path):
        """Main feature extraction function"""
        processed = self.preprocess_image_lightweight(image_path)
        if processed is None:
            return None

        gray, edges = processed
        features = self.extract_minimal_features(gray, edges)

        # Clean up memory
        del gray, edges
        gc.collect()

        return features

    def simple_knn_classifier(self, X_train, y_train, k=3):
        """Ultra-simple k-NN classifier implementation"""
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train
        self.k = k

        # Calculate feature statistics for normalization
        self.feature_stats = {
            'mean': np.mean(X_train, axis=0),
            'std': np.std(X_train, axis=0) + 1e-7  # Avoid division by zero
        }

    def normalize_features(self, features):
        """Normalize features using training statistics"""
        if self.feature_stats is None:
            return features

        return (features - self.feature_stats['mean']) / self.feature_stats['std']

    def predict_knn(self, features):
        """Predict using k-NN"""
        if self.X_train is None:
            return None, 0.0

        # Normalize test features
        features_norm = self.normalize_features(features.reshape(1, -1))

        # Calculate distances (Euclidean)
        distances = np.sqrt(np.sum((self.X_train - features_norm) ** 2, axis=1))

        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        # Vote (most common label)
        unique, counts = np.unique(k_labels, return_counts=True)
        predicted_label = unique[np.argmax(counts)]
        confidence = np.max(counts) / self.k

        return predicted_label, confidence

    def train_lightweight(self, X, y):
        """Train the lightweight classifier"""
        print("Training lightweight k-NN classifier...")

        # Convert labels to integers
        y_encoded = np.array([self.label_map[label] for label in y])

        # Use simple train/test split (80/20)
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        split_idx = int(0.8 * n_samples)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

        # Train k-NN
        self.simple_knn_classifier(X_train, y_train, k=5)

        # Evaluate
        correct = 0
        total = len(X_test)

        for i in range(total):
            pred_label, _ = self.predict_knn(X_test[i])
            if pred_label == y_test[i]:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Training completed! Accuracy: {accuracy:.3f}")

        return accuracy

    def predict(self, image_path):
        """Predict shape class for an image"""
        features = self.extract_features(image_path)
        if features is None:
            return None, 0.0

        pred_label, confidence = self.predict_knn(features)
        if pred_label is None:
            return None, 0.0

        predicted_class = self.reverse_label_map[pred_label]
        return predicted_class, confidence

    def save_model_lightweight(self, filepath):
        """Save lightweight model"""
        model_data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'k': self.k,
            'feature_stats': self.feature_stats,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'all_classes': self.all_classes
        }

        # Use highest compression
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Check file size
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Model saved: {filepath} ({size_mb:.2f} MB)")

    def load_model_lightweight(self, filepath):
        """Load lightweight model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']
        self.k = model_data['k']
        self.feature_stats = model_data['feature_stats']
        self.label_map = model_data['label_map']
        self.reverse_label_map = model_data['reverse_label_map']
        self.all_classes = model_data['all_classes']

        print(f"Lightweight model loaded from {filepath}")

    def get_memory_usage(self):
        """Get approximate memory usage"""
        total_size = 0

        if self.X_train is not None:
            total_size += self.X_train.nbytes
        if self.y_train is not None:
            total_size += self.y_train.nbytes

        return total_size / (1024 * 1024)  # Convert to MB


# Memory-efficient dataset loader
def load_dataset_memory_efficient(dataset_path, max_images_per_class=50):
    """Load dataset with memory constraints"""
    print("Loading dataset (memory-efficient mode)...")
    print(f"Scanning directory: {dataset_path}")

    classifier = LightweightShapeClassifier()
    X = []
    y = []

    # First, check what's in the dataset directory
    try:
        items = os.listdir(dataset_path)
        print(f"Found items: {items}")
    except Exception as e:
        print(f"Error reading directory: {e}")
        return np.array([]), np.array([])

    # Look for class directories
    found_classes = []
    for class_name in classifier.all_classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path) and os.path.isdir(class_path):
            found_classes.append(class_name)
        else:
            # Also check for common variations
            variations = [
                class_name.upper(),
                class_name.lower(),
                class_name.capitalize()
            ]
            for variation in variations:
                var_path = os.path.join(dataset_path, variation)
                if os.path.exists(var_path) and os.path.isdir(var_path):
                    found_classes.append(variation)
                    classifier.all_classes[classifier.all_classes.index(class_name)] = variation
                    break

    print(f"Found class directories: {found_classes}")

    if not found_classes:
        print("❌ No valid class directories found!")
        print(f"Looking for directories named: {classifier.all_classes}")
        print("Available directories in the path:")
        try:
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}")
        except:
            pass
        return np.array([]), np.array([])

    total_images_loaded = 0

    for class_name in found_classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            continue

        print(f"\n📂 Processing {class_name}...")
        count = 0

        try:
            all_files = os.listdir(class_path)
            image_files = [f for f in all_files
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            print(f"   Found {len(image_files)} image files")

            # Limit images per class to save memory
            image_files = image_files[:max_images_per_class]

            for filename in image_files:
                if count >= max_images_per_class:
                    break

                image_path = os.path.join(class_path, filename)
                print(f"   Processing: {filename}", end=" -> ")

                try:
                    features = classifier.extract_features(image_path)

                    if features is not None and len(features) > 0:
                        X.append(features)
                        y.append(class_name)
                        count += 1
                        total_images_loaded += 1
                        print("✓")
                    else:
                        print("✗ (no features)")

                    # Memory cleanup every 10 images
                    if count % 10 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"✗ (error: {e})")
                    continue

        except Exception as e:
            print(f"   Error reading directory: {e}")
            continue

        print(f"   ✅ Loaded {count} images for {class_name}")

    print(f"\n🎉 Total images loaded: {total_images_loaded}")

    if total_images_loaded == 0:
        print("\n❌ No images were successfully loaded!")
        print("Common issues:")
        print("1. Check image file formats (.jpg, .png, .bmp)")
        print("2. Ensure images are not corrupted")
        print("3. Check file permissions")

    return np.array(X, dtype=np.float32), np.array(y)


# Raspberry Pi optimized main function
# Quick training function for testing with minimal data
def quick_start_demo():
    """Quick demo with single image per class"""
    print("=" * 50)
    print("QUICK START DEMO MODE")
    print("=" * 50)
    print("This will help you test with individual images")

    classifier = LightweightShapeClassifier()

    # Create minimal training data
    print("Creating minimal training dataset...")
    X_demo = []
    y_demo = []

    # Ask user for sample images
    for class_name in classifier.all_classes:
        while True:
            image_path = input(f"Enter path to a {class_name} image (or 'skip'): ").strip()

            if image_path.lower() == 'skip':
                print(f"Skipping {class_name}")
                break

            if not os.path.exists(image_path):
                print("File not found! Try again.")
                continue

            features = classifier.extract_features(image_path)
            if features is not None:
                X_demo.append(features)
                y_demo.append(class_name)
                print(f"✓ Added {class_name} sample")
                break
            else:
                print("Could not process image. Try another one.")

    if len(X_demo) < 2:
        print("Need at least 2 samples to train!")
        return

    # Train with demo data
    X_demo = np.array(X_demo, dtype=np.float32)
    y_demo = np.array(y_demo)

    print(f"\nTraining with {len(X_demo)} samples...")
    accuracy = classifier.train_lightweight(X_demo, y_demo)

    # Save demo model
    classifier.save_model_lightweight('demo_model.pkl')

    # Test mode
    print("\nDemo model ready! Test with new images:")
    while True:
        test_path = input("Enter test image path (or 'quit'): ").strip()
        if test_path.lower() == 'quit':
            break

        if os.path.exists(test_path):
            prediction, confidence = classifier.predict(test_path)
            if prediction:
                shape_type = "3D" if prediction in classifier.shape_classes_3d else "2D"
                print(f"Prediction: {prediction.upper()} ({shape_type})")
                print(f"Confidence: {confidence:.3f}")
            else:
                print("Could not classify")
        else:
            print("File not found!")


def main_raspberry_pi():
    """Main function optimized for Raspberry Pi"""
    print("=" * 50)
    print("LIGHTWEIGHT SHAPE CLASSIFIER FOR RASPBERRY PI")
    print("=" * 50)
    print("Optimized for Raspberry Pi Zero 2 W (512MB RAM)")

    classifier = LightweightShapeClassifier()

    # Check if model exists
    model_path = 'lightweight_shape_model.pkl'

    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        classifier.load_model_lightweight(model_path)
        print(f"Model memory usage: {classifier.get_memory_usage():.1f} MB")

        # Skip to prediction mode
        print("Model loaded successfully! Ready for predictions.")

    else:
        print("No existing model found.")
        print("\nChoose training option:")
        print("1. Train with organized dataset folders")
        print("2. Quick demo with individual images")

        choice = input("Choose option (1/2): ").strip()

        if choice == '2':
            quick_start_demo()
            return

        elif choice == '1':
            print("\nDataset should be organized as:")
            print("dataset_folder/")
            print("├── circle/")
            print("│   ├── image1.jpg")
            print("│   ├── image2.jpg")
            print("├── cylinder/")
            print("│   ├── pillar1.jpg")
            print("│   ├── pillar2.jpg")
            print("└── ...")
            print("\nEnter the PATH TO THE MAIN DATASET FOLDER (not individual images)")

            dataset_path = input("Dataset folder path: ").strip()

            # Handle common mistakes
            if dataset_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print("❌ Error: You entered an image file path!")
                print("Please enter the folder containing the shape class folders.")
                parent_dir = os.path.dirname(dataset_path)
                suggestion = os.path.dirname(parent_dir)
                print(f"💡 Try: {suggestion}")
                return

            if not os.path.exists(dataset_path):
                print(f"❌ Dataset folder not found: {dataset_path}")
                print("Please check the path and try again.")
                return

            if not os.path.isdir(dataset_path):
                print("❌ Path is not a directory!")
                return

            # Load dataset with memory constraints
            X, y = load_dataset_memory_efficient(dataset_path, max_images_per_class=30)

            if len(X) == 0:
                print("\n❌ No images loaded! Please check your dataset structure.")
                print("💡 Try option 2 (Quick demo) to test with individual images.")
                return

            print(f"Dataset loaded: {len(X)} images, {len(set(y))} classes")

            # Train model
            accuracy = classifier.train_lightweight(X, y)

            # Save model
            classifier.save_model_lightweight(model_path)
            print(f"Model memory usage: {classifier.get_memory_usage():.1f} MB")
        else:
            print("Invalid choice!")
            return

    # Interactive prediction loop
    while True:
        print("\n" + "-" * 40)
        image_path = input("Enter image path (or 'quit' to exit): ").strip()

        if image_path.lower() == 'quit':
            break

        if not os.path.exists(image_path):
            print("Image not found!")
            continue

        try:
            prediction, confidence = classifier.predict(image_path)

            if prediction:
                shape_type = "3D" if prediction in classifier.shape_classes_3d else "2D"
                print(f"Prediction: {prediction.upper()} ({shape_type})")
                print(f"Confidence: {confidence:.3f}")
            else:
                print("Could not classify image")

        except Exception as e:
            print(f"Error: {e}")

        # Memory cleanup after each prediction
        gc.collect()


# Utility function to test memory usage
def test_memory_usage():
    """Test memory usage on Raspberry Pi"""
    import psutil

    print("Memory Usage Test")
    print("-" * 30)

    # Initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory: {initial_memory:.1f} MB")

    # Load classifier
    classifier = LightweightShapeClassifier()

    # Create dummy data
    X_dummy = np.random.rand(100, 15).astype(np.float32)
    y_dummy = np.random.choice(classifier.all_classes, 100)

    # Train
    classifier.train_lightweight(X_dummy, y_dummy)

    # Check memory after training
    after_training = process.memory_info().rss / 1024 / 1024
    print(f"After training: {after_training:.1f} MB")
    print(f"Model memory: {classifier.get_memory_usage():.1f} MB")
    print(f"Total increase: {after_training - initial_memory:.1f} MB")

    # Memory efficient enough for Pi Zero 2 W?
    if after_training < 200:  # Keep well under 512MB limit
        print("✅ Memory usage suitable for Raspberry Pi Zero 2 W")
    else:
        print("⚠️  Memory usage might be tight on Raspberry Pi Zero 2 W")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-memory":
        test_memory_usage()
    else:
        main_raspberry_pi()
