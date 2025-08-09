from geoimg3 import LightweightShapeClassifier
import sys

if len(sys.argv) < 2:
    print("Usage: python3 test_features.py image_path")
    exit()

classifier = LightweightShapeClassifier()
features = classifier.extract_features(sys.argv[1])

if features is not None:
    print(f"Features extracted: {len(features)} values")
    print("Key features:")
    print(f"  Circularity: {features[2]:.3f}")
    print(f"  Aspect ratio: {features[3]:.3f}")
    print(f"  Vertices: {features[6]:.3f}")
    print(f"  Solidity: {features[5]:.3f}")
else:
    print("Failed to extract features")
