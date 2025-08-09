# live_classify.py
#from picamera2 import Piccamera2, Preview
from picamera2 import Picamera2, Preview

import cv2, time, os
from georeal import LightweightShapeClassifier  # or wherever your class lives

def main():
    clf = LightweightShapeClassifier()
    clf.load_model_lightweight("lightweight_shape_model.pkl")
    print("Model loaded. Starting camera... (press Ctrl+C to stop)")

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}
    )
    picam2.configure(cfg)
    picam2.start_preview(Preview.NULL)   # no display required
    picam2.start()
    time.sleep(0.2)  # warm-up

    try:
        while True:
            frame = picam2.capture_array()              # (H,W,4)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            features = clf.extract_features(frame_bgr)
            if features is not None:
                label_id, conf = clf.predict_knn(features)
                if label_id is not None:
                    pred = clf.reverse_label_map[label_id]
                    kind = "3D" if pred in clf.shape_classes_3d else "2D"
                    print(f"{pred.upper()} ({kind})  conf={conf:.3f}")
                else:
                    print("Could not classify frame")
            else:
                print("No features")

            # optional: save the latest frame for inspection
            cv2.imwrite("last_frame.jpg", frame_bgr)

            # small throttle so we don't max out CPU on Zero 2 W
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
