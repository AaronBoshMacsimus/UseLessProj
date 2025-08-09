from picamera2 import Picamera2
import cv2, time

picam2 = Picamera2()
cfg = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
picam2.configure(cfg)
picam2.start()
time.sleep(0.2)  # warm-up

frame = picam2.capture_array()  # numpy array (H,W,4)
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

# If you have a GUI:
cv2.imshow("PiCam", frame_bgr)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Always save to file so it works headless too:
cv2.imwrite("test_picamera2.jpg", frame_bgr)
print("Saved test_picamera2.jpg")
import cv2

cap = cv2.VideoCapture(0)  # Default camera
if not cap.isOpened():
    print("Error: Could not access the camera.")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Camera", frame)
        cv2.waitKey(0)
    else:
        print("Failed to capture image")

cap.release()
cv2.destroyAllWindows()
