from picamera2 import Picamera2, Preview
import cv2, time, os

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)}
))
picam2.start_preview(Preview.NULL)  # no GUI required
picam2.start()
time.sleep(0.2)  # warm-up

frame = picam2.capture_array()            # (H,W,4) RGBA-like
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
cv2.imwrite("test_picamera2.jpg", frame_bgr)
print("Saved test_picamera2.jpg")

picam2.stop()
