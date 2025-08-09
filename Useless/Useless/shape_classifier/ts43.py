from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start_preview(Preview.NULL)
picam2.start()
time.sleep(1)
frame = picam2.capture_array()
print("Captured frame shape:", frame.shape)
picam2.stop()
