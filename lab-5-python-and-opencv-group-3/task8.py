import cv2
from picamera2 import Picamera2, Preview
import numpy as np

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

while True:
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imshow("Task 8", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        np.save("task8.npy", frame)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()