import cv2
from picamera2 import Picamera2, Preview
import numpy as np
import time

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})

picam2.configure(camera_config)
picam2.start()

time.sleep(2)

frame_width = 640
frame_height = 480

frames = []

while True:
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frames.append(frame)

    cv2.imshow("Task 6", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

frames_array = np.array(frames, dtype=np.uint8)
np.save("task6.npy", frames_array)

picam2.stop()
cv2.destroyAllWindows()