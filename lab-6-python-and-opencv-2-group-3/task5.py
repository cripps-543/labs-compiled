import cv2
from picamera2 import Picamera2, Preview
import time

outputVid = "task5.mp4"

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})

picam2.configure(camera_config)
picam2.start()

time.sleep(2)

frame_width = 640
frame_height = 480
fps = 10

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(outputVid, fourcc, fps, (frame_width, frame_height))

while True:
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    out.write(frame)

    cv2.imshow("Task 5", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

picam2.stop()
out.release()
cv2.destroyAllWindows()