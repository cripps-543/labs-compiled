import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(greyFrame, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

    for (x,y,w,h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if len(faces)>0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x,y ), (x+w, y+h), (0, 255, 0), 2)
        face_crop = frame[y:y+h, x:x+w]
        cv2.imshow('Face', face_crop)

    cv2.imshow("Task 8", frame)
    

    if cv2.waitKey(25) and 0xFF == ord('q'):
        break
