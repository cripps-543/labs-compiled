import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
inputVid = "task1.mp4"

cap = cv2.VideoCapture(inputVid)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break


    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(greyFrame, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60))

    for (x,y,w,h) in faces:
        # print(f'x:{x} y{y} w:{w} H{h}')
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Task 9", frame)

    if cv2.waitKey(25) and 0xFF == ord('q'):
        break

cap.release()
