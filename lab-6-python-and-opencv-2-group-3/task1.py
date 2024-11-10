import cv2

videoPath = "task1.mp4"

cap = cv2.VideoCapture(videoPath)

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()