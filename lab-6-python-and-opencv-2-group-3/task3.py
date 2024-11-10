import cv2

videoPath = "task1.mp4"

cap = cv2.VideoCapture(videoPath)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Video with progress bar', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Video with progress bar', frame_width, frame_height)

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    progress = current_frame/total_frames

    bar_height = 20
    bar_width = int(progress*frame_width)
    bar_color = (0, 0, 255)
    bar_position = (0, frame_height - bar_height - 20)

    cv2.rectangle(frame, bar_position, (bar_width, frame_height - 20), bar_color, -1)

    cv2.imshow('Video with progress bar', frame)

    if cv2.waitKey(25) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()