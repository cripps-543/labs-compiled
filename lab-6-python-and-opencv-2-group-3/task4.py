import cv2

inputVid = "task1.mp4"
outputVid = "task4.mp4"

cap = cv2.VideoCapture(inputVid)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(outputVid, fourcc, fps, (frame_width, frame_height))

kernel_size = (15, 15)

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)
    out.write(blurred_frame)

cap.release()
out.release()