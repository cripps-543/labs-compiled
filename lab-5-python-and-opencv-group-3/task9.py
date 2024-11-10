import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image = cv2.imread("task5.jpg")
greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(greyImage, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

for (x,y,w,h) in faces:
    print(f'x:{x} y{y} w:{w} H{h}')
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Task 9", image)
cv2.imwrite("task9.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
