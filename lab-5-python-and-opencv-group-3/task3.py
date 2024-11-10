import cv2

image = cv2.imread('task1.jpg')

cv2.line(image, (50, 50), (150, 150), (0,0,255), 2)
cv2.rectangle(image, (50, 50), (150, 150), (0,255,0), 1)
cv2.circle(image, (100, 100), 100, (255, 0, 0), 3)
cv2.imshow("Task3", image)
cv2.imwrite("task3.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()