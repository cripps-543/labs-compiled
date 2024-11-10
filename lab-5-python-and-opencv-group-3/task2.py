import cv2

image = cv2.imread('task1.jpg')
greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Task2", greyImage)
cv2.imwrite("task2.jpg", greyImage)
cv2.waitKey(0)
cv2.destroyAllWindows()