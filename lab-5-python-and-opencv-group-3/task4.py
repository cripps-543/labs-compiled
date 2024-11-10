import cv2
import numpy as np

image = cv2.imread('task1.jpg')
greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(greyImage, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(greyImage, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(sobel_combined)
cv2.imshow("Task4", sobel_combined)
cv2.imwrite("task4.jpg", sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()