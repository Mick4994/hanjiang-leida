import cv2
import numpy as np

left_image = cv2.imread('res/fix1.png')
right_image = cv2.imread('res/fix2.png')

rate = 1

merge_image = np.hstack((left_image, right_image))
resize_shape = (int(merge_image.shape[1]/rate), int(merge_image.shape[0]/rate))

resize_image = cv2.resize(merge_image, resize_shape)

print(resize_shape)

cv2.namedWindow('merge_image', cv2.WINDOW_NORMAL)
cv2.imshow('merge_image', resize_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('res/merge.png', resize_image)