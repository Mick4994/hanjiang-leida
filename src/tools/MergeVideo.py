import cv2
import numpy as np

cap_left = cv2.VideoCapture('res/video_left.avi')
cap_right = cv2.VideoCapture('res/video_right.avi')

frame_gather = []

while(cap_left.isOpened() and cap_right.isOpened()):
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if ret_left and ret_right:
        merge_frame = np.hstack((frame_left, frame_right))
        frame_gather.append(merge_frame)
        cv2.namedWindow('merge_frame', cv2.WINDOW_NORMAL)
        cv2.imshow('merge_frame', merge_frame)
        if ord('b') == cv2.waitKey(1):
            break
    else:
        break

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
weight = frame_gather[0].shape[1]
hight = frame_gather[0].shape[0]
fps = 15
out = cv2.VideoWriter('merge.mp4',fourcc, fps, (weight,hight), True )
for frame in frame_gather:
    out.write(frame)

cv2.destroyAllWindows()
cap_left.release()
cap_right.release()
