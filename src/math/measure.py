import cv2
from math import atan, degrees, radians, tan, sqrt, cos
import numpy as np

L = 15.6e-3
F = 6e-3
W = 1080
H = 4
pitch = 60

ly = np.array([26,43,46])  
uy = np.array([34,255,255])  


src = cv2.imread("res/yellow.png")
print(src.shape)
hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
img = cv2.inRange(hsv_img , ly, uy)

in_yellow = False
for index, raw in zip(range(len(img)), img):
    start_index = 0
    end_index = 0
    is_start = False
    is_end = False
    for i, pixel in zip(range(len(raw)), raw):
        if pixel != 0:
            if not is_start:
                start_index = i
                is_start = True
        else:
            if not is_end and is_start:
                end_index = i
                is_end = True

    line_length = 0
    if is_start:
        line_length = end_index - start_index + 1
        if not in_yellow:
            in_yellow = True
            h_pixel_length = L / W
            w_pixel_length = (27.7e-3 / 1920)
            r = (int(W / 2) - index) * h_pixel_length
            deg = pitch + degrees(atan(r / F))
            x = tan(radians(deg)) * H
            cal_x = sqrt(x ** 2 + 16) * cos(radians(deg - 60))
            cal_line_length = 16 / cal_x * F / w_pixel_length
            # print(f'cal_line_length:{cal_line_length:.2f}')
            print(f'length:{line_length}, error = {line_length - cal_line_length:.2f}')
            true_length = (line_length * w_pixel_length) * cal_x / F
            print(f'true_length:{true_length:.2f}m')
            print(f'The {index} raw, {deg:.2f} °, x = {x:.2f}m')

    else:
        in_yellow = False
        
# cv2.namedWindow('test',cv2.WINDOW_NORMAL)
# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

class Position:
    def __init__(self) -> None:
        pass

class YOLO_cls:
    def __init__(self) -> None:
        pass


class Instance:
    def __init__(self) -> None:
        self.position = Position()
        self.cls = YOLO_cls()