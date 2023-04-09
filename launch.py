import threading
import cv2
import time
import numpy as np
from math import sqrt

from detect import YOLO_Detect
from src.cv_utils.projectPoints import Maper, CV2RM

class Car:
    def __init__(self, num, color) -> None:
        # 最终信息数据
        self.position = []
        self.num = num
        self.color = color
        # 中间过程数据
        self.image = []
        self.xyxy = []

    def updata(self, position):
        self.position = position

class Solution:
    def deal_yolo_out(yolo_out):
        new_output = yolo_out[-1]
        car_xyxy = []
        armors = []
        for line in new_output:
            cls, *xyxy, conf = line
            if cls == 0: # 车标签
                # print(xyxy)
                car_xyxy.append(xyxy)
            elif cls >=  3:
                # print(xyxy)
                armors.append(xyxy)
        xcn_map = []
        # xcn_map = zip(car_xyxy, np.zeros((len(car_xyxy), 4), dtype=np.int32))
        for car in car_xyxy:
            car_x1, car_y1, car_x2, car_y2 = car
            for armor in armors:
                armor_x1, armor_y1, armor_x2, armor_y2 = armor
                if armor_x1 > car_x1 and armor_y1 > car_y1:
                    if armor_x2 < car_x2 and armor_y2 < car_y2:
                        xcn_map.append([car, armor])
                        break
        return xcn_map
    
    def map_pos(xcn_map, maper_points):
        xcnp_map = []
        for car, armor in xcn_map:
            car_center_x = car[0] + int((car[2] - car[0]) / 2) # x2 - x1
            car_center_y = car[3] # y2
            min_error = 200000 #pixel
            min_point_3d = []
            for i in range(len(maper_points)):
                point_3d, point_2d = maper_points[i]
                x, y = point_2d[0], point_2d[1]
                error = sqrt((car_center_x - x)*(car_center_x - x) + (car_center_y - y)*(car_center_y - y))
                if error < min_error:
                    min_error = error
                    min_point_3d = point_3d
            min_point_3d = CV2RM(min_point_3d)
            min_point_3d[0] += 0.3
            xcnp_map.append([car, armor, min_point_3d])
        return xcnp_map
    
    # def lock2car(self):
    #     pass

if __name__ == "__main__":

    yolo_detect = YOLO_Detect(source="1600x1200.png")
    maper = Maper(camera_x = -4.8, camera_y = 3.8, camera_z = 1.2, r_x = 20)
    maper_points = maper.get_points_map()

    detect_thread = threading.Thread(target=yolo_detect.detect, daemon=True)
    detect_thread.start()
    
    is_print = False
    while 1:
        image = np.array(yolo_detect.src_image)
        yolo_out = yolo_detect.output
        if len(image):
            image = cv2.resize(image, (1600, 1200))
            # print("image.shape", image.shape)

            maper.draw_points_2d(image)
            if len(yolo_out):
                xcn_map = Solution.deal_yolo_out(yolo_out)
                if xcn_map:
                    xcnp_map = Solution.map_pos(xcn_map, maper_points)
                    # if xcnp_map:
                    if not is_print:
                        print()
                        for car in xcnp_map:
                            print(car)
                        is_print = True
        time.sleep(0.02)