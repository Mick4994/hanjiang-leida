import threading
import sys
import time
import numpy as np
from math import sqrt

from PyQt5.QtWidgets import QApplication, QMainWindow

from arguments import *

from detect import YOLO_Detect
from src.cv_utils.projectPoints import Maper, CV2RM, red2blue
from src.myserial.Serial import SerialSender
from src.qt.ui import MainUI

cls2serial = np.array([0, 0, 0] + red_robot_id_serial + [0, 0] + blue_robot_id_serial + [0, 0],
                       dtype=np.uint16)

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
            if is_bule:
                if cls == 0: # 车标签
                    car_xyxy.append(xyxy)
                elif 3 <= cls <= 7:
                    armors.append([cls, xyxy])
            else:
                if cls == 0: # 车标签
                    car_xyxy.append(xyxy)
                elif 10 <= cls <= 14:
                    armors.append([cls, xyxy])
        xcn_map = []
        # xcn_map = zip(car_xyxy, np.zeros((len(car_xyxy), 4), dtype=np.int32))
        for car in car_xyxy:
            car_x1, car_y1, car_x2, car_y2 = car
            for cls, armor in armors:
                armor_x1, armor_y1, armor_x2, armor_y2 = armor
                if armor_x1 > car_x1 and armor_y1 > car_y1:
                    if armor_x2 < car_x2 and armor_y2 < car_y2:
                        xcn_map.append([car, cls2serial[cls]])
                        break
        return xcn_map
    
    def map_pos(xcn_map, maper_points):
        xcnp_map = []
        for car, id in xcn_map:
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
            if is_bule:
                min_point_3d = red2blue(min_point_3d)
            min_point_3d[0] += 0.3
            xcnp_map.append([id, min_point_3d])
        return xcnp_map
    
    # def lock2car(self):
    #     pass
class Control_Thread_Data:
    def __init__(self) -> None:
        self.exit_flag = 1

def backbone(mainWindow: MainUI, yolo_detect: YOLO_Detect, ctd: Control_Thread_Data):
    serial_sender = SerialSender(com=COM)

    # 等待启动
    # time.sleep(10)

    start_time = time.time()

    while 1:
        image = np.array(yolo_detect.src_image)
        yolo_out = yolo_detect.output
        if len(image):
            # image = cv2.resize(image, (1600, 1200))
            # print("image.shape", image.shape)
            # maper.draw_points_2d(image)
            maper = Maper(camera_x = mainWindow.getCameraX(),
                          camera_y = mainWindow.getCameraY(),
                          camera_z = mainWindow.getCameraZ(),
                               r_x = mainWindow.getPitch())
            maper_points = maper.get_points_map()
            vision_img = maper.draw_points_2d(image)
            # vision_img = maper.draw_points_noshow(image)
            mainWindow.img = vision_img
            if len(yolo_out):
                xcn_map = Solution.deal_yolo_out(yolo_out)
                if xcn_map:
                    xcnp_map = Solution.map_pos(xcn_map, maper_points)
                    if xcnp_map:
                        serial_sender.Send(xcnp_map)
        print(f'running {time.time() - start_time:>7.2f}s', end='\r')
        # if not detect_thread.is_alive():
        #     break
        if ctd.exit_flag == 0:
            break
        time.sleep(0.02)
    print("\nexit!")
    exit(0)


if __name__ == "__main__":
    ctd = Control_Thread_Data()
    yolo_detect = YOLO_Detect(source=SOURCE)
    detect_thread = threading.Thread(target=yolo_detect.detect, args=(ctd, ), daemon=True)
    detect_thread.start()

    leida_app = QApplication(sys.argv)
    mainWindow = MainUI()
    mainWindow.show()
    backbone_thread = threading.Thread(target=backbone, args=(mainWindow, yolo_detect, ctd), daemon=True)
    backbone_thread.start()
    ctd.exit_flag = leida_app.exec_()
    yolo_detect.exit_flag = ctd.exit_flag
    print("exit_args:", ctd.exit_flag)
    sys.exit(ctd.exit_flag)