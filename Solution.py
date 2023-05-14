import torch
import cv2
import sys
import numpy as np
import time
from math import sqrt
from arguments import *
from src.qt.ui import MainUI

from detect import YOLO_Detect
from track import YOLO_DEEPSORT

from torch.autograd import Variable
from src.cv_utils.projectPoints import Maper, CV2RM, red2blue
from src.myserial.Serial import SerialSender

lr = np.array([156, 43, 46])  #红色hsv范围下限
ur = np.array([180, 255, 255])  #红色hsv范围上限

lb = np.array([100, 43, 46])  #蓝色hsv范围下限
ub = np.array([124, 255, 255])   #蓝色hsv范围下限

class Car:
    def __init__(self, bbox) -> None:
        self.bbox = bbox
        self.serial_id = 0

        self.color = 'unknow'
        self.armor = [0, 0, 0, 0, 0, 0, 0]
        self.pos = []
        self.live = 5

    def getid(self, model, src, armor, device):
        """
        args : model, src, armor, device
        return: None
        """
        x1, y1, x2, y2 = armor
        crop = src[y1: y2, x1 : x2]
        crop = np.array(crop)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (22, 30))
        crop = np.reshape(crop.astype(np.float32) / 255.0, (1, 1, crop.shape[0], crop.shape[1]))
        crop = torch.from_numpy(crop)
        input = Variable(crop).to(device)
        out = model(input)
        self.armor[np.argmax(out[0].tolist())] += 1
        serial_shifting = 100 if self.color == 'blue' else 0
        self.serial_id = max(self.armor) + serial_shifting + 1

    def getColor(self, src, armor):
        x1, y1, x2, y2 = armor
        crop = src[y1: y2, x1 : x2]
        crop = np.array(crop)
        cv2.resize(crop, (25, 25))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        inRange_Red = cv2.inRange(hsv, lr, ur)  #滤去颜色
        inRange_Red.reshape((inRange_Red.shape[0] * inRange_Red.shape[1]))
        red_count = inRange_Red.size - inRange_Red.tolist().count(0)
        inRange_Blue = cv2.inRange(hsv, lb, ub)  #滤去颜色
        inRange_Blue.reshape((inRange_Blue.shape[0] * inRange_Blue.shape[1]))
        blue_count = inRange_Blue.size - inRange_Blue.tolist().count(0)
        if max([blue_count, red_count]) < 10:
            return
        self.color = "blue" if blue_count > red_count else "red"


def IsEnemy(color):
    enemy = "blue" if is_bule else "red"
    return color == enemy

def IsINbbox(car_bbox, armor_bbox):
    return car_bbox[0] <= armor_bbox[0] <= car_bbox[0] + car_bbox[2] \
        and car_bbox[1] <= armor_bbox[1] <= car_bbox[1] + car_bbox[3]   


class Solutionv2:
    def __init__(self) -> None:
        self.exit_flag = -1

        self.armor_model = torch.load('weights/2023_4_9_hj_num_3.pt').cuda()
        self.armor_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.car_dict = {}

        self.car_bboxs = []
        self.setlive = 5

    def yolo_deepsort_layer(self, yolo_deepsort : YOLO_DEEPSORT):
        self.armor_bboxs = yolo_deepsort.armor_bboxs
        self.src_img = yolo_deepsort.src_img

        pop_list = []
        for id, bbox in yolo_deepsort.car_bboxs:
            if self.car_dict.get(id) is None:
                self.car_dict[id] : Car = Car(bbox)
            else:
                self.car_dict[id].bbox = bbox
                self.car_dict[id].live = self.setlive
            self.car_dict[id].live -= 1
            if self.car_dict[id].live <= 0:
                self.car_dict[id].append(id)
        for id in pop_list:
            self.car_dict.pop(id)
        
    def armor_color_layer(self):
        for car in self.car_bboxs:
            for armor_bbox in self.armor_bboxs:
                if IsINbbox(car.bbox, armor_bbox):
                    car.getColor(self.src_img, armor_bbox)
                    if IsEnemy(car.color):
                        car.getid(
                            self.armor_model, 
                            self.src_img, 
                            armor_bbox,
                            self.device
                            )
                        
    def project_layer(self, maper_points):
        serial_out = []
        for car in self.car_bboxs:
            car_center_x = car[0] + int((car[2] - car[0]) / 2)
            car_center_y = car[3] # y2
            min_error = 200000 #pixel
            min_point_3d = []
            for i in range(len(maper_points)):
                point_3d, point_2d = maper_points[i]
                x, y = point_2d[0], point_2d[1]
                error = sqrt((car_center_x - x)*(car_center_x - x) + 
                             (car_center_y - y)*(car_center_y - y) +
                             100 * (1 + point_3d[1])
                             )
                if error < min_error:
                    min_error = error
                    min_point_3d = point_3d
            min_point_3d = CV2RM(min_point_3d)
            if is_bule:
                min_point_3d = red2blue(min_point_3d)
            min_point_3d[0] += 0.3
            car.pos = min_point_3d
            serial_out.append([car.serial_id, car.pos])
        return serial_out

    def backbone(self,
                 mainWindow: MainUI, 
                 yolo_deepsort: YOLO_DEEPSORT, 
                 serial_sender: SerialSender):
        while 1:
            # t1 = time.time()
            image = np.array(yolo_deepsort.src_img)
            

            if len(image):
                
                maper = Maper(camera_x = mainWindow.getCameraX(),
                            camera_y = mainWindow.getCameraY(),
                            camera_z = mainWindow.getCameraZ(),
                                r_x = mainWindow.getPitch(),
                                yaw = mainWindow.getYaw(),
                                roll= mainWindow.getRoll(),
                                points_dis = mainWindow.getPointsDis())
                maper_points = maper.get_points_map()
                vision_img = maper.draw_points_2d(image)
                mainWindow.img = vision_img

                self.yolo_deepsort_layer(yolo_deepsort)
                
                self.armor_color_layer()
                
                serial_sender.Send(self.project_layer(maper_points))

            # print(f'took {time.time() - t1:.3f}s')

            if self.exit_flag == 0:
                break
            time.sleep(0.02)
        print("\nexit!")
        exit(0)
    
    def Exit(self, 
             yolo_detect: YOLO_DEEPSORT):
        for i in range(10):
            yolo_detect.exit_flag = self.exit_flag
            time.sleep(0.1)
        print("exit_args:", self.exit_flag)
        sys.exit(self.exit_flag)



class Solutionv1:
    def __init__(self) -> None:
        self.exit_flag = -1

    def deal_yolo_out(yolo_out):
        new_output = yolo_out[-1]
        car_xyxy = []
        armors = []
        for line in new_output:
            # print("\nline:", line)
            cls, *xyxy, conf = line
            cls = int(cls)
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
                error = sqrt((car_center_x - x)*(car_center_x - x) + 
                             (car_center_y - y)*(car_center_y - y) +
                             100 * (1 + point_3d[1])
                             )
                if error < min_error:
                    min_error = error
                    min_point_3d = point_3d
            min_point_3d = CV2RM(min_point_3d)
            if is_bule:
                min_point_3d = red2blue(min_point_3d)
            min_point_3d[0] += 0.3
            xcnp_map.append([id, min_point_3d])
        return xcnp_map
    
    def backbone(self,
                 mainWindow: MainUI, 
                 yolo_detect: YOLO_Detect, 
                 serial_sender: SerialSender):

        while 1:
            image = np.array(yolo_detect.src_image)
            
            yolo_out = yolo_detect.output

            if len(image):

                maper = Maper(camera_x = mainWindow.getCameraX(),
                            camera_y = mainWindow.getCameraY(),
                            camera_z = mainWindow.getCameraZ(),
                                r_x = mainWindow.getPitch(),
                                yaw = mainWindow.getYaw(),
                                roll= mainWindow.getRoll(),
                                points_dis = mainWindow.getPointsDis())
                maper_points = maper.get_points_map()
                vision_img = maper.draw_points_2d(image)
                mainWindow.img = vision_img
                if len(yolo_out):
                    xcn_map = self.deal_yolo_out(yolo_out)
                    if xcn_map:
                        xcnp_map = self.map_pos(xcn_map, maper_points)
                        if xcnp_map:
                            print("send")
                            serial_sender.Send(xcnp_map)

            if self.exit_flag == 0:
                break
            time.sleep(0.02)
        print("\nexit!")
        exit(0)
    
    def Exit(self, 
             yolo_detect: YOLO_Detect):
        for i in range(10):
            yolo_detect.exit_flag = self.exit_flag
            time.sleep(0.1)
        print("exit_args:", self.exit_flag)
        sys.exit(self.exit_flag)