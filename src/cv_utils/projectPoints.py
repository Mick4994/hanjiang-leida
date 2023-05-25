import numpy as np
import cv2
import time
from math import radians
from copy import deepcopy
try:
    from arguments import *
except:
    import sys
    import os
    sys.path.append(os.getcwd())
    from arguments import *

def CV2RM(points_3d):
    x, y, z = points_3d
    return [z, x, -y]

def RM2SER(points_3d):
    x, y, z = points_3d
    return [x, RM_FIELD_WEIGHT - y, z]

def red2blue(points_3d):
    x, y, z = points_3d
    return [RM_FIELD_LENGTH - x, RM_FIELD_WEIGHT - y, z]

class Maper:
    def __init__(self, 
            camera_x = -4.8, 
            camera_y = 3.8, 
            camera_z = 1.2, 
            r_x = 20,
            yaw = 0,
            roll = 0,
            points_dis = 24) -> None:
        # 相机外参
        self.camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
        self.camera_dir = np.array([radians(r_x), radians(yaw), radians(roll)], dtype=np.float32)
        self.points_dis = points_dis

        # 焦距与传感器尺寸
        self.focal_length = Focal_Length
        self.sensor_size_x = Sensor_X # 1/2.7 inch
        self.sensor_size_y = Sensor_Y
        self.pixel_size = Unit_Size

        self.mapper = []
        try:
            self.points_3d = np.load('src/cv_utils/' + Load_NPY)
            self.point_four_3d = np.load('src/cv_utils/' + Load_Four_NPY)
        except:
            self.points_3d = np.load(Load_NPY)
            self.point_four_3d = np.load(Load_Four_NPY)        

    def update(self, camera_x = -4.8, 
            camera_y = 3.8, 
            camera_z = 1.2, 
            r_x = 20,
            yaw = 0,
            roll = 0,
            points_dis = 24):
        self.camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
        self.camera_dir = np.array([radians(r_x), radians(yaw), radians(roll)], dtype=np.float32)
        self.points_dis = points_dis

    def get_points_map(self, image): # = np.zeros((1024, 1280, 3), dtype=np.uint8)

        points_3d = np.array(self.points_3d, dtype=np.float32)
        points_four_3d = np.array(self.point_four_3d, dtype=np.float32)
        self.points_3d = points_3d.copy()

        # print(image.shape)
        # 图像尺寸
        image_width = image.shape[1]
        image_height = image.shape[0]

        fx = self.focal_length / self.pixel_size
        fy = self.focal_length / self.pixel_size
        cx = image_width / 2
        cy = image_height / 2

        # 仿真内参
        K = np.array([[fx, 0, cx], 
                      [0, fy, cy], 
                      [0,  0,  1]], dtype=np.float32)
        # 真实内参
        # K = np.array([[1579.6, 0, 627.56], 
        #               [0, 1579.87, 508.65],
        #               [0, 0, 1]], dtype=np.float32)

        self.K = K
        # print(K)
        # 计算被旋转后的平移向量
        R, _ = cv2.Rodrigues(self.camera_dir) 
        # R 叉乘 camera_pos
        tvec = R @ self.camera_pos      
        # rvec, _ = cv2.Rodrigues(R) 

        points_2d, _ = cv2.projectPoints(
            points_3d, self.camera_dir, tvec, K, None)
        points_four_2d, _ = cv2.projectPoints(
            points_four_3d, self.camera_dir, tvec, K, None)
        self.points_four_2d = np.array(points_four_2d, dtype=np.int32)
        self.points_2d = np.array(points_2d, dtype=np.int32)
        # print(self.points_2d.shape)
        self.reshape_points_four_2d = np.reshape(self.points_four_2d, (self.points_four_2d.shape[0], 2))
        self.reshape_points_2d = np.reshape(self.points_2d, (self.points_2d.shape[0], 2))

        maper_points = list(zip(self.points_3d, self.reshape_points_2d))
        # print('finished project!')
        self.get_all_maper()
        return maper_points
    
    def get_all_map(self, maper_points):
        all_maper = {}
        # for point_2d, point_3d
        return all_maper

    def draw_points_noshow(self, image):
        if self.points_dis:
            for p in self.points_2d[::self.points_dis]:
                cv2.circle(image, p[0], 5, (0, 255, 0), -1) 
        return image

    def draw_points_2d(self, image = np.zeros((SCREEN_H, SCREEN_W, 3))):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        if self.points_dis:
            for p in self.points_2d[::self.points_dis]:
                cv2.circle(image, p[0], 5, (0, 255, 0), -1) 

        cv2.imshow("image", image)
        if ord('b') == cv2.waitKey(1):
            cv2.destroyAllWindows()
            exit(0)
        return image

    def demo_display(self, image = np.zeros((SCREEN_H, SCREEN_W, 3))):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # i = 0
        for p in self.points_2d:
            cv2.circle(image, p[0], 9, (0, 255, 0), -1) 

        # 宣判死刑
            # print(p)
        #     p = p[0]
        #     # exit(0)
        #     if 0 <= p[1] < SCREEN_H and 0 <= p[0] < SCREEN_W:
        #         try:
        #             # print(i, end='\r')
        #             image[p[1]][p[0]] = (255, 255, 255)
        #             image[p[1] + 1][p[0] + 1] = (255, 255, 255)
        #             image[p[1] + 1][p[0] - 1] = (255, 255, 255)
        #             image[p[1] - 1][p[0] + 1] = (255, 255, 255)
        #             image[p[1] - 1][p[0] - 1] = (255, 255, 255)
        #         except KeyboardInterrupt:
        #             exit(0)
        #         except:
        #             # print('faild', end='\r')
        #             pass
        #     # i += 1
        # print("")
            # cv2.circle(image, p[0], 5, (0, 255, 0), -1) 
        
        kernel = np.ones((1, 5), np.uint8)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  
        _, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
        # cv2.imshow('binary', binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)
        contours, _ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
        print(contours[0].shape)
        cv2.drawContours(image,contours,-1,(0,0,255),3) 

        cv2.setMouseCallback("image", mouseCallback, 1)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def demo_display_slow(self, image = np.zeros((SCREEN_H, SCREEN_W, 3))):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        for p in self.reshape_points_2d:
            cv2.circle(image, p, 5, (0, 255, 0), -1) 
            
            cv2.imshow("image", image)
            if ord('b') == cv2.waitKey(1):
                cv2.destroyAllWindows()
                exit(0)

    def get_all_maper(self):
        self.image_3d = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        # print(len(self.reshape_points_four_2d))
        for i in range(len(self.reshape_points_four_2d))[::4]:
            # t1 = time.time()
            
            temp_points = [self.reshape_points_four_2d[i + j] for j in range(4)]
            k = int(i / 4)
            color = [j * 5 for j in self.points_3d[k]]
            # print(color)
            self.image_3d = cv2.fillConvexPoly(self.image_3d, np.array(temp_points), color)
            # print(f'took {(time.time() - t1) * 1000:.2f} ms', end='\r')
        cv2.imshow("img", self.image_3d)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.image_3d

def mouseCallback(event, x, y, flags, param):
    print(f"event:{event:<20}x:{x:<10}y:{y:<10}flag:{flags:<20}", end='\r')

def testCamera():
    maper = Maper()
    is_maper = False
    cap = cv2.VideoCapture(2)
    cap.set(3, 1280)
    cap.set(4, 1024)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            if not is_maper:
                maper.get_points_map(img)
                is_maper = True
            else:
                maper.draw_points_2d(img)
            # cv2.imshow('img', img)
            # if cv2.waitKey(1) == ord('b'):
            #     break
    else:
        print(cap)
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    maper = Maper()
    maper.get_points_map(np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8))
    print("\ncameraMatrix:\n", maper.K)
    # maper.get_points_map(np.zeros((1024, 1280, 3), dtype=np.uint8))
    maper.demo_display(np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8))
    # maper.demo_display_slow()
    # testCamera()