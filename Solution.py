import torch
import cv2
import sys
import numpy as np
import time
import threading
from math import sqrt
from arguments import *
from src.qt.ui import MainUI

from detect import YOLO_Detect
from track import YOLO_DEEPSORT

from keras.models import load_model
from torch.autograd import Variable
from src.cv_utils.projectPoints import Maper, CV2RM, red2blue, RM2SER
from src.myserial.Serial import SerialSender

lr = np.array([156, 43, 46])  #红色hsv范围下限
ur = np.array([180, 255, 255])  #红色hsv范围上限

lb = np.array([100, 43, 46])  #蓝色hsv范围下限
ub = np.array([124, 255, 255])   #蓝色hsv范围下限
k = np.ones((10, 10), np.uint8)

class Car:
    def __init__(self, bbox) -> None:
        self.bbox = bbox
        self.serial_id = 0

        self.color = 'unknow'
        self.armor = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.pos = []
        self.live = 5

    def getid_minist(self, model, src, armor):
        """
        输入到 Minist手写数字模型中获取装甲板id

        :param model：加载的装甲板分类pt模型
        :param src：单帧原图
        :param armor：装甲板ROI

        :return None
        """

        # 优化裁剪的ROI
        x1, y1, x2, y2 = armor
        w = x2 - x1
        h = y2 - y1
        y1 -= int(0.1 * h) if y1 - int(0.1 * h) >= 0 else 0
        y2 += int(0.1 * h) if y2 + int(0.1 * h) < len(src) else len(src) - 1 
        x1 -= int(0.1 * w) if x1 - int(0.1 * w) >= 0 else 0
        x2 += int(0.1 * w) if x2 + int(0.1 * w) < len(src[0]) else len(src[0]) - 1
        crop = src[y1: y2, x1 : x2]

        # 二值化数字为了输入神经网络
        crop = np.array(crop)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (280, 280))
        _, crop = cv2.threshold(crop, 155, 255, cv2.THRESH_BINARY)

        # 裁剪数字周围多余白区
        crop = cut_num(crop)
        crop = cv2.resize(crop, (28, 28))

        # 接收输出识别到的数字，放入装甲板索引统计表
        out = model.predict(np.array([crop]))
        max_index = np.argmax(out[0].tolist())
        self.armor[max_index] += 1

        # 转换为裁判系统规定编号
        serial_shifting = 100 if self.color == 'blue' else 0

        # 每次统计次数最多的编号为最为可能的该车编号
        self.serial_id = np.argmax(np.array(self.armor)) + serial_shifting

    def getid_lyk(self, model, src, armor, device):
        """
        输入到李彦宽的装甲板识别模型中获取装甲板id，
        无返回值，结果保存在对象的 serial_id 属性中

        :param model：加载的装甲板分类pt模型
        :param src：单帧原图
        :param armor：装甲板ROI
        :param device：选择CPU或者GPU（'cuda' 或 'cpu'）

        :return None
        """
        x1, y1, x2, y2 = armor

        #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓将裁剪的装甲板ROI二值化↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        crop = src[y1: y2, x1 : x2]
        crop = np.array(crop)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (280, 280))
        _, crop = cv2.threshold(crop, 155, 255, cv2.THRESH_BINARY) #二值化
        crop = cv2.erode(crop, None, iterations=5) #先腐蚀
        crop = cv2.dilate(crop, None, iterations=5) #后膨胀
        #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑将裁剪的装甲板ROI二值化↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        #↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓将斜向的数字通过仿射变换拉直↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        coords = np.column_stack(np.where(crop > 0))
        _, _, angle =cv2.minAreaRect(coords)
        if angle > 45:
            angle = 90 - angle
        else:
            angle = -angle
        center = (280//2, 280//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0) 
        crop = cv2.warpAffine(crop, M, (280,280), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        #↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑将斜向的数字通过仿射变换拉直↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        crop = cut_num(crop)
        # 裁剪装甲板数字周围多余的白色区域，使数字在图中的占比最大

        crop = cv2.resize(crop, (22, 30))
        _, crop = cv2.threshold(crop, 155, 255, cv2.THRESH_BINARY)
        # 注：在resize后需要再次二值化

        # 转换到torch变量中输入模型，out接收输出
        crop = np.reshape(crop.astype(np.float32) / 255.0, (1, 1, crop.shape[0], crop.shape[1]))
        crop = torch.from_numpy(crop)
        input = Variable(crop)
        out = model(input)

        # 放入装甲板索引统计表，对应的编号统计＋1
        max_index = np.argmax(out[0].tolist())
        self.armor[max_index] += 1 

        # 转换为裁判系统规定编号
        serial_shifting = 100 if self.color == 'blue' else 0

        # 每次统计次数最多的编号为最为可能的该车编号
        self.serial_id = np.argmax(np.array(self.armor)) + serial_shifting

    def getColor(self, src, armor):
        """
        通过传统视觉，获取该车装甲板灯条的颜色，为了区分红蓝方
        :param src：单帧原图
        :param armor：装甲板ROI

        :return None
        """
        x1, y1, x2, y2 = armor
        crop = src[y1: y2, x1 : x2]

        # 转换为HSV色彩空间方便滤色
        crop = np.array(crop)
        cv2.resize(crop, (25, 25))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 分别对红蓝滤色
        inRange_Red = cv2.inRange(hsv, lr, ur)  #滤剩红色
        inRange_Red = inRange_Red.reshape((inRange_Red.shape[0] * inRange_Red.shape[1]))
        red_count = inRange_Red.tolist().count(255)
        inRange_Blue = cv2.inRange(hsv, lb, ub)  #滤剩蓝色
        inRange_Blue = inRange_Blue.reshape((inRange_Blue.shape[0] * inRange_Blue.shape[1]))
        blue_count = inRange_Blue.tolist().count(255)

        # 超过 10 个像素才有效否则被返回
        if max([blue_count, red_count]) < 10:
            return
        self.color = "blue" if blue_count > red_count else "red"

def cut_num(crop:np.ndarray):
    """
    在YOLO识别出装甲板的ROI基础上切出装甲板上的数字/图像
    把二值图数字周围多余的白色区域裁剪掉
    
    :param crop：裁剪的ROI图像

    :return crop：数字裁剪到占比最大的ROI图像
    """
    up_bound = 0
    down_bound = 0
    len_crop = len(crop)

    # 翻转图像
    len_crop_T = len(crop.T)
    crop_T = crop.T

    # 截取纵向
    for x in range(len_crop):
        if crop[x].tolist().count(255) > 25 and not up_bound :
            up_bound = x
        if crop[-x].tolist().count(255) > 25 and not down_bound:
            down_bound = len_crop - x
        if up_bound and down_bound:
            break

    left_bound = 0
    right_bound = 0

    # 截取横向
    for x in range(len_crop_T):
        if crop_T[x].tolist().count(255) > 25 and not left_bound:
            left_bound = x
        if crop_T[-x].tolist().count(255) > 25 and not right_bound:
            right_bound = len_crop_T - x
        if left_bound and right_bound:
            break

    # 微调边距
    up_bound -= 3 if up_bound - 3 >= 0 else 0
    down_bound += 3 if down_bound + 3 < len_crop else len_crop - 1 
    left_bound -= 3 if left_bound - 3 >= 0 else 0
    right_bound += 3 if right_bound + 3 < len_crop_T else len_crop_T - 1

    return crop[up_bound:down_bound, left_bound:right_bound]

def IsEnemy(color):
    enemy = "blue" if is_bule else "red"
    return color == enemy

def IsINbbox(car_bbox, armor_bbox):
    return car_bbox[0] <= armor_bbox[0] <= car_bbox[0] + car_bbox[2] \
        and car_bbox[1] <= armor_bbox[1] <= car_bbox[1] + car_bbox[3]   

def drawFPS(image, spread):
    fps = f'FPS: {1 / spread:.1f}'
    cv2.putText(image, fps, (10, 50), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                    (0, 255,255), 2)

class Camera:
    def __init__(self, 
            camera_x = -4.8, 
            camera_y = 3.8, 
            camera_z = 1.2, 
            r_x = 20,
            yaw = 0,
            roll = 0,
            points_dis = 24) -> None:
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.camera_z = camera_z
        self.r_x = r_x
        self.yaw = yaw
        self.roll = roll
        self.points_dis = points_dis
    def __eq__(self, other) -> bool:
        return (self.camera_x == other.camera_x 
            and self.camera_y == other.camera_y
            and self.camera_z == other.camera_z
            and self.r_x == other.r_x
            and self.yaw == other.yaw
            and self.points_dis == other.points_dis)


class Solutionv2:
    """
    第二版本的程序主干（当前）
    """
    def __init__(self) -> None:
        self.exit_flag = -1

        self.armor_lyk_model = torch.load('weights/2023_4_9_hj_num_3.pt').cpu()
        self.armor_lyk_model.eval()
        # self.armor_minist_model = load_model("weights/Minist_1_5.h5")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.car_dict = {}

        self.car_bboxs = []
        self.setlive = 5

    def yolo_deepsort_layer(self, yolo_deepsort : YOLO_DEEPSORT):
        """
        YOLO 和 DEEPSORT 检测追踪层
        """
        self.armor_bboxs = yolo_deepsort.armor_bboxs
        self.src_img = yolo_deepsort.src_img

        pop_list = []
        # self.car_bboxs = []
        self.car_dict = {}
        for id, bbox in yolo_deepsort.car_bboxs:
            if self.car_dict.get(id) is None:
                self.car_dict[id] : Car = Car(bbox)
            else:
                self.car_dict[id].bbox = bbox
                self.car_dict[id].live = self.setlive
            self.car_dict[id].live -= 1
            if self.car_dict[id].live <= 0:
                self.car_dict[id].append(id)
            # self.car_bboxs.append(bbox)
        for id in pop_list:
            self.car_dict.pop(id)
        
    def armor_color_layer(self):
        """
        装甲板识别层：识别颜色，区分敌我，获取id层
        """
        car_id_dict = {}
        for id, car in self.car_dict.items():
            for armor_bbox in self.armor_bboxs:
                if IsINbbox(car.bbox, armor_bbox):
                    car.getColor(self.src_img, armor_bbox)
                    if IsEnemy(car.color):
                        car.getid_lyk(
                            self.armor_lyk_model, 
                            self.src_img, 
                            armor_bbox,
                            self.device
                            )
                        # car.getid_minist(
                        #     self.armor_minist_model, 
                        #     self.src_img, 
                        #     armor_bbox,
                        #     )
            car_id_dict[id] = car.serial_id
        # print(car_id_dict)
                        
    def project_layer(self, maper_points):
        """
        （已弃用）旧图形学解算层（像素坐标系映射世界坐标系），体积碰撞法

        :param maper_points：像素映射位置表

        :return serial_out：输出到串口上传发送的数据（车辆编号和位置）
        """
        serial_out = []
        for id, car in self.car_dict.items():
            if IsEnemy(car.color):
                car_bbox = car.bbox
                car_center_x = car_bbox[0] + int((car_bbox[2] - car_bbox[0]) / 2)
                car_center_y = car_bbox[3] # y2
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

                min_point_3d = RM2SER(min_point_3d)
                car.pos = min_point_3d
                serial_out.append([car.serial_id, car.pos])
        return serial_out

    def new_project_layer(self, maper : Maper):
        """
        新图形学解算层，哈希法

        :param maper：像素坐标系映射世界坐标系的映射关系实例

        :return serial_out：输出到串口上传发送的数据（车辆编号和位置）
        """
        image_3d = maper.image_3d
        serial_out = []
        for id, car in self.car_dict.items():
            if IsEnemy(car.color):
                car_bbox = car.bbox
                car_center_x = car_bbox[0] + int((car_bbox[2] - car_bbox[0]) / 2)
                car_center_y = car_bbox[3] # y2
                pos = image_3d[car_center_y][car_center_x] / 5
                
                pos = CV2RM(pos)
                if is_bule:
                    pos = red2blue(pos)

                pos[0] += 0.3

                pos = RM2SER(pos)
                car.pos = pos
                serial_out.append([car.serial_id, car.pos])
        return serial_out



    def backbone(self,
                 mainWindow: MainUI, 
                 yolo_deepsort: YOLO_DEEPSORT, 
                 serial_sender: SerialSender):
        maper = Maper(  mainWindow.getCameraX(),  
                        mainWindow.getCameraY(),
                        mainWindow.getCameraZ(),
                        mainWindow.getPitch(),
                        mainWindow.getYaw(),
                        mainWindow.getRoll(),
                        mainWindow.getPointsDis()
                        )
        """
        程序骨干主函数，大循环在此

        :param mainWindow：雷达的调参界面实例，传入是为了读取调参信息，这里是后端
        :param yolo_deepsort：对单帧进行车辆目标检测追踪的线程实例
        :param serial_sender：串口通讯的线程实例
        :param maper：像素坐标系映射世界坐标系的映射关系通过参数信息计算的映射类

        :return None
        """
        is_first_run = True
        while 1:
            t1 = time.time()
            image = np.array(yolo_deepsort.src_img)
            camera = Camera(mainWindow.getCameraX(),  
                            mainWindow.getCameraY(),
                            mainWindow.getCameraZ(),
                            mainWindow.getPitch(),
                            mainWindow.getYaw(),
                            mainWindow.getRoll(),
                            mainWindow.getPointsDis()
                            )
            
            if len(yolo_deepsort.out_img):
                # mainWindow.loaded += 1
                if not is_first_run:
                    if camera != last_camera:
                        maper.update(camera_x = camera.camera_x,
                                     camera_y = camera.camera_y,
                                     camera_z = camera.camera_z,
                                     r_x = camera.r_x,
                                     yaw = camera.yaw, 
                                     roll = camera.roll,
                                     points_dis = camera.points_dis
                                     )
                        maper_points = maper.get_points_map(image)
                else:
                    # print('first loading maper points!')
                    is_first_run = False
                    maper_points = maper.get_points_map(image)
                    # print('finished load!')
                vision_img = yolo_deepsort.out_img
                # vision_img = maper.draw_points_2d(vision_img)
                vision_img = maper.draw_points_noshow(yolo_deepsort.out_img)
                mainWindow.img = vision_img

                # 几乎无性能影响
                self.yolo_deepsort_layer(yolo_deepsort)
                
                # 影响最大
                self.armor_color_layer()
                
                # 影响很大
                # serial_sender.Send(self.project_layer(maper_points))

                
                # 影响很小
                
                serial_out = self.new_project_layer(maper)
                
                # print(f'took {spread * 1000:.3f}ms', end='\r')

                serial_thread = threading.Thread(target=serial_sender.Send, args=(serial_out,), daemon=True)
                serial_thread.start()

                spread = time.time() - t1
                if spread > 0.003:
                    
                    drawFPS(vision_img, spread)
                    mainWindow.img = vision_img
                else:
                    # print('good fps!', end='\r')
                    time.sleep(0.02)
                    
            else:
                if len(yolo_deepsort.src_img):
                    if is_first_run:
                        is_first_run = False
                        # print('first loading maper points!')
                        maper_points = maper.get_points_map(yolo_deepsort.src_img)
                        # print('finished load!')
                    maper.update(camera_x = camera.camera_x,
                                camera_y = camera.camera_y,
                                camera_z = camera.camera_z,
                                r_x = camera.r_x,
                                yaw = camera.yaw, 
                                roll = camera.roll,
                                points_dis = camera.points_dis
                                )
                    vision_img = maper.draw_points_noshow(yolo_deepsort.src_img)
                    mainWindow.img = vision_img
                    # mainWindow.img = yolo_deepsort.src_img
                time.sleep(0.02)

            last_camera : Camera = camera
            if self.exit_flag == 0:
                break
            # time.sleep(0.02)
        print("\nexit!")
        # cv2.destroyAllWindows()
        exit(0)
    
    def Exit(self, 
             yolo_detect: YOLO_DEEPSORT):
        """
        退出方法，本来没必要，但是yolo总是会卡死
        """
        for i in range(10):
            yolo_detect.exit_flag = self.exit_flag
            time.sleep(0.1)
        print("exit_args:", self.exit_flag)
        sys.exit(self.exit_flag)



class Solutionv1:
    """
    第一版本的程序主干（联调期间，已废弃）
    """
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
        """
        退出方法，本来没必要，但是yolo总是会卡死
        """
        for i in range(10):
            yolo_detect.exit_flag = self.exit_flag
            time.sleep(0.1)
        print("exit_args:", self.exit_flag)
        sys.exit(self.exit_flag)