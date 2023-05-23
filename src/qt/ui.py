import time

try:
    from hj_uiv5 import Ui_leida
except:
    from src.qt.hj_uiv5 import Ui_leida
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsOpacityEffect
from PyQt5.QtGui import QImage, QPixmap, QPaintEvent
from PyQt5 import QtCore
import sys  
import cv2
import numpy as np
from src.myserial.Serial import SerialSender
from arguments import *
class MainUI(QMainWindow, Ui_leida):
    def __init__(self, serialer: SerialSender):
        QMainWindow.__init__(self)
        Ui_leida.__init__(self)

        self.setupUi(self)
        self.serialer = serialer
        # self.open_flag = False
        self.img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.putText(self.img, 
                    "loading", 
                    (SCREEN_W // 2 - 200, SCREEN_H // 2), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                    (255,255,255), 2)

        self._translate = QtCore.QCoreApplication.translate
        self.css_head = '<html><head/><body><p><span style=\"font-family:\'Microsoft YaHei\'; font-size:9pt; font-weight:300;\">'
        self.css_end = '</span></p></body></html>'
        self.map_img = cv2.imread('res/maskv2/map_.png')
        self.map_img = cv2.resize(self.map_img, (800, 400))
        self.map_img = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2RGB)
        # b_channel, g_channel, r_channel = cv2.split(self.map_img)
        self.op_05 = QGraphicsOpacityEffect()
        self.op_05.setOpacity(0.5)
        self.op_01 = QGraphicsOpacityEffect()
        self.op_01.setOpacity(0.8)
        self.op3 = QGraphicsOpacityEffect()
        self.op3.setOpacity(0.7)
        self.loaded = 0
        # alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 230  # alpha通道每个像素点区间为[0,255], 0为完全透明，255是完全不透明

        # self.map_img= cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


    def paintEvent(self, a0: QPaintEvent):
        # try:
        frame = self.img.copy()
        self.frame_width = self.img_label.geometry().width()
        self.frame_height = self.img_label.geometry().height()
        frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img_label.setGraphicsEffect(self.op_01)
        self.Qframe = QImage(
            frame.data, frame.shape[1], frame.shape[0], 
            frame.shape[1]*3, QImage.Format_RGB888)
        
        self.img_label.setPixmap(QPixmap.fromImage(self.Qframe))
        
        map_img = self.map_img.copy()
        map_width = self.map.geometry().width()
        map_height = self.map.geometry().height()
        map_img = cv2.resize(map_img, (map_width, map_height))
        # 设置透明度的值，0.0到1.0，最小值0是透明，1是不透明
    
        self.map.setGraphicsEffect(self.op_05)

        pos_width = self.pos_msg.geometry().width()
        pos_height = self.pos_msg.geometry().height()
        pos_img = np.ones((pos_height, pos_width), dtype=np.uint8) * 255
        pos_img = cv2.cvtColor(pos_img, cv2.COLOR_GRAY2BGR)
        x ,y = 5, 20
        for id, min_point_3d in self.serialer.xcnp_map:
            pos_x, pos_y, pos_z = min_point_3d
            pos_msg = f'{id}: {pos_x:.1f}, {pos_y:.1f}'
            cv2.putText(pos_img, pos_msg, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                (0,0,0), 1)
            p_y = int((RM_FIELD_WEIGHT - pos_y) / RM_FIELD_WEIGHT *  map_height)
            p_x = int(pos_x / RM_FIELD_LENGTH *  map_width)
            cv2.circle(map_img, (p_x, p_y), 3, (0, 255, 0), -1) 
            y += 20
        self.QPos = QImage(
            pos_img.data, pos_img.shape[1], pos_img.shape[0], 
            pos_img.shape[1]*3, QImage.Format_RGB888)
        self.pos_msg.setPixmap(QPixmap.fromImage(self.QPos))
        
        self.QMap = QImage(
                    map_img.data, map_img.shape[1], map_img.shape[0], 
                    map_img.shape[1]*3, QImage.Format_RGB888)
        self.map.setPixmap(QPixmap.fromImage(self.QMap))

        if len(self.serialer.send_data):
            send = '  ' + self.serialer.send_data.hex()
            self.serial_send.setText(self._translate(
                "leida", 
                self.css_head + send + self.css_end))

        time.sleep(0.02)
        self.update()
        # except:
        #     time.sleep(1)
        #     print("failed load")
            # exit(-1)

    def getCameraX(self):
        return self.x_spinBox.value() / 100

    def getCameraY(self):
        return self.y_spinBox.value() / 100

    def getCameraZ(self):
        return self.z_spinBox.value() / 100

    def getPitch(self):
        return self.pitch_spinBox.value() / 10

    def getYaw(self):
        return self.yaw_spinBox.value() / 10
    
    def getRoll(self):
        return self.roll_spinBox.value() / 10

    def getPointsDis(self):
        try:
            return self.points_dis_spinBox.value()
        except:
            return self.points_dis.value()

def get_ele(leida_ui:Ui_leida):
    while 1:
        try:
            print(leida_ui.spinBox.value(), end='\r')
        except:
            pass
        time.sleep(0.02)

if __name__ == "__main__":
    leida_app = QApplication(sys.argv)
    mainWindow = MainUI()

    # leida_ui = MainUI()
    # leida_ui.setupUi(mainWindow)

    mainWindow.show()

    # listen_thread = threading.Thread(target=get_ele, daemon=True, args=(leida_ui, ))
    # listen_thread.start()

    sys.exit(leida_app.exec_())


