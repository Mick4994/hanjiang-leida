import threading
import time

# from hj_uiv3 import Ui_leida
try:
    from hj_uiv4 import Ui_leida
except:
    from src.qt.hj_uiv4 import Ui_leida
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QPaintEvent
from PyQt5 import QtCore
import sys  
import cv2
import numpy as np
from src.myserial.Serial import SerialSender
class MainUI(QMainWindow, Ui_leida):
    def __init__(self, serialer: SerialSender):
        QMainWindow.__init__(self)
        Ui_leida.__init__(self)

        self.setupUi(self)
        self.serialer = serialer
        # self.open_flag = False
        self.img = np.zeros((1024, 1280, 3), dtype=np.uint8)
        cv2.putText(self.img, 
                    "loading", 
                    (450, 500), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                    (255,255,255), 2)

        self._translate = QtCore.QCoreApplication.translate
        self.css_head = '<html><head/><body><p><span style=\"font-family:\'Microsoft YaHei\'; font-size:9pt; font-weight:300;\">'
        self.css_end = '</span></p></body></html>'

    def paintEvent(self, a0: QPaintEvent):
        # try:
        frame = self.img.copy()
        frame = cv2.resize(frame, (400, 280), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.Qframe = QImage(
            frame.data, frame.shape[1], frame.shape[0], 
            frame.shape[1]*3, QImage.Format_RGB888)
        
        self.img_label.setPixmap(QPixmap.fromImage(self.Qframe))

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
        return self.pitch_spinBox.value()

    def getYaw(self):
        return self.yaw_spinBox.value()
    
    def getRoll(self):
        return self.roll_spinBox.value()

    def getPointsDis(self):
        return self.points_dis_spinBox.value()

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


