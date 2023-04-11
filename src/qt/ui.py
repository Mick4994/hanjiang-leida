import threading
import time
try:
    from hj_uiv3 import Ui_leida
except:
    from src.qt.hj_uiv3 import Ui_leida
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QPaintEvent
import sys  
import cv2
import numpy as np
class MainUI(QMainWindow, Ui_leida):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_leida.__init__(self)
        self.setupUi(self)
        self.open_flag = False
        self.img = np.zeros((1024, 1280, 3), dtype=np.uint8)
        cv2.putText(self.img, "loading", (450, 500), cv2.FONT_HERSHEY_COMPLEX_SMALL,4,(255,255,255),2)

    def paintEvent(self, a0: QPaintEvent):
        # try:
        frame = self.img.copy()
        frame = cv2.resize(frame, (400, 280), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.Qframe = QImage(
            frame.data, frame.shape[1], frame.shape[0], frame.shape[1]*3, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(self.Qframe))
        time.sleep(0.02)
        self.update()
        # except:
        #     time.sleep(1)
        #     print("failed load")
            # exit(-1)

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


