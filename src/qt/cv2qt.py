import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import cv2qtui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import time

class MainCode(QMainWindow, cv2qtui.Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        cv2qtui.Ui_MainWindow.__init__(self)
        self.setupUi(self)
        # self.pushButton.clicked.connect(self.on_video)
        # self.open_flag = False
        self.video_stream = cv2.VideoCapture(2)
        # for i in range(20):
        #     print(f"cap:{i}", self.video_stream.get(i))
        # print(self.video_stream.get(cv2.CAP))
        self.video_stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # self.painter = QPainter(self)

    # def on_video(self):
    #     if self.open_flag:
    #         self.pushButton.setText('open')
    #     else:
    #         self.pushButton.setText('close')
        # self.open_flag = bool(1-self.open_flag)

    def paintEvent(self, a0: QtGui.QPaintEvent):
        # ret, frame = self.video_stream.read()
        # if self.open_flag:

            ret, frame = self.video_stream.read()
            try:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # cv2.imshow('test',frame)
                # cv2.waitKey(10)
                self.Qframe = QImage(
                    frame.data, frame.shape[1], frame.shape[0], frame.shape[1]*3, QImage.Format_RGB888)
                # print(Qframe)
                #pix = QPixmap(Qframe).scaled(frame.shape[1], frame.shape[0])
                # self.setPixmap(pix)
                # QRect qq(20,50,self.img.width,self.img.height)
                self.img_label.setPixmap(QPixmap.fromImage(self.Qframe))
                # self.painter.drawImage(QPoint(20,50),Qframe)
                # print(Qframe)
                time.sleep(0.02)
                self.update()
            except:
                try:
                    self.video_stream.release()
                except:
                    pass
        # time.sleep(0.02)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())
