import threading
import sys

from PyQt5.QtWidgets import QApplication
from time import sleep
from arguments import *

from Solution import Solutionv1, Solutionv2
from detect import YOLO_Detect
from track import YOLO_DEEPSORT
from src.myserial.Serial import SerialSender
from src.qt.ui import MainUI

def main_v1():    
    return Solutionv1(), YOLO_Detect(SOURCE)

def main_v2():
    return Solutionv2(), YOLO_DEEPSORT(SOURCE)

if __name__ == "__main__":

    # solution, yolo_thread = main_v1()
    solution, yolo = main_v2()
    yolo_thread = threading.Thread(target=yolo.detect,
                                   args=(solution, ),
                                   daemon=True
                                  )
    yolo_thread.start()
    serial_sender = SerialSender(com=COM)
    leida_app = QApplication(sys.argv)
    mainWindow = MainUI(serialer=serial_sender)
    mainWindow.show()
    backbone_thread = threading.Thread(target=solution.backbone, 
                                       args=(mainWindow, yolo, serial_sender), 
                                       daemon=True
                                       )
    backbone_thread.start()
    solution.exit_flag = leida_app.exec_()
    solution.Exit(yolo)