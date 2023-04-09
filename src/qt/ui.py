import threading
import time
from hj_ui import Ui_leida
from PyQt5.QtWidgets import QApplication, QMainWindow

import sys  

def get_ele(leida_ui:Ui_leida):
    while 1:
        try:
            print(leida_ui.spinBox.value(), end='\r')
        except:
            pass
        time.sleep(0.02)

if __name__ == "__main__":
    leida_app = QApplication(sys.argv)
    mainWindow = QMainWindow()

    leida_ui = Ui_leida()
    leida_ui.setupUi(mainWindow)

    mainWindow.show()

    listen_thread = threading.Thread(target=get_ele, daemon=True, args=(leida_ui, ))
    listen_thread.start()

    sys.exit(leida_app.exec_())


