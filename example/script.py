import pyautogui as pag
import time
import win32gui


if __name__ == "__main__":
    time.sleep(1)
    # pag.click()
    # hwnd = win32gui.GetForegroundWindow()
    # print(win32gui.GetWindowText(hwnd))
    # s = 0
    # funcs = [pag.click, pag.keyDown, pag.keyUp, pag.typewrite, pag.keyDown, pag.keyUp, pag.keyDown, pag.keyUp]
    # args = [[1479, 133, 1, pag.LEFT],["ENTER"], ["ENTER"], [f"{s}"], ["ENTER"], ["ENTER"], ["ENTER"], ["ENTER"]]
    # pag.click()
    # while 1:
    #     for i in range(len(funcs)):
    #         # hwnd = win32gui.GetForegroundWindow()
    #         # title = win32gui.GetWindowText(hwnd)
    #         # if title != "RSView" and title != "Select Print Frame":
    #         #     break
    #         print(f"{s}")
    #         funcs[i](*args[i])
    #         time.sleep(0.1)
    #     else:
    #         continue

    #     break
        # print(title)
    i = 2000 
    while 1:
        # print(pag.position() ,end='\r')
        pag.click(x=1479, y=133, clicks=1, button=pag.LEFT)
        time.sleep(0.1)

        pag.keyDown("ENTER")
        time.sleep(0.1)
        pag.keyUp("ENTER")

        pag.typewrite(f"{i}")
        time.sleep(0.1)
        pag.keyUp("1")

        pag.keyDown("ENTER")
        time.sleep(0.1)
        pag.keyUp("ENTER")

        pag.keyDown("ENTER")
        time.sleep(0.1)
        pag.keyUp("ENTER")

        i += 1
        time.sleep(0.1)