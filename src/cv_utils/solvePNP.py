import cv2
import numpy as np

F = 6e-3
CMOS_W = 7.18e-3
CMOS_H = 5.32e-3
W_PIXEL = 1600
H_PIXEL = 1200
dx = CMOS_W / W_PIXEL
dy = CMOS_H / H_PIXEL
fx = F / dx
fy = F / dy
u0 = W_PIXEL / 2
v0 = H_PIXEL / 2


class Abstract_Camera:
    def __init__(self) -> None:
        self.focal_lenght = 6e-3 # (m)
        self.cmos_weight = 7.18e-3 # (m)
        self.cmos_height = 5.32e-3 # (m)
        self.frame_weight = 1600 # (pixel)
        self.frame_height = 1200 # (pixel)

        self.dx = self.cmos_weight / self.frame_weight
        self.dy = self.cmos_height / self.frame_height
        self.u0 = self.frame_weight / 2
        self.v0 = self.frame_height / 2
        self.fx = self.focal_lenght / self.dx
        self.fy = self.focal_lenght / self.dy
        
        self.cameraMatrix = np.array([[self.fx, 0, self.u0],
                                      [0, self.fy, self.v0],
                                      [0,       0,       1]],
                                    dtype=np.float64)
        
        print("cameraMatrix:\n", self.cameraMatrix)

        self.dist_coeffs = np.zeros((4, 1))

        self.frame = cv2.imread("res/43.png")

    def get_R4_2D(self, img:np.ndarray, show = False):
        r0_clip = [550, 650, 850, 960]
        r3_clip = [800, 900, 150, 350]
        return_points = []
        # clip = r3_clip
        for clip in [r0_clip, r3_clip]:
            roi = img[clip[0]:clip[1], clip[2]:clip[3]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if clip[0] == r0_clip[0]:
                _, roi = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)
            elif clip[0] == r3_clip[0]:
                _, roi = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
            if show:
                cv2.imshow("roi", roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            countours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 查找轮廓
            min_x = W_PIXEL
            max_x = 0
            min_y = H_PIXEL
            max_y = 0
            for cnt in countours: #遍历轮廓
                x,y,w,h = cv2.boundingRect(cnt) #取左上角x，y和边框宽高
                x += clip[2]
                y += clip[0]
                if((w > 10 and h > 10) if clip[0] == r0_clip[0] else (w < 20 and h < 10)):
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    if x + w > max_x:
                        max_x = x + w
                    if y + h > max_y:
                        max_y = y + h
            if show:
                cv2.rectangle(img,(min_x, min_y),(max_x, max_y),(0,255,0),2)
                cv2.imshow("img", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        # return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]
            if clip[0] == r0_clip[0]:
                return_points.append([min_x, min_x]) # 左上角
            return_points.append([max_x, min_y]) # 右上角
            return_points.append([min_x, max_y]) # 左下角
            if clip[0] == r0_clip[0]:
                return_points.append([max_x, max_y]) # 右下角
        return return_points

    def getPosition(self, show = False):
        # r0左上角 # r0右上角 # r0左下角 # r0右下角 # r3左下角 # r3右上角
        R4_symbol_3D = [[5.3037, 8.1432, 0.56769], 
                        [5.913, 8.1432, 0.56769],
                        [5.3037, 8.1432,0.11077],
                        [5.913, 8.1432, 0.11077],
                        [1.797, 4.4705, 0.37015],
                        [2.0739, 4.8397, 0.37015]]

        img = self.frame.copy()
        R4_symbol_2D = self.get_R4_2D(img, show=show)
        
        objPoints = np.array(R4_symbol_3D, dtype=np.float64)
        imgPoints = np.array(R4_symbol_2D, dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(objPoints, 
                                           imgPoints, 
                                           self.cameraMatrix, 
                                           self.dist_coeffs, 
                                           flags=cv2.SOLVEPNP_ITERATIVE)

        print("tvec:\n", tvec)
        if success:
            rotM = cv2.Rodrigues(rvec)[0]
            position = -np.matrix(rotM).T * np.matrix(tvec)
            print("position:\n", position)

    def getUV(self):
        rvet = np.array([0, 0, 0])
        tvet = np.array([0, 0, 0])
        # cv2.projectPoints()



if __name__ == "__main__":
    camera = Abstract_Camera()
    camera.getPosition()

    # img = cv2.imread("res/43.png")
    # roiR3 = img[800:900, 150:350]
    # roiR3 = cv2.cvtColor(roiR3, cv2.COLOR_BGR2GRAY)
    # _, roiR3 = cv2.threshold(roiR3, 180, 255, cv2.THRESH_BINARY)
    # cv2.imshow("roiR3", roiR3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # countours, _ = cv2.findContours(roiR3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 查找轮廓
    # min_x = 1600
    # max_x = 0
    # min_y = 1200
    # max_y = 0
    # for cnt in countours: #遍历轮廓
    #     x,y,w,h = cv2.boundingRect(cnt) #取左上角x，y和边框宽高
    #     x += 150
    #     y += 800
    #     if w < 20 and h < 10:
    #         if x < min_x:
    #             min_x = x
    #         if y < min_y:
    #             min_y = y
    #         if x + w > max_x:
    #             max_x = x + w
    #         if y + h > max_y:
    #             max_y = y + h
    # cv2.rectangle(img,(min_x, min_y),(max_x, max_y),(0,255,0),2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()