import numpy as np
import cv2
from math import radians

def CV2RM(points_3d):
    x, y, z = points_3d
    return [z, x, y]

class Maper:
    def __init__(self, camera_x = -4.8, camera_y = 3.8, camera_z = 1.2, r_x = 20) -> None:
        # 相机外参
        self.camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
        self.camera_dir = np.array([radians(r_x), radians(0), radians(0)], dtype=np.float32)

        # 焦距与传感器尺寸
        self.focal_length = 6
        self.sensor_size_x = 7.176 # 1/1.8 inch
        self.sensor_size_y = 5.32

        self.mapper = []

    def get_points_map(self, image = np.zeros((1200, 1600, 3), dtype=np.uint8)):
        # 场地点云

        points_3d = []
        distance = 0.2
        x_len = int(13.879 // distance) + 1
        z_len = int(25.879 // distance) + 1

        x = 0
        z = 0
        while x < 13.879:
            while z < 25.879:
                points_3d.append([x, 0, z])
                z += distance
            x += distance
            z = 0

        self.reshape_points_3d = np.reshape(points_3d, (x_len, z_len, 3))

        points_3d = np.array(points_3d, dtype=np.float32)

        self.points_3d = points_3d.copy()

        # 图像尺寸
        image_width = image.shape[1]
        image_height = image.shape[0]

        fx = self.focal_length * image_width / self.sensor_size_x
        fy = self.focal_length * image_height / self.sensor_size_y
        cx = image_width / 2
        cy = image_height / 2
        # 
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        print(K)
        # 计算被旋转后的平移向量
        R, _ = cv2.Rodrigues(self.camera_dir) 
        # R 叉乘 camera_pos
        tvec = R @ self.camera_pos      
        # rvec, _ = cv2.Rodrigues(R) 

        points_2d, _ = cv2.projectPoints(points_3d, self.camera_dir, tvec, K, None)
        self.points_2d = np.array(points_2d, dtype=np.int32)

        self.reshape_points_2d = np.reshape(self.points_2d, (x_len * z_len, 2))

        maper_points = list(zip(self.points_3d, self.reshape_points_2d))
        return maper_points



    def draw_points_2d(self, image = np.zeros((1200, 1600, 3))):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        for p in self.points_2d[::24]:
            cv2.circle(image, p[0], 5, (0, 255, 0), -1) 

        cv2.imshow("image", image)
        if ord('b') == cv2.waitKey(1):
            cv2.destroyAllWindows()
            exit(0)

    def demo_display(self, image = np.zeros((1200, 1600, 3))):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        for p in self.points_2d:
            cv2.circle(image, p[0], 5, (0, 255, 0), -1) 

        cv2.setMouseCallback("image", mouseCallback, 1)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def demo_display_slow(self, image = np.zeros((1200, 1600, 3))):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        for p in self.reshape_points_2d:
            cv2.circle(image, p, 5, (0, 255, 0), -1) 
            
            cv2.imshow("image", image)
            if ord('b') == cv2.waitKey(1):
                cv2.destroyAllWindows()
                exit(0)

def mouseCallback(event, x, y, flags, param):
    print(f"event:{event:<20}x:{x:<10}y:{y:<10}flag:{flags:<20}", end='\r')

if __name__ == "__main__":
    maper = Maper()
    maper.get_points_map()
    maper.demo_display()
    # maper.demo_display_slow()