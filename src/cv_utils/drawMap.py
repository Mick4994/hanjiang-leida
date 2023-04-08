import cv2
import numpy as np

global rm_map

points = []
temp_points = []

class Data:
    def __init__(self) -> None:
        self.mask_img = np.array([0])

def mouseCallback(event, x, y, flags, param:Data):
    # print(f"event:{event:<20}x:{x:<10}y:{y:<10}flag:{flags:<20}", end='\r')
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x:{x:<5}y:{y:<5} points:{len(points)}")
        points.append([x, y])
        temp_points.append([x, y])
        # print("param:", param)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points):
            print(f'pop point, points:{len(points)}')
            points.pop()
            temp_points.pop()
    elif event == cv2.EVENT_MBUTTONDOWN:
        cv2.imwrite("res/map_mask.png", mask_img_copy)
        data.mask_img = mask_img_copy.copy()
        temp_points.clear()


if __name__ == "__main__":
    rm_map = cv2.imread("res/rm2023_map.png")
    # cv2.fillPoly(rm_map, )
    data = Data()
    mask_img = np.zeros(rm_map.shape, dtype=rm_map.dtype)
    data.mask_img = mask_img
    cv2.namedWindow("rm_map_copy", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("rm_map_copy", mouseCallback, data)
    while 1:
        rm_map_copy = rm_map.copy()
        mask_img_copy = data.mask_img.copy()
        for i in points:
            cv2.circle(rm_map_copy, i, 2, (0, 255, 0), -1) 
            cv2.circle(mask_img_copy, i, 2, (255, 255, 255), -1) 
            if len(temp_points) > 2:
                cv2.fillConvexPoly(mask_img_copy, np.array(temp_points), (255, 255, 255))
        cv2.imshow("rm_map_copy", rm_map_copy)
        cv2.imshow("mask", mask_img_copy)
        if ord('b') == cv2.waitKey(20):
            cv2.destroyAllWindows()
            break
