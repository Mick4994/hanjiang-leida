import cv2
from src.math.parallax import get_position

L = 26.714
W = 14.7
X0, Y0, Z0 = 5.1, 0.8, 4.2

def pos2map(map_copy, positions, delay = 0):
    for position in positions:
        x, y, name = position
        x_pixel = int(abs(float(y)/L * x_b))
        y_pixel = int(abs(float(x)/W * y_b))
        cv2.circle(map_copy, (x_pixel, y_pixel),5,(0,255,0),cv2.FILLED)
        cv2.putText(map_copy, name, (x_pixel - 10, y_pixel + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
    cv2.namedWindow('map', cv2.WINDOW_NORMAL)
    cv2.imshow("map",map_copy)
    if cv2.waitKey(delay) == ord('b'):
        cv2.destroyAllWindows()
        return False
    if delay:
        cv2.destroyAllWindows()
    return True

def singe():
    with open('C:/hanjiang/test.txt') as f:
        lines = f.readlines()
    positions = []
    for line in lines:
        name, x, y, z, _ = line.split(' ')
        positions.append([x, y, name])
    pos2map(rm_map_src.copy(), positions)

def multi():
    for i in range(1, 161):
        rm_map = rm_map_src.copy()
        with open(f'C:/hanjiang/test/frame_{i}.txt') as f:
            lines = f.readlines()
        positions = []
        for line in lines[:10]:
            name, x, y, z, _ = line.split(' ')
            positions.append([x, y, name])
            if not pos2map(rm_map, positions, 100):
                break

def fromParallax():
    pos2map(rm_map_src.copy(), get_position())

def mergePara_true():
    rm_map = rm_map_src.copy()
    with open('C:/hanjiang/test.txt') as f:
        lines = f.readlines()
    for line in lines:
        name, x, y, z, _ = line.split(' ')
        x_pixel = int(abs(float(y)/L * x_b))
        y_pixel = int(abs(float(x)/W * y_b))
        cv2.circle(rm_map, (x_pixel, y_pixel),5,(255,255,0),cv2.FILLED)
        cv2.putText(rm_map, name, (x_pixel - 10, y_pixel + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),2)
    pos2map(rm_map, get_position())

def y2x():
    rm_map = rm_map_src.copy()
    with open('C:/hanjiang/test.txt') as f:
        lines = f.readlines()
    for line in lines:
        name, x, y, z, _ = line.split(' ')
        x_pixel = int(abs(float(y)/L * x_b))
        y_pixel = int(abs(float(x)/W * y_b))


if __name__ == "__main__":
    rm_map_src = cv2.imread("res/rm2023_map.png")
    x_b = rm_map_src.shape[1] * 4
    y_b = rm_map_src.shape[0] * 4
    rm_map_src = cv2.resize(rm_map_src, (x_b, y_b))
    # singe()
    # multi()
    # fromParallax()
    mergePara_true()

