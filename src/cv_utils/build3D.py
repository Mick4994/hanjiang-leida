import numpy as np
import cv2
try:
    from arguments import *
except:
    import sys
    import os
    sys.path.append(os.getcwd())
    from arguments import *

if __name__ == "__main__":
    trapezoid_highland_mask = 'res/mask/map_mask_37.png'
    circular_highland_mask = 'res/mask/map_mask_55.png'
    jump_highland_mask = 'res/mask/map_mask_18.png'
    hights = [-37, -55, -18]
    masks = []
    for hight in hights:
        masks.append(
            cv2.imread('res/mask/map_mask_' + str(abs(hight)) + '.png', 0)
            )
    RM_field_lenght = RM_FIELD_LENGTH
    RM_field_weight = RM_FIELD_WEIGHT
    # RM_field_lenght = 7
    # RM_field_weight = 6
    points_3d = []
    distance = 0.2
    x_len = int(RM_field_weight // distance) + 1
    z_len = int(RM_field_lenght // distance) + 1

    x = 0
    z = 0
    while x < RM_field_weight:
        while z < RM_field_lenght:
            points_3d.append([x, 0, z])
            z += distance
        x += distance
        z = 0

    for hight, mask_img in zip(hights, masks):
        for point in points_3d:
            img_height, img_weight = mask_img.shape
            # print(mask_img.shape)
            pixel_x, pixel_y = int(point[2] / RM_field_weight * img_height), int(point[0] / RM_field_lenght * img_weight)
            try:
                if mask_img[pixel_y][pixel_x]:
                    point[1] = hight/100
            except:
                pass

    np.save('points_3d.npy', np.array(points_3d, dtype=np.float32))