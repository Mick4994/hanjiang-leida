try:
    from box import Box
except:
    from src.math.box import Box

F = 6e-3
B = 0.4
L = 14.11e-3
W = 3840
CAR_Y = 0.25
X0, Y0, Z0 = 5.1, 0.8, 4.2
X1 = 5.4
error = 0.4

def writeXY2txt():
    all_boxes = Box(W)
    all_boxes.load_txt()
    with open('test.txt', 'w') as f:
        for left_box, right_box in zip(all_boxes.left_boxes, all_boxes.right_boxes):
            pixel_w = L / W
            deviation_center_x = 1920 - right_box[0]
            parallax_pixel = abs(right_box[0] - left_box[0])
            # print('parallax_pixel:',parallax_pixel)
            parallax = parallax_pixel * pixel_w
            base_line = B - parallax
            tanA = parallax / F
            y_distance = base_line / tanA
            x_distance_frame = L * (deviation_center_x / W)
            x_distance = y_distance * x_distance_frame / F
            f.write(f'X1:{X0 - x_distance:.2f} Y1:{y_distance - Y0:.2f}, c={float(right_box[4]):.2f}\n')

def get_position():
    all_boxes = Box(W)
    all_boxes.load_txt()
    position_list = []
    for left_box, right_box in zip(all_boxes.left_boxes, all_boxes.right_boxes):
        pixel_w = L / W
        deviation_center_x = right_box[0] - 1920
        parallax_pixel = abs(right_box[0] - left_box[0])
        parallax = parallax_pixel * pixel_w
        base_line = B - parallax
        tanA = parallax / F
        y_distance = base_line / tanA
        x_distance_frame = L * (deviation_center_x / W)
        x_distance = y_distance * x_distance_frame / F
        position_list.append([x_distance + X0 + error, y_distance - Y0, f'{float(right_box[-1][:-1]):.2f}'])
    return position_list

