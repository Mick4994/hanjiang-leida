
# 比赛参数
is_bule = False # 是否为蓝方
red_robot_id_serial = [1, 2, 3, 4, 5] # 串口协议红/蓝方机器人id
blue_robot_id_serial = [101, 102, 103, 104, 105]
cls2serial = [0, 0, 0] + red_robot_id_serial + [0, 0] + blue_robot_id_serial + [0, 0]

# 场地参数
RM_FIELD_LENGTH = 28.052
RM_FIELD_WEIGHT = 15.035
Load_NPY = 'points_3d.npy'

# 相机外参 #云台相机位姿
Camera_X = -4.8 
Camera_Y = 3.8
Camera_Z = 1.2
Rotato_X = 20
# Camera_X = -1.5
# Camera_Y = 1.4
# Camera_Z = -0.5
# Rotato_X = 20
Rotato_Yaw = 0
Rotato_roll = 0

# 相机内参
Focal_Length = 4.2e-3
Sensor_X = 5.37e-3
Sensor_Y = 4.04e-3
Unit_Size = 2.8e-6
# SCREEN_W = 1920
# SCREEN_H = 1080


# 投影点间距（调试界面的实时画面处）
points_distance = 2

# 硬件参数：
COM = "COM4" # 串口号
SOURCE = "0" #视频/图片源 "1600x1200.png" "'yolov5/0000-0170.mp4'"

# yolo参数
Yolo_Weight = '20230505_2cls.pt'
Input_Size = 1280
nc = 2
names = ['car', 'armor']

# nc = 17 # 类的个数 
# names = ['car','watcher','base',
#          'armor-red-1','armor-red-2','armor-red-3','armor-red-4','armor-red-5',
#          'armor-red-6','armor-red-8',
#          'armor-blue-1','armor-blue-2','armor-blue-3','armor-blue-4','armor-blue-5',
#          'armor-blue-6','armor-blue-8']