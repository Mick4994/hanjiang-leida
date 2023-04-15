
# 比赛参数
is_bule = False # 是否为蓝方
Camera_X = -4.8 #云台相机位姿
Camera_Y = 3.8
Camera_Z = 1.2
Rotato_X = 20
Rotato_Yaw = 0
Rotato_roll = 0

red_robot_id_serial = [1, 2, 3, 4, 5] # 串口协议红/蓝方机器人id
blue_robot_id_serial = [101, 102, 103, 104, 105]

points_distance = 2

# 硬件参数：
COM = "COM4" # 串口号
SOURCE = "0" #视频/图片源 "1600x1200.png"


# yolo参数
nc = 17 # 类的个数 

# class names
names = ['car','watcher','base',
         'armor-red-1','armor-red-2','armor-red-3','armor-red-4','armor-red-5',
         'armor-red-6','armor-red-8',
         'armor-blue-1','armor-blue-2','armor-blue-3','armor-blue-4','armor-blue-5',
         'armor-blue-6','armor-blue-8']