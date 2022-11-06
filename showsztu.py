import open3d as o3d
import numpy as np
import os
import time

# import cv2

# img = np.array([255,255,255])
# cv2.cvtColor(img,cv2.RGB)
start_time = time.time()

def get_runtime() -> str:
    runtime = '{:.1f}'.format(time.time() - start_time)
    return 'runtime:'+ runtime + 's '

def forward():
    f = open('256.txt','r')
    all_lines = f.readlines()
    f.close()

    value_list = []

    for line in all_lines:
        rgb_index = line.index('rgb')
        rgb_end_index = rgb_index + line[line.index('rgb'):-1].index(')')
        value_str = [int(i) for i in line[rgb_index+4:rgb_end_index].split(',')]
        value_list.append(value_str)

    value_array = np.array(value_list, dtype=np.uint8)

    print(get_runtime()+'loading pcd')
    all_pcd_file = os.listdir('3D-models/terra_pcd/')
    all_pcd_file = ['3D-models/terra_pcd/' + i for i in all_pcd_file if i.endswith('.pcd')]
    pcd_list = [o3d.io.read_point_cloud(pcd_file) for pcd_file in all_pcd_file]
    # print('finished load')

    xyz_list = []
    color_list = []

    print(get_runtime()+'getting xyz and colors')
    count = 0
    for i in pcd_list:
        xyz_list += np.asarray(i.points).tolist()
        color_list += np.asarray(i.colors).tolist()
        count += 1
        print(get_runtime()+' get:',count)

    # print(get_runtime()+'finish get, turning to array')

    # xyz = np.array(xyz_list)
    # colors = np.array(color_list)
    # print(get_runtime()+'finish turn')

    int_xyz_dic = {}
    for i in range(len(xyz_list)):
        int_xyz_dic[str([int(xyz_list[i][0]),int(xyz_list[i][1]),int(xyz_list[i][2])])] = color_list[i]
        if i!= 0:
            print(get_runtime() + 'simplfing:{:.2f}%'.format(i*100/len(xyz_list)),'len int_xyz_dic/i:{:.2f}'.format(len(int_xyz_dic)/i), end='\r')
    print('\n',end='')

    int_xyz = []
    int_colors = []
    count = 0
    for key, value in int_xyz_dic.items():
        key = key[1:-1]
        a,b,c = key.split(',')
        int_xyz.append([int(a),int(b),int(c)])
        int_colors.append(value)
        count += 1
        print(get_runtime() + 'rebuild:{:.2f}%'.format(count*100/len(int_xyz_dic)), end='\r')
    print('\n',end='')
    np.save('int_xyz.npy',np.array(int_xyz))
    np.save('int_colors.npy',np.array(int_colors))
    print(get_runtime() + 'finish save numpy data')



def match_color():
    f = open('256.txt','r')
    all_lines = f.readlines()
    f.close()
    value_list = []

    for line in all_lines:
        rgb_index = line.index('rgb')
        rgb_end_index = rgb_index + line[line.index('rgb'):-1].index(')')
        value_str = [int(i) for i in line[rgb_index+4:rgb_end_index].split(',')]
        value_list.append(value_str)
    colors = np.load('int_colors.npy')

    match_colors_list = []
    colors_255 = np.array(colors * 255, dtype = np.uint8)
    for i in range(len(colors_255)):
        diff_list = [sum([abs(value_list[j][k] - colors_255[i][k]) for k in range(len(value_list[j]))]) for j in range(len(value_list))]
        min_index = diff_list.index(min(diff_list))
        match_colors_list.append(np.array(value_list[min_index]))
        print(get_runtime()+'matching:{:.2f}'.format(i*100/len(colors_255)),"%",end='\r')
    print('\nfinish match')
    match_colors = np.array(match_colors_list)/255

    print(get_runtime()+'saving result')
    f = open('match_colors.txt','w')
    for i in range(len(match_colors_list)):
        f.write(str(match_colors_list[i]) + '\n')
        print(get_runtime()+'saving:{:.2f}'.format(i*100/len(match_colors_list)), end='\r')
    print('\n',end='')
    f.close()

    xyz = np.load('int_xyz.npy')
    return match_colors, xyz

def xyz_dis():
    x, y, z = zip(*np.asarray(sztu_pcd.points))
    print('x:',max(x)-min(x))
    print('y:',max(y)-min(y))
    print('z:',max(z)-min(z))

def save_colors_255():
    colors = np.load('int_colors.npy')
    colors_255 = np.array(colors * 255, dtype = np.uint8)
    f = open('colors_255.txt','w')
    count = 0
    for i in colors_255:
        f.write(str(i)+'\n')
        count += 1
        print(get_runtime() + 'save colors_255.txt:{:.2f}%'.format(count*100/len(colors_255)),end='\r')
    print('\n',end='')
    f.close()

if __name__ == '__main__':
    # match_colors, xyz = match_color()
    # print(colors_255)
    # xyz_dis()
    # sztu_pcd.colors = o3d.utility.Vector3dVector(colors)
    # print(get_runtime()+'loading Vision')
    xyz = np.load('int_xyz.npy')
    match_colors = np.load('int_colors.npy')
    # print('len(xyz):',len(xyz),'len(match_colors):',len(match_colors))
    # sztu_pcd = o3d.geometry.PointCloud()
    # sztu_pcd.points = o3d.utility.Vector3dVector(xyz)
    # sztu_pcd.colors = o3d.utility.Vector3dVector(match_colors)
    # o3d.visualization.draw([sztu_pcd])
    



