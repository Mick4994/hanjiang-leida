
import os
import numpy as np

def read_obj(obj_path):
    with open(obj_path,'r',encoding='GB2312') as file:
        points = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append(((strs[2]), (strs[3]), (strs[4])))
            if strs[0] == "f":
                faces.append(((strs[1]),(strs[2]),(strs[3])))
    points = np.array(points)
    faces = np.array(faces)
    return points, faces

def save_pcd(filename, pcd):
    num_points = np.shape(pcd)[0]
    f = open(filename, 'w')
    f.write('# .PCD v0.7 - Point Cloud Data file format \nVERSION 0.7 \nFIELDS x y z \nSIZE 4 4 4 \nTYPE F F F \nCOUNT 1 1 1 \n')
    f.write('WIDTH {} \nHEIGHT 1 \nVIEWPOINT 0 0 0 1 0 0 0 \n'.format(num_points))
    f.write('POINTS {} \nDATA ascii\n'.format(num_points))
    for i in range(num_points):
        new_line = str(pcd[i,0]) + ' ' + str(pcd[i,1]) + ' ' + str(pcd[i,2]) + '\n'
        f.write(new_line)
    f.close()

if __name__ == "__main__":
    objfile = '3D-models/rm2022.obj'
    points, _ = read_obj(objfile)
    pcdfile = objfile.replace('.obj', '.pcd')
    save_pcd(pcdfile, points)