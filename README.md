# 深圳技术大学悍匠战队雷达

>开发者：悍匠视觉组成员——许咏琪 <br>
>v1.5版前的外观建模和3D打印：许咏琪 <br>
>v1.5版后两任务移交：悍匠机械组成员——洪俊钦 <br>
>文档编写：许咏琪

## 跳转目录

- [介绍](#介绍)<br>
- [环境配置](#环境配置)<br>
    - [部署](#部署)<br>
    - [开发](#开发)<br>
- [文件目录结构](#文件目录结构)<br>
- [技术积累与算法](#技术积累与算法)<br>

## 介绍
本雷达采用纯视觉解算算法，采用双海康工业相机对准关键区域，进行图形学解算，将相机二维画面投射到虚拟的三维映射空间中（其中设计大量坐标系转换），因此目标监测和识别结果将通过这个映射关系获得裁判系统所需的三维坐标，并通过串口发送，至此完成一轮算法周期。

- 所用到的技术和理论：OpenCV，相机光学，qt界面设计，数据集的预处理，目标检测算法的训练和推理调参，串口通讯

- 关于算法，图形学三维转换和坐标系转换部分是该工程的主要研究部分，是数据处理的下游和初始化部分，然后上游是目标检测和识别的处理，而目标检测和识别算法目前已经很火，网上已经有很多成熟的训练和推理yolov5教程，具体可在b站或CSDN搜索相关教程。

- 关于测试，由于雷达测试条件很苛刻，所以在这里我是用blender这个3d建模软件（如下图），通过尽可能地调整成比赛环境来模拟。然后在此基础上做一个光追渲染的动画，将小车的运动渲染成视频导出，然后基于视频做调试，参数调整之类。

<div align="center">
<img src="doc/blender仿真.png"/> 
<br>
图（blender仿真工程界面）
</div>
<br>

- 关于二次开发，由于blender其实并不适合做仿真环境（本质是做设计，影视和动画的工业软件，依赖后期渲染），所以后续可能会考虑用UE或者Unity这些游戏引擎来做仿真引擎更加符合需求（实时渲染和机制设定）。然后就是每个算法周期的性能高度依赖于目标检测算法的推理速度，所以后续应该会不断更新目标检测的算法，使之更快和性能开销更低。

## 环境配置
开发平台为Windows11，基于python开发，pytorch的yolov5框架，所以需要在Windows下安装python和pytorch的gpu版本，再安装cuda，然后在终端pip安装所有用到的第三方库。

建议部署在中高性能的游戏本上，一个是因为yolov5需要高性能的显卡进行推理，二是为了便携以方便比赛前的准备和比赛时时的布置。这里开发者采用的是CPU为i5 11400H，GPU为RTX 3060，内存为16GB大小的游戏本，运行帧率大概在40-50FPS。

#### <div id="部署"></div>如果单纯是部署运行，则具体步骤如下：

1. 安装python3.8或以上，单独安装cuda_11.0,<br>`./torch-1.7.1+cu110-cp38-cp38-win_amd64.whl`,<br>`./torchvision-0.8.2+cu110-cp38-cp38-win_amd64.whl`，<br>这三个我都放在百度网盘中：[cuda-11.0](https://pan.baidu.com/s/1LSHlWMMSNM4RKag0VI8GmQ?pwd=dt38) ，[torch-1.7.1](https://pan.baidu.com/s/13H1zv93IP6blQZIZQcBnWg?pwd=b566) 和 [torchvision-0.8.2](https://pan.baidu.com/s/1MIAymG-uJrRqLPa6ttnC4A?pwd=jk37)，除了cuda，其他是该文件目录下打开终端，然后执行`pip install [此处替换为以上高亮内容]`，以上有两个包，所以替换两次，如`pip install ./torch-1.7.1+cu110-cp38-cp38-win_amd64.whl`

2. 安装完后，在powershell或者cmd中运行`pip install -r requirements.txt`安装所有第三方依赖，建议用魔法科学上网，可以更快完成。

3. 在工程目录下(`./hanjiang-leida/`)下，打开终端执行`python launch.py`即可运行雷达程序

4. 然后会弹出一个调参的界面，通过观察实时画面的点云，来对相机外参进行调参（如下图所示），接着调整界面右侧的相机外参即可移动点云，调整至与场地匹配即可。

<div align="center">
<img src="doc/雷达调参.png"/> 
<br>
图（雷达运行后的调参界面）
</div>
<br>

#### <div id="开发"></div>如果是作为开发，则具体步骤如下：

- **在完成部署步骤的基础上开始！！**

1. 建议使用PyCharmIDE或者配置完python环境插件VSCode进行开发。所以可以根据他们各自的优缺点进行取舍(PyCharm功能全，配置工程简单，但是启动慢，占用内存大，VSCode轻量化，启动快，但是配置杂)，二选其一或者都进行安装。

2. 在工程目录下打开为项目，VSCode打开为工作区，然后建议使用git进行代码管理。

3. 在进行修改和二次开发前请务必熟读文档，熟读后和上网搜索解决不了的问题再发邮件询问开发者

- 代码逻辑：
    - 主函数在`launch.py`，然后用多线程分别启动目标检测线程，主干算法线程，界面线程和串口线程
    - 若按模块划分，则分为qt界面模块，日志模块，yolov5模块，串口收发模块，装甲板识别模块和图形学转换模块（详细见文件目录结构）

- 附: 建议通过修改注册表等能彻底关闭Windows更新的方法防止紧急情况时更新

## 文件目录结构

注：
- 其中很多文件夹里都有`__pycache__`文件夹，这是python运行后的缓存文件，已经添加在.gitignore了，不用管。
- ./前缀的为文件夹，否则为文件，用... ...代替的是不重要的文件，都是一些临时测试的文件
- 如果是通过git clone的没有./res目录下的视频资源，请点这里[百度网盘下载](https://pan.baidu.com/s/1QX-9IK-TDzhJDhOQn4JQpQ?pwd=v9my)
```
├─./.idea----pycharm配置文件，忽略
│  └─inspectionProfiles
├─./.vscode----vscode配置文件，忽略
├─./deep_sort_pytorch----deepsort目标追踪算法的文件夹，对追踪有更高要求时调参时再看
│  ├─./config----主要调参的就是其中的deep_sort.yaml文件，具体参数调整教程可以搜关于deepsort调参的CSDN
│  ├─./deep_sort
│  │  ├─./deep
│  │  ... ...
├─./doc----放置文档的资源文件夹
├─./example----open3d测试激光雷达点云的文件夹
├─./log----日志文件夹
│  └─./serial_log----串口收发日志
│     └─ ... ...
├─./models----yolo模型配置文件夹，可不管
│  └─ ... ...
├─./output----deepsort输出文件夹，可不管
├─./res----第一个重点文件夹！！！存放测试资源文件夹
│  ├─./3D-models----场地点云和网格模型（论坛找的）
│  ├─./exp21----exp前缀的文件夹均是在yolo推理的输出文件，拿来测试用
│  │  └─./labels----存放输出的检测框的标签和坐标的txt文件夹
│  ├─./exp22
│  │  └─./labels
│  ├─./exp31
│  │  └─./labels
│  ├─./exp33
│  │  └─./labels
│  ├─./mask----先前在v1.5版用作生成场地高低差的点云遮罩
│  ├─./maskv2----v2版本后的遮罩
│  ├─./o3Ddata----open3d测试的配置文件
│  │  ├─./DepthCamera_JSON
│  │  ├─./DepthCapture_PNG
│  │  ├─./RenderOption_JSON
│  │  ├─./ScreenCamera_JSON
│  │  └─./ScreenCamera_PNG
│  ├─./true_location----blender脚本导出的车辆每一帧位置的数据
│  ├─./video_left----blender内左相机（仿真小相机）的yolo输出
│  │  └─./labels
│  ├─./video_right----blender内右相机的yolo输出
│  │  └─./labels
│  ├─1080P5min.mp4----小工业相机blender仿真测试视频
│  ├─left.mp4----海康左相机blender仿真测试视频
│  ├─right.mp4----海康右相机blender仿真测试视频
│  ├─hiv_left1.avi----海康左相机长沙比赛录制视频1
│  ├─hiv_left2.avi----海康左相机长沙比赛录制视频2
│  ├─hiv_right1.avi----海康右相机长沙比赛录制视频1
│  ├─hiv_right2.avi----海康右相机长沙比赛录制视频2
│  ├─new_ui.png----调参界面gui背景图
│  ├─rm2023_map.png----调参界面的平面地图
│  └─ ... ...
├─./src----第二个重点文件夹！！！存放主要算法代码（具体解析看算法一栏）
│  ├─./cv_utils----坐标系转换和二维到三维的映射算法
│  ├─./math----双目测距定位算法（v1版前的算法，暂时弃用）
│  ├─./myserial----串口工具
│  ├─./qt----调参界面设计与各版迭代
│  ├─./tools----场地，地图gui或者双目相机视频流合并的处理脚本
├─./test----带jupyter notebook文件，有时写一些测试代码用，其中有用到Open3d库
├─./utils----完整的yolo的工具包
├─./weights----权重（放置yolov5.5训练的模型）文件
├─./yolo----简化的yolo包，可不管，暂时弃用
├─./yolov5----v1前用的目标检测，暂时弃用
├─launch.py----启动主函数的代码文件 
├─arguments.py----启动参数文件（重要文件！！！）
├─Solution.py----主干算法周期（重要文件！！！）,其中包含v1和v2版本，以类名做区分了
├─map_demo.py----测试给平面地图标点
├─track.py----deepsort目标追踪代码的主函数，被Sulution调用
├─detect.py----yolov5目标检测代码主函数，被launch以多线程启动
├─modelv4.py----装甲板识别的神经网络结构类的定义，被pytorch隐性调用
├─requirements.txt----记录pip安装第三方依赖的文本文件
├─.gitignore----git忽略管理的文件/文件夹
... ...
```

## 技术积累与算法

### 算法
1. 最关键的算法就是获取二维转三维（重要）的映射方法，位于`./src/cv_utils/projectPoints.py`的`Maper`类下的`get_points_map`方法，处理完参数后调用了`OpenCV`的`projectPoints`函数去解算在相机坐标系下的点云投影在像素坐标系 

    - 关于`OpenCV`的`projectPoints`函数的详细解析与说明可以上OpenCV官方文档或者CSDN看，这里只贴出参数说明（见下图）
    <br>
    <br>
    <div align="center">
    <img src="doc/projectPoints函数.png"/> 
    <br>
    projectPoints函数的参数说明图
    </div>
    <br>

    - 如果想要深入了解原理，其中涉及到的，专业方面是相机原理和计算机图形学知识，通识方面是线性代数，这里给想要了解的人推荐两个视频
    1. [无所不能的矩阵 - 三维图形变换](https://www.bilibili.com/video/BV1b34y1y7nF)
    2. [探秘三维透视投影 - 齐次坐标的妙用](https://www.bilibili.com/video/BV1LS4y1b7xZ)

2. 另一个重要的是坐标系转换，在这个工程中涉及三个坐标系（RM坐标系，CV坐标系，相机坐标系），同时RM坐标系也可细分为红蓝双方不同的子坐标系。
