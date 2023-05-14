import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
from random import randint
from arguments import *
from yolo.experimental_copy import attempt_load
from yolo.datasets_copy import LoadWebcam, LoadImages
from yolo.general_copy import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, plot_one_box
class YOLO_Detect:
    def __init__(self, source = '2') -> None:
        self.source:str = source
        self.weights:str = 'yolov5/weights/' + Yolo_Weight
        self.imgsz: int = 1280
        self.conf_thres: float = 0.50
        self.iou_thres: float = 0.45
        self.augment:bool = False
        self.classes = None
        self.agnostic_nms:bool = False

        self.src_image = []
        self.output = []

    def detect(self, ctd):
        source, weights, imgsz = self.source, self.weights, self.imgsz

        device = torch.device('cuda:0')
        half = device.type != 'cpu'  # half precision only supported on CUDA

        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        
        # Set Dataloader
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadWebcam(ctd, source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        
        for _, img, im0s, _ in dataset:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=self.augment)[0]

            # print(" opt.classes:", opt.classes)
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

            # Process detections
            for _, det in enumerate(pred):  # detections per image
                im0, frame = im0s, getattr(dataset, 'frame', 0)
                
                self.src_image = im0
                # cv2.namedWindow("im0", cv2.WINDOW_NORMAL)
                # cv2.imshow("im0", im0)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    lines = []
                    for *xyxy, conf, cls in reversed(det):
                        # if cls == 0:
                        xyxy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        line = (cls, *xyxy, conf)  # label format
                        lines.append(line)
                        # with open('output.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        label = f'{names[int(cls)]} {conf:.2f}'
                        # print("im0.shape:", im0.shape)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    self.output.append(lines)
                    if len(self.output) > 100:
                        self.output = self.output[-100:]
                    # else:
                    #     print(len(self.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/robomaster.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='merge.png', 
                        help="source,'robomaster_data/images/test/test.mp4','801_res.mp4','merge.png'")
    parser.add_argument('--img-size', type=int, default=3840, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    opt = parser.parse_args()
    print(opt)

    yolo_detect = YOLO_Detect()

    with torch.no_grad():
        yolo_detect.detect()
