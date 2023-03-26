class Box:
    def __init__(self, weight) -> None:
        self.left_boxes = []
        self.right_boxes = []
        self.weight = weight
    def load_txt(self, file_path = 'res/exp33/labels/merge.txt'):
        with open(file_path) as f:
            all_line = f.readlines()
            for line in all_line:
                _, x1, y1, x2, y2, c = line.split(' ')
                xyxy = [int(x1), int(y1), int(x2), int(y2), c]
                if int(x1) >= self.weight:
                    xyxy[0] -= self.weight
                    xyxy[2] -= self.weight
                    self.right_boxes.append(xyxy)
                else:
                    self.left_boxes.append(xyxy)

        self.left_boxes = sorted(self.left_boxes, key = lambda d:d[0])
        self.right_boxes = sorted(self.right_boxes, key=lambda d:d[0])

        # print('left_boxes:', self.left_boxes)
        # print('right_boxes:', self.right_boxes)
