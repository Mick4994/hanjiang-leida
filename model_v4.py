import torch
from torch import nn
from torchstat import stat
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from hj_generate_v4_5 import classes

c0_inc = 1
c1_inc = 3
c1_ouc = 16
c2_ouc = 32
c3_ouc = 64
c4_ouc = 64
c5_inc = 256
c5_ouc = 32
classes = 9


class Conv(nn.Module):
    # dropout should not impact in convolutional layer
    def __init__(self, inc, ouc, k=3, s=1, p=0, g=1, act=True) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inc,
                              out_channels=ouc,
                              kernel_size=k,
                              stride=s,
                              padding=p,
                              groups=g,
                              bias=False)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):  # B,C,H,W
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class SPPNet(nn.Module):
    def __init__(self, level=2, size=2):
        super(SPPNet, self).__init__()
        self.spp = [nn.AdaptiveMaxPool2d(size ** i) for i in range(level)]  # overlap when cannot exact division
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = [self.flatten(s(x)) for s in self.spp]
        return torch.cat(x, 1)


# spell wrong, reality is BottleNeck
class BlockNeck(nn.Module):
    #  do not change the scale, just alter the channel
    def __init__(self, inc, ouc, s=1, g=1, act=True) -> None:
        super(BlockNeck, self).__init__()
        self.block_neck = nn.Sequential(
            nn.Conv2d(in_channels=inc,
                      out_channels=inc,
                      kernel_size=1,
                      stride=s,
                      groups=inc,
                      bias=False,
                      ),
            nn.Conv2d(in_channels=inc,
                      out_channels=inc,
                      kernel_size=3,
                      stride=s,
                      padding=1,
                      groups=inc,
                      bias=False
                      ),
            nn.Conv2d(in_channels=inc,
                      out_channels=ouc,
                      kernel_size=1,
                      stride=s,
                      groups=g,
                      bias=False
                      ),
        )
        self.act = nn.ReLU() if act else nn.Identity()
        self.bn = nn.BatchNorm2d(ouc)

    def forward(self, x):
        return self.act(self.bn(self.block_neck(x)))

class FC(nn.Module):
    def __init__(self, ins, ous, bias=False, drop=False, act=False, bn=True):
        super(FC,self).__init__()
        self.fc = nn.Linear(in_features=ins,
                            out_features=ous,
                            bias=bias)
        self.dropout = nn.Dropout(0.5 if drop else 0)
        self.act = nn.ReLU() if act else nn.Identity()
        self.bn = nn.BatchNorm1d(ous) if bn else nn.Identity()


    def forward(self, x):
        return self.dropout(self.act(self.bn(self.fc(x))))


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        # extend module in class will increase the memory of pytorch model file, but do not function in onnx
        self.conv0 = Conv(inc=c0_inc,ouc=c1_inc, k=3, p=1)
        self.block_neck1 = BlockNeck(inc=c1_inc, ouc=c1_ouc)  # shortcut conjunction
        self.conv1 = Conv(inc=c1_inc, ouc=c1_ouc, k=1)  # dimension up
        # Max pooling 24->12
        self.block_neck2 = BlockNeck(inc=c1_ouc, ouc=c2_ouc)
        self.conv2 = Conv(inc=c1_ouc, ouc=c2_ouc, k=1)
        # Max pooling 12->6
        self.block_neck3 = BlockNeck(inc=c2_ouc, ouc=c3_ouc)
        self.conv3 = Conv(inc=c2_ouc, ouc=c3_ouc, k=1, g=c2_ouc)
        # Inception
        self.conv4_1 = Conv(inc=c3_ouc, ouc=c4_ouc, k=1, g=c3_ouc)
        self.conv4_2 = Conv(inc=c3_ouc, ouc=c4_ouc, k=3, p=1, g=c3_ouc)
        self.conv4_3 = Conv(inc=c3_ouc, ouc=c4_ouc, k=5, p=2, g=c3_ouc)
        # self.conv4_4 = BlockNeck(inc=c3_ouc,ouc=c4_ouc,drop=True)  # 即使没有调用，pt文件也会保存这一层结构, 但onnx不会
        self.concat = Concat()
        # Max pooling 6->3
        # AdaptiveMaxPool2d是自适应kernel，故动态尺寸输入时会出现输出0尺寸，
        # AdaptiveAveragePool2d是对输入求平均即可，故不会为0尺寸输出，即使0尺寸也会为1尺寸输出
        self.dense = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # it is something like the SPPNet, SPPNet is not indispensable for this model
            Conv(inc=c5_inc, ouc=c5_ouc, k=1),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features=c5_ouc, out_features=classes, bias=False),
            nn.Softmax(1),
        )
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x=x.view(-1,1,24,24)#front is rows, back is cols
        x = self.conv0(x)
        hid1 = self.max_pool(self.block_neck1(x) + self.conv1(x))
        hid2 = self.max_pool(self.block_neck2(hid1) + self.conv2(hid1))
        hid3 = self.block_neck3(hid2) + self.conv3(hid2)
        hid5 = self.max_pool(self.concat((hid3, self.conv4_1(hid3), self.conv4_2(hid3), self.conv4_3(hid3))))
        return self.dense(hid5)


class MultiTaskModel(nn.Module):
    def __init__(self) -> None:
        super(MultiTaskModel, self).__init__()
        # extend module in class will increase the memory of pytorch model file, but do not function in onnx
        self.block_neck1 = BlockNeck(inc=c1_inc, ouc=c1_ouc)  # shortcut conjunction
        self.conv1 = Conv(inc=c1_inc, ouc=c1_ouc, k=1)  # dimension up
        # Max pooling 24->12
        self.block_neck2 = BlockNeck(inc=c1_ouc, ouc=c2_ouc)
        self.conv2 = Conv(inc=c1_ouc, ouc=c2_ouc, k=1)
        # Max pooling 12->6
        self.block_neck3 = BlockNeck(inc=c2_ouc, ouc=c3_ouc)
        self.conv3 = Conv(inc=c2_ouc, ouc=c3_ouc, k=1, g=c2_ouc)
        # Inception
        self.conv4_1 = Conv(inc=c3_ouc, ouc=c4_ouc, k=1, g=c3_ouc)
        self.conv4_2 = Conv(inc=c3_ouc, ouc=c4_ouc, k=3, p=1, g=c3_ouc)
        self.conv4_3 = Conv(inc=c3_ouc, ouc=c4_ouc, k=5, p=2, g=c3_ouc)
        # self.conv4_4 = BlockNeck(inc=c3_ouc,ouc=c4_ouc,drop=True)  # 即使没有调用，pt文件也会保存这一层结构, 但onnx不会
        # self.concat = Concat()
        # Max pooling 6->3
        self.dense = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Conv(inc=c5_inc, ouc=c5_ouc, k=1),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features=c5_ouc, out_features=classes + 2, bias=False),
        )
        self.softmax = nn.Softmax(1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x=x.view(-1,1,24,24)#front is rows, back is cols
        hid1 = self.max_pool(self.block_neck1(x) + self.conv1(x))
        hid2 = self.max_pool(self.block_neck2(hid1) + self.conv2(hid1))
        hid3 = self.block_neck3(hid2) + self.conv3(hid2)
        hid4 = self.max_pool(torch.cat((hid3, self.conv4_1(hid3), self.conv4_2(hid3), self.conv4_3(hid3)), dim=1))
        fc = self.dense(hid4)
        output = self.concat([self.softmax(fc[:, :2]), self.softmax(fc[:, 2:classes+2])])  # ends should point out
        return output

class Model9(nn.Module):
    def __init__(self) -> None:
        super(Model9, self).__init__()
        # extend module in class will increase the memory of pytorch model file, but do not function in onnx
        self.block_neck1 = BlockNeck(inc=c0_inc, ouc=c1_ouc)  # shortcut conjunction
        self.conv1 = Conv(inc=c0_inc, ouc=c1_ouc, k=1)  # dimension up
        # Max pooling 24->12
        self.block_neck2 = BlockNeck(inc=c1_ouc, ouc=c2_ouc)
        self.conv2 = Conv(inc=c1_ouc, ouc=c2_ouc, k=1)
        # Max pooling 12->6
        self.block_neck3 = BlockNeck(inc=c2_ouc, ouc=c3_ouc)
        self.conv3 = Conv(inc=c2_ouc, ouc=c3_ouc, k=1, g=c2_ouc)
        # Inception
        self.conv4_1 = Conv(inc=c3_ouc, ouc=c4_ouc, k=1, g=c3_ouc)
        self.conv4_2 = Conv(inc=c3_ouc, ouc=c4_ouc, k=3, p=1, g=c3_ouc)
        self.conv4_3 = Conv(inc=c3_ouc, ouc=c4_ouc, k=5, p=2, g=c3_ouc)
        # self.conv4_4 = BlockNeck(inc=c3_ouc,ouc=c4_ouc,drop=True)  # 即使没有调用，pt文件也会保存这一层结构, 但onnx不会
        self.concat = Concat()
        # Max pooling 6->3
        # AdaptiveMaxPool2d是自适应kernel，故动态尺寸输入时会出现输出0尺寸，
        # AdaptiveAveragePool2d是对输入求平均即可，故不会为0尺寸输出，即使0尺寸也会为1尺寸输出
        self.dense = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # it is something like the SPPNet, SPPNet is not indispensable for this model
            nn.Flatten(),
            nn.Dropout(),
            FC(ins=c5_inc,ous=c5_ouc, drop=True, act=True),
            nn.Linear(in_features=c5_ouc, out_features=classes, bias=False),
            nn.Softmax(1),
        )
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x=x.view(-1,1,24,24)#front is rows, back is cols
        hid1 = self.max_pool(self.block_neck1(x) + self.conv1(x))
        hid2 = self.max_pool(self.block_neck2(hid1) + self.conv2(hid1))
        hid3 = self.block_neck3(hid2) + self.conv3(hid2)
        hid5 = self.max_pool(self.concat((hid3, self.conv4_1(hid3), self.conv4_2(hid3), self.conv4_3(hid3))))
        return self.dense(hid5)


if __name__ == "__main__":
    model = Model()
    print(model)
    stat(model, (1, 30, 22))

