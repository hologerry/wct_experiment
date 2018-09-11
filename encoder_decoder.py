import torch.nn as nn
import torch


class Encoder1(nn.Module):
    def __init__(self, vgg1):
        super(Encoder1, self).__init__()
        # dissemble vgg2 and decoder2 layer by layer
        # then resemble a new encoder-decoder network
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg1.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg1.get(0).bias.float())
        # 224 x 224
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg1.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg1.get(2).bias.float())

        self.relu = nn.ReLU(inplace=True)
        # 224 x 224

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 0)
        # 224 x 224

    def forward(self, x):
        out = self.reflecPad2(x)
        out = self.conv3(out)
        return out


class Encoder2(nn.Module):
    def __init__(self, vgg2):
        super(Encoder2, self).__init__()
        # vgg2
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg2.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg2.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg2.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg2.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg2.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg2.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg2.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg2.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool = self.relu3(out)
        out, pool_idx = self.maxPool(pool)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        return out


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        # decoder
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        out = self.reflecPad5(x)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.unpool(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        return out


class Encoder3(nn.Module):
    def __init__(self, vgg3):
        super(Encoder3, self).__init__()
        # vgg3
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg3.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg3.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg3.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg3.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg3.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg3.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg3.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg3.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(vgg3.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg3.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(vgg3.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg3.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out, pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out, pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        return out


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        # decoder
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.unpool2(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out


class Encoder4(nn.Module):
    def __init__(self, vgg4):
        super(Encoder4, self).__init__()
        # vgg4
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg4.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg4.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg4.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg4.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg4.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg4.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg4.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg4.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(vgg4.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg4.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(vgg4.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg4.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(vgg4.get(19).weight.float())
        self.conv7.bias = torch.nn.Parameter(vgg4.get(19).bias.float())
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv8.weight = torch.nn.Parameter(vgg4.get(22).weight.float())
        self.conv8.bias = torch.nn.Parameter(vgg4.get(22).bias.float())
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv9.weight = torch.nn.Parameter(vgg4.get(25).weight.float())
        self.conv9.bias = torch.nn.Parameter(vgg4.get(25).bias.float())
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv10.weight = torch.nn.Parameter(vgg4.get(29).weight.float())
        self.conv10.bias = torch.nn.Parameter(vgg4.get(29).bias.float())
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out, pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out, pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        pool3 = self.relu9(out)
        out, pool_idx3 = self.maxPool3(pool3)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        return out


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out


class Encoder5(nn.Module):
    def __init__(self, vgg5):
        super(Encoder5, self).__init__()
        # vgg5
        # 224 x 224
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg5.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg5.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg5.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg5.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg5.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg5.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg5.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg5.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(vgg5.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg5.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(vgg5.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg5.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(vgg5.get(19).weight.float())
        self.conv7.bias = torch.nn.Parameter(vgg5.get(19).bias.float())
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv8.weight = torch.nn.Parameter(vgg5.get(22).weight.float())
        self.conv8.bias = torch.nn.Parameter(vgg5.get(22).bias.float())
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv9.weight = torch.nn.Parameter(vgg5.get(25).weight.float())
        self.conv9.bias = torch.nn.Parameter(vgg5.get(25).bias.float())
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 28 x 28

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv10.weight = torch.nn.Parameter(vgg5.get(29).weight.float())
        self.conv10.bias = torch.nn.Parameter(vgg5.get(29).bias.float())
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv11.weight = torch.nn.Parameter(vgg5.get(32).weight.float())
        self.conv11.bias = torch.nn.Parameter(vgg5.get(32).bias.float())
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv12.weight = torch.nn.Parameter(vgg5.get(35).weight.float())
        self.conv12.bias = torch.nn.Parameter(vgg5.get(35).bias.float())
        self.relu12 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv13.weight = torch.nn.Parameter(vgg5.get(38).weight.float())
        self.conv13.bias = torch.nn.Parameter(vgg5.get(38).bias.float())
        self.relu13 = nn.ReLU(inplace=True)
        # 28 x 28

        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 14 x 14

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv14.weight = torch.nn.Parameter(vgg5.get(42).weight.float())
        self.conv14.bias = torch.nn.Parameter(vgg5.get(42).bias.float())
        self.relu14 = nn.ReLU(inplace=True)
        # 14 x 14

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out, pool_idx = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out, pool_idx2 = self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out, pool_idx3 = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out, pool_idx4 = self.maxPool4(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        return out


class Decoder5(nn.Module):
    def __init__(self):
        super(Decoder5, self).__init__()

        # decoder
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu24 = nn.ReLU(inplace=True)

        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu26 = nn.ReLU(inplace=True)

        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        # decoder
        out = self.reflecPad15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out
