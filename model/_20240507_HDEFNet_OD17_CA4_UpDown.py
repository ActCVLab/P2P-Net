import torch
import torch.nn as nn
from modules.odconv import ODConv2d
# from einops import rearrange
# import math

BN_MOMENTUM = 0.1

################################## ODconv ##################################
def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)

class OD_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(OD_BasicBlock, self).__init__()
        self.conv1 = odconv3x3(inplanes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = odconv3x3(planes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class OD_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=0.0625, kernel_num=1):
        super(OD_Bottleneck, self).__init__()
        self.conv1 = odconv1x1(inplanes, planes, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = odconv3x3(planes, planes, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = odconv1x1(planes, planes * self.expansion, reduction=reduction, kernel_num=kernel_num)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class OD_StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        # 继承父类参数
        super().__init__()
        # 初始化输出、输出分支
        self.input_branches = input_branches
        self.output_branches = output_branches
        # 初始化分支列表 用于存储每个分支中basic block的个数
        self.branches = nn.ModuleList()

        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数 每个分支的通道个数是翻倍的，第一个是32，第二个是64，第三个是128，第四个是256
            branch = nn.Sequential(
                OD_BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        # 构建融合结构
        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            # 对每一分支再嵌套一个 ModuleList()
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                ############### 当输入输出平行时，不进行任何操作 ###############
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理 self.fuse_layers[-1]表示大列表中的最后一个嵌套列表
                    self.fuse_layers[-1].append(nn.Identity())
                ############### 进行上采样 ###############
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                                     # 输入通道个数为第j个输入分支 输出通道个数为第j个输出分支
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                ############### 进行下采样 ###############
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    ############### 前i-j-1个卷积层不用变通道，只进行下采样 构建 Conv模块
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    ############### 最后一个卷积层不仅要调整通道，还要进行下采样 构建 Conv2d
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

            # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)): # 表示输出通道索引
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))]) # j表示输入通道索引或特征图通道  其中 在x[j]中表示特征图通道
                )
            )

        return x_fused


############################################################################

#################################### CA ####################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
############################################################################

########################## UpDown_StageModule ##############################
class UpDown_StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        # 继承父类参数
        super().__init__()
        # 初始化输出、输出分支
        self.input_branches = input_branches
        self.output_branches = output_branches
        # 初始化分支列表 用于存储每个分支中basic block的个数
        self.branches = nn.ModuleList()

        # 实现分支下采样操作
        self.downsample_1_2 = nn.ModuleList()
        self.downsample_1_2.append(
            nn.Sequential(
                nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 2, momentum=BN_MOMENTUM)
            )
        )

        self.downsample_2_3 = nn.ModuleList()
        self.downsample_2_3.append(
            nn.Sequential(
                nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 4, momentum=BN_MOMENTUM)
            )
        )

        self.downsample_3_4 = nn.ModuleList()
        self.downsample_3_4.append(
            nn.Sequential(
                nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 8, momentum=BN_MOMENTUM)
            )
        )


        # 实现分支上采样操作
        self.upsample_2_1 = nn.ModuleList()
        self.upsample_2_1.append(
            nn.Sequential(
                # 输入通道个数为第j个输入分支 输出通道个数为第j个输出分支
                nn.Conv2d(c * 2 , c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c, momentum=BN_MOMENTUM),
                nn.Upsample(scale_factor=2.0 , mode='nearest')
            )
        )

        self.upsample_3_2 = nn.ModuleList()
        self.upsample_3_2.append(
            nn.Sequential(
                # 输入通道个数为第j个输入分支 输出通道个数为第j个输出分支
                nn.Conv2d(c * 4, c * 2, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c * 2, momentum=BN_MOMENTUM),
                nn.Upsample(scale_factor=2.0, mode='nearest')
            )
        )

        self.upsample_4_3 = nn.ModuleList()
        self.upsample_4_3.append(
            nn.Sequential(
                # 输入通道个数为第j个输入分支 输出通道个数为第j个输出分支
                nn.Conv2d(c * 8, c * 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c * 4, momentum=BN_MOMENTUM),
                nn.Upsample(scale_factor=2.0, mode='nearest')
            )
        )




        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数 每个分支的通道个数是翻倍的，第一个是32，第二个是64，第三个是128，第四个是256
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        # 构建融合结构
        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            # 对每一分支再嵌套一个 ModuleList()
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理 self.fuse_layers[-1]表示大列表中的最后一个嵌套列表
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                                     # 输入通道个数为第j个输入分支 输出通道个数为第j个输出分支
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1 对每个分支的第1个BasicBlock[0]进行前向传播
        BasicBlock0_outputs = []
        for idx, (branch, xi) in enumerate(zip(self.branches, x)):
            BasicBlock0_output = branch[0](xi)
            BasicBlock0_outputs.append(BasicBlock0_output)

        # 对第一分支的BasicBlock[0]输出进行下采样，并与第二分支的BasicBlock[0]输出融合
        if self.input_branches > 1 and len(BasicBlock0_outputs) >= 3:
            BasicBlock0_0downsampled_output = self.downsample_1_2[0](BasicBlock0_outputs[0])
            BasicBlock0_outputs[1] = self.relu(BasicBlock0_outputs[1] + BasicBlock0_0downsampled_output)
            # 对第二分支融合后的BasicBlock[0]输出进行下采样，并与第三分支的BasicBlock[0]输出融合
            BasicBlock0_1downsampled_output = self.downsample_2_3[0](BasicBlock0_outputs[1])
            BasicBlock0_outputs[2] = self.relu(BasicBlock0_outputs[2] + BasicBlock0_1downsampled_output)

        if self.input_branches > 1 and len(BasicBlock0_outputs) >= 4:
            # 对第三分支融合后的BasicBlock[0]输出进行下采样，并与第四分支的BasicBlock[0]输出融合
            BasicBlock0_2downsampled_output = self.downsample_3_4[0](BasicBlock0_outputs[2])
            BasicBlock0_outputs[3] = self.relu(BasicBlock0_outputs[3] + BasicBlock0_2downsampled_output)



        # 2 对每个分支的第2个BasicBlock[1]进行前向传播
        BasicBlock1_outputs = []
        for idx, (branch, BasicBlock0_output) in enumerate(zip(self.branches, BasicBlock0_outputs)):
            BasicBlock1_output = branch[1](BasicBlock0_output)
            BasicBlock1_outputs.append(BasicBlock1_output)

        # 对第一分支的BasicBlock[0]输出进行下采样，并与第二分支的BasicBlock[0]输出融合
        if self.input_branches > 1 and len(BasicBlock1_outputs) >= 3:
            BasicBlock1_0downsampled_output = self.downsample_1_2[0](BasicBlock1_outputs[0])
            BasicBlock1_outputs[1] = self.relu(BasicBlock1_outputs[1] + BasicBlock1_0downsampled_output)
            # 对第二分支融合后的BasicBlock[0]输出进行下采样，并与第三分支的BasicBlock[0]输出融合
            BasicBlock1_1downsampled_output = self.downsample_2_3[0](BasicBlock1_outputs[1])
            BasicBlock1_outputs[2] = self.relu(BasicBlock1_outputs[2] + BasicBlock1_1downsampled_output)

        if self.input_branches > 1 and len(BasicBlock1_outputs) >= 4:
            # 对第三分支融合后的BasicBlock[0]输出进行下采样，并与第四分支的BasicBlock[0]输出融合
            BasicBlock1_2downsampled_output = self.downsample_3_4[0](BasicBlock1_outputs[2])
            BasicBlock1_outputs[3] = self.relu(BasicBlock1_outputs[3] + BasicBlock1_2downsampled_output)



        # 3 对每个分支的第3个BasicBlock[2]进行前向传播
        BasicBlock2_outputs = []
        for idx, (branch, BasicBlock1_output) in enumerate(zip(self.branches, BasicBlock1_outputs)):
            BasicBlock2_output = branch[2](BasicBlock1_output)
            BasicBlock2_outputs.append(BasicBlock2_output)
        # 对第一分支的BasicBlock[2]输出进行下采样，并与第二分支的BasicBlock[2]输出融合
        if self.input_branches > 1 and len(BasicBlock2_outputs) >= 3:
            BasicBlock2_0downsampled_output = self.downsample_1_2[0](BasicBlock2_outputs[0])
            BasicBlock2_outputs[1] = self.relu(BasicBlock2_outputs[1] + BasicBlock2_0downsampled_output)
            # 对第二分支融合后的BasicBlock[2]输出进行下采样，并与第三分支的BasicBlock[2]输出融合
            BasicBlock2_1downsampled_output = self.downsample_2_3[0](BasicBlock2_outputs[1])
            BasicBlock2_outputs[2] = self.relu(BasicBlock2_outputs[2] + BasicBlock2_1downsampled_output)

        # 对第三分支的BasicBlock[2]输出进行下采样，并与第四分支的BasicBlock[2]输出融合
        if self.input_branches > 1 and len(BasicBlock2_outputs) >= 4:
            BasicBlock2_2downsampled_output = self.downsample_3_4[0](BasicBlock2_outputs[2])
            BasicBlock2_outputs[3] = self.relu(BasicBlock2_outputs[3] + BasicBlock2_2downsampled_output)


        # 4 对每个分支的第4个BasicBlock[3]进行前向传播
        BasicBlock3_outputs = []
        for idx, (branch, BasicBlock2_output) in enumerate(zip(self.branches, BasicBlock2_outputs)):
            BasicBlock3_output = branch[3](BasicBlock2_output)
            BasicBlock3_outputs.append(BasicBlock3_output)

        # 对第一分支的BasicBlock[2]输出进行下采样，并与第二分支的BasicBlock[2]输出融合
        if self.input_branches > 1 and len(BasicBlock3_outputs) >= 3:
            BasicBlock3_0downsampled_output = self.downsample_1_2[0](BasicBlock3_outputs[0])
            BasicBlock3_outputs[1] = self.relu(BasicBlock3_outputs[1] + BasicBlock3_0downsampled_output)
            # 对第二分支融合后的BasicBlock[2]输出进行下采样，并与第三分支的BasicBlock[2]输出融合
            BasicBlock3_1downsampled_output = self.downsample_2_3[0](BasicBlock3_outputs[1])
            BasicBlock3_outputs[2] = self.relu(BasicBlock3_outputs[2] + BasicBlock3_1downsampled_output)

        # 对第三分支的BasicBlock[2]输出进行下采样，并与第四分支的BasicBlock[2]输出融合
        if self.input_branches > 1 and len(BasicBlock2_outputs) >= 4:
            BasicBlock3_2downsampled_output = self.downsample_3_4[0](BasicBlock3_outputs[2])
            BasicBlock3_outputs[3] = self.relu(BasicBlock3_outputs[3] + BasicBlock3_2downsampled_output)




        # 最后，更新最终输出列表x为每个分支的最后输出
        x = BasicBlock3_outputs

        # 进行原有的融合操作
        x_fused = []
        for i in range(len(self.fuse_layers)):  # 表示输出通道索引
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])  # j表示输入通道索引
                )
            )

        return x_fused
############################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        # 继承父类参数
        super().__init__()
        # 初始化输出、输出分支
        self.input_branches = input_branches
        self.output_branches = output_branches
        # 初始化分支列表 用于存储每个分支中basic block的个数
        self.branches = nn.ModuleList()

        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数 每个分支的通道个数是翻倍的，第一个是32，第二个是64，第三个是128，第四个是256
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        # 构建融合结构
        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            # 对每一分支再嵌套一个 ModuleList()
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理 self.fuse_layers[-1]表示大列表中的最后一个嵌套列表
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                                     # 输入通道个数为第j个输入分支 输出通道个数为第j个输出分支
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]



            # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)): # 表示输出通道索引
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))]) # j表示输入通道索引或特征图通道  其中 在x[j]中表示特征图通道
                )
            )

        return x_fused


class HighResolutionNet_OD17_CA4_UpDown_Final(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 17):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            OD_Bottleneck(256, 64)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            #nn.Identity(),  # None,  - Used in place of "None" because it is callable
            CoordAtt(base_channel, base_channel), # add by nrx
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            ################ 该部分进行下采样 ###############
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            UpDown_StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            OD_StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        # transition3
        self.transition3 = nn.ModuleList([
            CoordAtt(base_channel, base_channel),  # add by nrx
            CoordAtt(base_channel * 2, base_channel * 2),  # add by nrx
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            OD_StageModule(input_branches=4, output_branches=1, c=base_channel)

        )

        self.coordAtt = CoordAtt(base_channel, base_channel) # add by nrx

        # Final layer
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        x = self.coordAtt(x[0]) # add by nrx

        ################## 使用多级特征图回归需在此添加代码   ##################
        x = self.final_layer(x) # change by nrx

        return x

if __name__ =="__main__":
    cmd = HighResolutionNet_OD17_CA4_UpDown_Final(32, 1)
    input = torch.randn(2, 3, 640, 640)
    output = cmd(input)
    print(output.shape)
    print(cmd)
