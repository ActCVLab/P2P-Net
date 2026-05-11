import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion
import torch.nn as nn



'''
在data_transform中，transforms.HalfBody和transforms.RandomHorizontalFlip是随机执行的转换，
而transforms.AffineTransform、transforms.KeypointToHeatMap、
transforms.ToTensor和transforms.Normalize是总是会执行的转换。
'''


def nrx_get_max_preds(batch_heatmaps, num_targets, threshold, value_range):
    # 传入的参数说明：
    # batch_heatmaps: 输入的四维热图张量 [batch_size, channels, height, width]
    # num_targets: 输出目标点的最大数量
    # threshold: 目标点的置信度阈值
    # value_range: 目标点的特征值大小在最大值点的范围内的阈值比例

    # 初始化输出张量
    preds = torch.zeros((batch_heatmaps.shape[0], num_targets, 2), dtype=torch.float32)
    maxvals = torch.zeros((batch_heatmaps.shape[0], num_targets, 1), dtype=torch.float32)

    # 遍历每个批次中的热图
    for batch_idx in range(batch_heatmaps.shape[0]):
        heatmaps = batch_heatmaps[batch_idx]  # 获取当前批次的热图

        # 初始化目标点列表
        target_points = []

        # 遍历热图的通道（假设通道对应不同的特征值）
        for channel_idx in range(heatmaps.shape[0]):
            heatmap = heatmaps[channel_idx]  # 获取当前通道的热图

            # 找到所有大于阈值的点的坐标 torch.where返回一个tuple(tensor([所有目标的y坐标]),tensor([所有目标的x坐标]))
            above_threshold = torch.where(heatmap > threshold)
            y_coords, x_coords = above_threshold[0], above_threshold[1]

            # 将坐标和对应的特征值大小存储在列表中
            for i in range(len(y_coords)):
                y, x = y_coords[i], x_coords[i]
                confidence = heatmap[y, x]
                target_points.append((x, y, confidence))

        # 根据置信度对目标点进行降序排序
        target_points.sort(key=lambda x: x[2], reverse=True)

        # 选取前num_targets个点作为目标点
        selected_points = []
        for point in target_points:
            x, y, confidence = point
            valid = True

            # 检查新点的特征值是否与已选点的特征值差距不超过指定的范围 value_range * max_value (置信度越大，该数值范围越宽)
            for selected_point in selected_points:
                sx, sy, _ = selected_point
                max_value = max(heatmap[y, x], heatmap[sy, sx])
                min_value = min(heatmap[y, x], heatmap[sy, sx])
                if max_value - min_value > value_range * max_value:
                    valid = False
                    break

            if valid:
                selected_points.append(point)
                if len(selected_points) == num_targets:
                    break

        # 将选定的目标点的坐标和置信度存储在输出张量中
        for i, point in enumerate(selected_points):
            x, y, confidence = point
            preds[batch_idx, i, 0] = x
            preds[batch_idx, i, 1] = y
            maxvals[batch_idx, i, 0] = confidence

    return preds, maxvals




def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T


# 将特征点映射回特征图
def nrx_affine_points(pt, t):
    # 定义函数nrx_affine_points，接受两个参数：
    # pt：二维点的集合，每个点是一个二维坐标
    # t：仿射变换矩阵

    # 在原始坐标矩阵中记录值为零的位置
    zero_positions = np.where(pt == 0)
    # 为了进行仿射变换，给原始点坐标增加一列1，以构成齐次坐标
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt_homogeneous = np.concatenate([pt, ones], axis=1).T
    # 应用仿射变换矩阵
    new_pt_homogeneous = np.dot(t, pt_homogeneous)
    # 将变换后的齐次坐标转换回非齐次坐标（即去掉最后一列）
    new_pt = new_pt_homogeneous.T[:, :2]
    # 将变换后的点中，原始为零的位置，替换回零
    new_pt[zero_positions] = 0

    return new_pt

def nrx_heatmap_nms(batch_heatmaps):
    # 提取指定批次和通道的热图数据
    heatmap = batch_heatmaps.clone().cpu()

    # 定义3x3最大池化进行NMS，保持尺寸不变
    pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    maxm = pool(heatmap)

    maxm = torch.eq(maxm, heatmap).float()

    # 使用NMS结果更新heatmap
    heatmap = heatmap * maxm

    # 绘制热图
    # plt.clf()
    # cax = plt.imshow(heatmap[0, 0].numpy(), cmap='viridis', interpolation='nearest')
    # plt.colorbar(cax, fraction=0.046, pad=0.04)  # 显示颜色条
    # plt.savefig("output0115.png",format='png', dpi=500)
    # plt.show()

    return heatmap


def nrx_get_final_preds(batch_heatmaps: torch.Tensor,
                    trans: list = None,
                    post_processing: bool = False,
                    output_dir: str = None,
                    img_name: str = None,
                    threshold: float = None,
                    value_range: float = None,
                    max_num_targets : int = None,
                    save_heatmap: bool =False ):
    assert trans is not None
    # coords:最大分数坐标 maxvals:最大值

    # 进行非极大值抑制
    nms_batch_heatmaps = nrx_heatmap_nms(batch_heatmaps)

    ################### nrx 绘制nms热图 ##################
    if save_heatmap:
        plt.clf()
        cax = plt.imshow(nms_batch_heatmaps[0, 0].numpy(), cmap='viridis', interpolation='nearest')
        plt.colorbar(cax, fraction=0.046, pad=0.04)  # 显示颜色条
        nms_heatmap_save_path = output_dir + "\\" + img_name + "nmsheatmap.png"
        plt.savefig(nms_heatmap_save_path,format='png', dpi=500)
    #####################################################

    coords, maxvals = nrx_get_max_preds(nms_batch_heatmaps, num_targets = max_num_targets, threshold = threshold, value_range = value_range)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    ########## nrx test 绘制热图 ##########
    if save_heatmap:
        batch_index = 0
        channel_index = 0
        # 提取指定批次和通道的热图数据
        heatmap = batch_heatmaps.clone().cpu()[batch_index, channel_index, :, :]
        # heatmap_numpy = heatmap.numpy()
        # 绘制热图
        # 清除当前的图表，以确保不在旧图上绘制
        plt.clf()
        cax = plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        plt.colorbar(cax, fraction=0.046, pad=0.04)  # 显示颜色条
        heatmap_save_path = output_dir + "\\" + img_name + "heatmap.png"
        plt.savefig(heatmap_save_path, format='png', dpi=500)
        # plt.show()
    ###########################################

    # post-processing 获取最终的坐标点
    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                ######## 修改为只有一张热图 #########
                hm = batch_heatmaps[n][0]
                if (coords[n][p][0] !=0)and(coords[n][p][1]!=0) :
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = torch.tensor(
                            [
                                hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px]
                            ]
                        )
                        coords[n][p] += torch.sign(diff) * .25
                else:
                    continue

    preds = coords.clone().cpu().numpy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = nrx_affine_points(preds[i], trans[i])


    return preds, maxvals.cpu().numpy()




def decode_keypoints(outputs, origin_hw, num_joints: int = 17):
    keypoints = []
    scores = []
    heatmap_h, heatmap_w = outputs.shape[-2:]
    for i in range(num_joints):
        pt = np.unravel_index(np.argmax(outputs[i]), (heatmap_h, heatmap_w))
        score = outputs[i, pt[0], pt[1]]
        keypoints.append(pt[::-1])  # hw -> wh(xy)
        scores.append(score)

    keypoints = np.array(keypoints, dtype=float)
    scores = np.array(scores, dtype=float)
    # convert to full image scale
    keypoints[:, 0] = np.clip(keypoints[:, 0] / heatmap_w * origin_hw[1],
                              a_min=0,
                              a_max=origin_hw[1])
    keypoints[:, 1] = np.clip(keypoints[:, 1] / heatmap_h * origin_hw[0],
                              a_min=0,
                              a_max=origin_hw[0])
    return keypoints, scores


def resize_pad(img: np.ndarray, size: tuple):
    h, w, c = img.shape
    src = np.array([[0, 0],  # 原坐标系中图像左上角点
                    [w - 1, 0],  # 原坐标系中图像右上角点
                    [0, h - 1]],  # 原坐标系中图像左下角点
                   dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    if h / w > size[0] / size[1]:
        # 需要在w方向padding
        wi = size[0] * (w / h)
        pad_w = (size[1] - wi) / 2
        dst[0, :] = [pad_w - 1, 0]  # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - pad_w - 1, 0]  # 目标坐标系中图像右上角点
        dst[2, :] = [pad_w - 1, size[0] - 1]  # 目标坐标系中图像左下角点
    else:
        # 需要在h方向padding
        hi = size[1] * (h / w)
        pad_h = (size[0] - hi) / 2
        dst[0, :] = [0, pad_h - 1]  # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - 1, pad_h - 1]  # 目标坐标系中图像右上角点
        dst[2, :] = [0, size[0] - pad_h - 1]  # 目标坐标系中图像左下角点

    trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
    # 对图像进行仿射变换
    resize_img = cv2.warpAffine(img,
                                trans,
                                size[::-1],  # w, h
                                flags=cv2.INTER_LINEAR)

    ########### nrx test ########## 至此为止目标图像已经变为256*192
    plt.imshow(resize_img)
    plt.show()

    dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
    reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

    return resize_img, reverse_trans


def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):
    """通过增加w或者h的方式保证输入图片的长宽比固定"""
    xmax = xmin + w
    ymax = ymin + h

    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        # 需要在w方向padding
        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        # 需要在h方向padding
        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    return xmin, ymin, xmax, ymax


def scale_box(xmin: float, ymin: float, w: float, h: float, scale_ratio: Tuple[float, float]):
    """根据传入的h、w缩放因子scale_ratio，重新计算xmin，ymin，w，h"""
    s_h = h * scale_ratio[0]
    s_w = w * scale_ratio[1]
    xmin = xmin - (s_w - w) / 2.
    ymin = ymin - (s_h - h) / 2.
    return xmin, ymin, s_w, s_h


def plot_heatmap(image, heatmap, kps, kps_weights):
    for kp_id in range(len(kps_weights)):
        if kps_weights[kp_id] > 0:
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.plot(*kps[kp_id].tolist(), "ro")
            plt.title("image")
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap[kp_id], cmap=plt.cm.Blues)
            plt.colorbar(ticks=[0, 1])
            plt.title(f"kp_id: {kp_id}")
            plt.show()


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class HalfBody(object):
    def __init__(self, p: float = 0.3, upper_body_ids=None, lower_body_ids=None):
        assert upper_body_ids is not None
        assert lower_body_ids is not None
        self.p = p
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

    def __call__(self, image, target):
        if random.random() < self.p:
            kps = target["keypoints"]
            vis = target["visible"]
            upper_kps = []
            lower_kps = []

            # 对可见的keypoints进行归类
            for i, v in enumerate(vis):
                if v > 0.5:
                    if i in self.upper_body_ids:
                        upper_kps.append(kps[i])
                    else:
                        lower_kps.append(kps[i])

            # 50%的概率选择上或下半身
            if random.random() < 0.5:
                selected_kps = upper_kps
            else:
                selected_kps = lower_kps

            # 如果点数太少就不做任何处理
            if len(selected_kps) > 2:
                selected_kps = np.array(selected_kps, dtype=np.float32)
                xmin, ymin = np.min(selected_kps, axis=0).tolist()
                xmax, ymax = np.max(selected_kps, axis=0).tolist()
                w = xmax - xmin
                h = ymax - ymin
                if w > 1 and h > 1:
                    # 把w和h适当放大点，要不然关键点处于边缘位置
                    xmin, ymin, w, h = scale_box(xmin, ymin, w, h, (1.5, 1.5))
                    target["box"] = [xmin, ymin, w, h]

        return image, target


class AffineTransform(object):
    """scale+rotation"""

    def __init__(self,
                 scale: Tuple[float, float] = None,  # e.g. (0.65, 1.35) 表示随机缩放比例在 0.65 到 1.35 之间。
                 rotation: Tuple[int, int] = None,  # e.g. (-45, 45) 表示旋转的角度范围。例如 (-45, 45) 表示随机旋转角度在 -45 到 45 度之间。
                 fixed_size: Tuple[int, int] = (512, 512)):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, img, target):
        # 对目标框进行适应性扩大，符合 fixed_size 的比例
        # src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*target["box"], fixed_size=self.fixed_size)
        # src_w = src_xmax - src_xmin
        # src_h = src_ymax - src_ymin
        #
        # # src_center 是目标框中心点的坐标。
        # src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])

        src_center = np.array([img.shape[1] / 2, img.shape[0] / 2])

        # src_p2 和 src_p3 是用于仿射变换的关键点，分别是目标框顶部中间和右侧中间的点。
        src_p2 = src_center + np.array([0, -img.shape[0] / 2])  # top middle
        src_p3 = src_center + np.array([img.shape[1] / 2, 0])  # right middle

        dst_center = np.array([(self.fixed_size[1] - 1) / 2, (self.fixed_size[0] - 1) / 2])
        dst_p2 = np.array([(self.fixed_size[1] - 1) / 2, 0])  # top middle
        dst_p3 = np.array([self.fixed_size[1] - 1, (self.fixed_size[0] - 1) / 2])  # right middle

        if self.scale is not None:
            scale = random.uniform(*self.scale)
            src_w = img.shape[1] * scale
            src_h = img.shape[0] * scale
            src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
            src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

        if self.rotation is not None:
            angle = random.randint(*self.rotation)  # 角度制
            angle = angle / 180 * math.pi  # 弧度制
            src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
            src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])

        src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

        trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
        dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
        reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

        # 对图像进行仿射变换
        resize_img = cv2.warpAffine(img,
                                    trans,
                                    tuple(self.fixed_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR)

        # ########### nrx test 测试缩放旋转后的图像 ##########
        # plt.imshow(resize_img)
        # plt.show()

        if "keypoints" in target:
            kps = target["keypoints"]
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            kps[mask] = affine_points(kps[mask], trans)
            target["keypoints"] = kps

        ########### nrx test 测试关键点显示是否正确 ##########
        # from draw_utils import draw_keypoints
        # resize_img = draw_keypoints(resize_img, target["keypoints"])
        # plt.imshow(resize_img)
        # plt.show()

        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        return resize_img, target


class RandomHorizontalFlip(object):
    """随机对输入图片进行水平翻转，注意该方法必须接在 AffineTransform 后"""

    def __init__(self, p: float = 0.5, matched_parts: list = None):
        assert matched_parts is not None
        self.p = p
        self.matched_parts = matched_parts

    def __call__(self, image, target):
        if random.random() < self.p:
            # [h, w, c]
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            keypoints = target["keypoints"]
            visible = target["visible"]
            width = image.shape[1]

            # Flip horizontal
            keypoints[:, 0] = width - keypoints[:, 0] - 1

            # Change left-right parts
            for pair in self.matched_parts:
                keypoints[pair[0], :], keypoints[pair[1], :] = \
                    keypoints[pair[1], :], keypoints[pair[0], :].copy()

                visible[pair[0]], visible[pair[1]] = \
                    visible[pair[1]], visible[pair[0]].copy()

            target["keypoints"] = keypoints
            target["visible"] = visible

        return image, target

class nrxKeypointToHeatMap_oneMapwithManyPoints(object):
    """
    在一张特征图上生成多个特征点对应的高斯矩阵
    """

    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (256 // 4, 192 // 4),
                 gaussian_sigma: int = 2,
                 keypoints_weights=None):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights

        # generate gaussian kernel(not normalized)
        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        # print(kernel)

        self.kernel = kernel

    def __call__(self, image, target):
        kps = target["keypoints"]
        num_kps = kps.shape[0]
        kps_weights = np.ones((num_kps,), dtype=np.float32)
        # if "visible" in target:
        #     visible = target["visible"]
        #     kps_weights = visible

        # 只生成一张特征图
        heatmap = np.zeros((1, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps / 4 + 0.5).astype(int)  # round
        for kp_id in range(num_kps):

            # v = kps_weights[kp_id]
            # if v < 0.5:
            #     # 如果该点的可见度很低，则直接忽略
            #     continue

            x, y = heatmap_kps[kp_id]
            if (x,y)==(0,0):
                continue
            # 根据高斯核的大小生成特征值，后续可更改为根据放大后的标注框大小生成数值
            ul = [x - self.kernel_radius, y - self.kernel_radius]  # up-left x,y
            br = [x + self.kernel_radius, y + self.kernel_radius]  # bottom-right x,y
            # 如果特征值生成范围大于整个特征图，则忽略该点
            # 如果以xy为中心kernel_radius为半径的辐射范围内与heatmap没交集，则忽略该点(该规则并不严格，因为在图像边缘的特征点会被忽略)
            if ul[0] > self.heatmap_hw[1] - 1 or \
                    ul[1] > self.heatmap_hw[0] - 1 or \
                    br[0] < 0 or \
                    br[1] < 0:
                # If not, just return the image as is
                kps_weights[kp_id] = 0
                continue

            # Usable gaussian range
            # 计算高斯核有效区域（高斯核坐标系）
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            # image range
            # 计算heatmap中的有效区域（heatmap坐标系）
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            if kps_weights[kp_id] > 0.5:
                # # 将高斯核有效区域复制到heatmap对应区域
                # heatmap[0][img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
                #     self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

                # 将高斯核有效区域复制到heatmap对应区域，如果存在重叠，则取平均值
                for i in range(g_y[0], g_y[1] + 1):
                    for j in range(g_x[0], g_x[1] + 1):
                        heatmap_y = img_y[0] + i - g_y[0]
                        heatmap_x = img_x[0] + j - g_x[0]
                        # 如果heatmap中该位置已有值，则取平均值
                        if heatmap[0, heatmap_y, heatmap_x] > 0:
                            heatmap[0, heatmap_y, heatmap_x] = \
                                (heatmap[0, heatmap_y, heatmap_x] + self.kernel[i, j]) / 2
                        else:
                            heatmap[0, heatmap_y, heatmap_x] = self.kernel[i, j]

        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)

        ############## nrx test heatmap ###################
        # # plot_heatmap(image, heatmap, kps, kps_weights)

        ########### nrx test 绘制热图 ##########
        # # batch_index = 0
        # channel_index = 0
        # # 提取指定批次和通道的热图数据
        # heatmap = heatmap[channel_index, :, :]
        # # heatmap_numpy = heatmap.numpy()
        # # 绘制热图
        # # 清除当前的图表，以确保不在旧图上绘制
        # plt.clf()
        # cax = plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        # plt.colorbar(cax, fraction=0.046, pad=0.04)  # 显示颜色条
        # # heatmap_save_path = output_dir + "\\" + img_name + "heatmap.png"
        # # plt.savefig(heatmap_save_path, format='png', dpi=500)
        # # plt.savefig("output2024.png", format='png', dpi=500)
        # plt.show()
        ###########################################

        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)

        return image, target


if __name__ == '__main__':
    ############################# 用于测试 nrxKeypointToHeatMap_oneMapwithManyPoints 函数 #################################
    random_ndarray = np.random.rand(512, 512, 3)
    # 创建 kps_weights 字段的 ndarray
    # kps_weights = np.random.randint(0, 128, size=(17, 2))
    kps_weights_modified = np.zeros((17, 2), dtype=int)
    kps_weights_modified[:2] = np.random.randint(0, 128, size=(2, 2))

    # 创建 visible 字段的 ndarray
    visible = np.zeros(17, dtype=int)
    visible[:4] = 2

    # 创建字典
    data_dict = {
        "keypoints": np.array([(256, 256), (256, 300), (0, 0), (0, 0), (0, 0),
                               (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                               (0, 0)], dtype=int),
        "heatmap": None,  # 暂时将 heatmap 设置为 None，可根据需要填充
        # "visible": visible
    }
    target = data_dict
    nrx_test = nrxKeypointToHeatMap_oneMapwithManyPoints(heatmap_hw=(128, 128),
                                                         gaussian_sigma=2,
                                                         keypoints_weights=[2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                            0, 0])
    nrx_test(random_ndarray, target)
    #####################################################################################################################