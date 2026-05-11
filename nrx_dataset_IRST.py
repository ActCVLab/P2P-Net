import os
import copy
import json
import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO


class IRST(data.Dataset):
    def __init__(self, root, dataset="train", transforms=None, det_json_path=None, fixed_size=(512, 512)):

        super().__init__()  # 调用父类（data.Dataset）的构造函数。
        assert dataset in ["train", "test"], 'dataset must be in ["train", "test"]'  # 确认数据集类型是有效的。

        # 构造注释文件路径。
        anno_file = f"{dataset}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)  # 检查根目录是否存在。
        self.img_root = os.path.join(root, dataset, f"{dataset}_images")  # 构造图像目录路径。
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)  # 检查图像目录是否存在。
        self.anno_path = os.path.join(root, dataset,"annotations", anno_file)  # 构造注释文件的完整路径。
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)  # 检查注释文件是否存在。

        self.fixed_size = fixed_size  # 设置固定的图像尺寸。
        self.mode = dataset  # 记录数据集模式（训练或验证）。
        self.transforms = transforms  # 设置图像转换方法。
        self.coco = COCO(self.anno_path)  # 加载COCO注释数据。

        with open(self.anno_path,'r') as file:
             self.annotations = json.load(file)

        img_ids = list(sorted(self.coco.imgs.keys()))  # 获取所有图像的ID。

        self.target_list = []  # 初始化目标列表。
        obj_idx = 0  # 初始化对象索引。
        for img_id in img_ids:
            # 对于每个图像ID，加载图像信息和注释。
            img_info = next((image for image in self.annotations["images"] if image["id"] == img_id), None)
            ann = next((anns for anns in self.annotations["annotations"] if anns["image_id"] == img_id), None)

            ################### 从json文件中提取出用于检测的关键信息 ################
            info = {
                "image_path": os.path.join(self.img_root, img_info["file_name"]),  # 图像路径。
                "image_id": img_id,  # 图像ID。
                "image_width": img_info['width'],  # 图像宽度。
                "image_height": img_info['height'],  # 图像高度。
                "obj_index": obj_idx,  # 对象索引。
                "score": ann["score"] if "score" in ann else 1.  # 分数（如果有）。
            }

            # 处理关键点信息。
            keypoints = np.array(ann["keypoints"]).reshape([-1, 3])  # 转换关键点格式。 将列表数据转换为[n行,3列]，其中-1是占位符，表示让 NumPy 自动计算这个维度的大小
            keypoints = keypoints[:, :2]  # 关键点位置。
            info["keypoints"] = keypoints

            self.target_list.append(info)  # 添加到有效人员列表。
            obj_idx += 1
    # 使用 DataLoader 遍历或访问数据集时，DataLoader 会自动调用 __getitem__ 方法来获取数据集中的特定项。
    def __getitem__(self, idx):
        # 获取标签json文件中有用数据
        target = copy.deepcopy(self.target_list[idx])  # 深复制目标数据。

        image = cv2.imread(target["image_path"])  # 读取图像。
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间。
        if self.transforms is not None:
            # 使用预设 transform pipline 对数据集进行变换
            image, person_info = self.transforms(image, target)


        return image, target  # 返回图像和目标数据。

    def __len__(self):
        # 返回数据集的长度。
        return len(self.target_list)

    @staticmethod
    def collate_fn(batch):
        # 批处理函数，用于数据加载器。
        imgs_tuple, targets_tuple = tuple(zip(*batch))  # 解包批数据。
        imgs_tensor = torch.stack(imgs_tuple)  # 转换为张量。
        return imgs_tensor, targets_tuple  # 返回图像张量和目标数据元组。



if __name__ == '__main__':
    train = IRST(r"N:\Scientist\111DATA\CodeTest20241209", dataset="train")
    # print(len(train))
    # t = train[0]
    # print(t)
