import math
import sys
import time
import numpy as np
import torch
import nrx_irst_transforms
import train_utils.distributed_utils as utils
import os
from .loss import KpLoss
import json

# 定义一个训练周期的函数
def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    # 将模型设置为训练模式
    model.train()
    # 初始化一个度量记录器，用于记录训练过程中的各种指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 添加一个记录学习率的度量器，只记录最近的一个值
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 设置日志头信息，显示当前是哪一个epoch
    header = 'Epoch: [{}]'.format(epoch)

    # 初始化学习率调度器为None
    lr_scheduler = None
    # 如果是第一轮训练，并且启用了warmup（热身训练）
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000  # 设置warmup因子
        warmup_iters = min(1000, len(data_loader) - 1)  # 设置warmup迭代次数

        # 创建一个warmup学习率调度器
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 初始化损失函数
    mse = KpLoss()
    # 初始化平均损失
    mloss = torch.zeros(1).to(device)
    # 遍历数据加载器
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 将图像数据移动到指定的设备上
        images = images.to(device)

        # 如果指定了scaler，则使用混合精度训练，否则正常执行
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = model(images)  # 前向传播，获取模型输出

            # 计算损失
            losses = mse(results, targets)

        # 在所有GPU上减少损失，用于日志记录目的
        loss_dict_reduced = utils.reduce_dict({"losses": losses})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()  # 获取损失值
        # 更新平均损失
        mloss = (mloss * i + loss_value) / (i + 1)

        # 如果损失值不是有限数，则停止训练
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 梯度清零
        optimizer.zero_grad()
        if scaler is not None:  # 使用混合精度进行反向传播和参数更新
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # 不使用混合精度，直接进行反向传播和参数更新
            losses.backward()
            optimizer.step()

        # 如果使用了warmup学习率调度器，则对其进行更新
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 更新度量记录器中的损失和学习率信息
        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    # 返回平均损失和当前学习率
    return mloss, now_lr


def validate_one_epoch(model, data_loader, device, scaler=None):
    model.eval()  # 设置模型为评估模式
    metric_logger = utils.MetricLogger(delimiter="  ")
    mse = KpLoss()  # 假设损失函数与训练时相同
    total_loss = 0
    num_loss_cal = 0
    with torch.no_grad():  # 不计算梯度
        for images, targets in metric_logger.log_every(data_loader, 50, "Val loss"):
            images = images.to(device)
            # 假设targets已经在正确的格式中，且mse函数能够直接处理
            # 如果targets需要特别处理，请在这里添加处理代码

            # 使用混合精度进行前向传播
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                i=outputs.shape[0]
                loss = mse(outputs, targets)  # 直接调用mse计算损失
                # 统计损失计算次数
                num_loss_cal += 1

            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"calculate loss: {num_loss_cal} times")
    return avg_loss
############################################## for val ###############################################
def get_gt_keypoints(gt_data, image_id):
    """
    从GT数据中获取指定图像的关键点
    """
    keypoints = []
    for annotation in gt_data['annotations']:
        if annotation['image_id'] == image_id:
            keypoints_data = annotation['keypoints']
            for i in range(0, len(keypoints_data), 3):
                keypoints.append(keypoints_data[i:i+2])  # 获取 x, y 坐标
    return np.array(keypoints)

# 计算两点之间的欧氏距离
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 根据预测关键点和真实关键点计算TP, FP, FN
def calculate_metrics(pred_keypoints, gt_keypoints, tp_distance):
    tp = 0
    fp = 0
    fn = 0
    matched_gt = set()
    matched_pred = set()  # 新增：用于跟踪已匹配的预测关键点

    for pred_index, pred in enumerate(pred_keypoints):
        if np.array_equal(pred, [0, 0]) or pred_index in matched_pred:  # 忽略占位符关键点和已匹配的预测关键点
            continue
        # 初始化closest_dist变量为无穷大，用于寻找到当前预测关键点最近的真实关键点
        closest_dist = np.inf
        # 初始化closest_index变量为None，用于存储最近真实关键点的索引。
        closest_index = None
        for i, gt in enumerate(gt_keypoints):
            if i not in matched_gt:  # 只考虑尚未匹配的真实关键点
                dist = calculate_distance(pred, gt)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_index = i
        if closest_dist <= tp_distance:
            tp += 1
            matched_gt.add(closest_index)
            matched_pred.add(pred_index)  # 标记这个预测关键点为已匹配
        else:
            fp += 1

    fn = len(gt_keypoints) - len(matched_gt)  # 未匹配的真实关键点被认为是FN
    return tp, fp, fn

def nrx_evaluate(model, data_loader, device, scaler, threshold, value_range, max_num_targets, output_dir, data_path, tp_distance):
    model.eval()  # 设置模型为评估模式

    metric_logger = utils.MetricLogger(delimiter="  ")
    # 设置tp fp fn计算初始值
    total_tp = 0
    total_fp = 0
    total_fn = 0

    fn_fp_image_ids = []  # 用于存储有FN和FP的图像ID

    anno_file = f"test.json"
    assert os.path.exists(data_path), "path '{}' does not exist.".format(data_path)  # 检查图像目录是否存在。
    gt_json_path = os.path.join(data_path, "test", "annotations", anno_file)  # 构造注释文件的完整路径。
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)


    # 遍历多批次文件
    with torch.no_grad():  # 不计算梯度
        for images, targets in metric_logger.log_every(data_loader, 50, "Evaluate"):

            images = images.to(device)
            # 假设targets已经在正确的格式中，且mse函数能够直接处理
            # 如果targets需要特别处理，请在这里添加处理代码


            # 使用混合精度进行前向传播
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                for i in range(outputs.shape[0]):
                    # 对每张图像进行计算
                    output = outputs[i]
                    # 使用unsqueeze函数在指定维度添加一个维度
                    output = output.unsqueeze(0)
                    target = targets[i]
                    img_name = None
                    keypoints, scores = nrx_irst_transforms.nrx_get_final_preds(output, [target["reverse_trans"]],
                                                                                True, output_dir,
                                                                                img_name, threshold, value_range,
                                                                                max_num_targets)
                    keypoints = np.squeeze(keypoints)
                    if keypoints.ndim == 1:
                        keypoints = np.array([keypoints])

                    image_id = target["image_id"]

                    gt_keypoints = get_gt_keypoints(gt_data, image_id)

                    # 计算TP, FP, FN
                    tp, fp, fn = calculate_metrics(keypoints, gt_keypoints, tp_distance)
                    # 可以在这里添加代码来绘制和保存关键点图像，如果需要的话
                    # 判断并返回有FN或FP的image_id
                    if fn > 0 or fp > 0:
                        fn_fp_image_ids.append(image_id)

                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

        # 计算总体的Precision, Recall, F1 Score
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 输出评价指标
        print(f"TP: {total_tp:.4f}, FP: {total_fp:.4f}, FN: {total_fn:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

        return total_tp, total_fp, total_fn, precision, recall, f1_score

        # # 将有FN和FP的图像ID保存到文件前，先按数字大小进行排序
        # fn_fp_image_ids_sorted = sorted(fn_fp_image_ids)
        # # 将有FN和FP的图像ID保存到文件
        # with open(os.path.join(output_dir, 'val_result.txt'), 'w') as f:
        #     f.write(f"TP: {total_tp:.4f}, FP: {total_fp:.4f}, FN: {total_fn:.4f}\n")
        #     f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")
        #     f.write("误检，漏检图像id如下:\n")
        #     for image_id in fn_fp_image_ids_sorted:
        #         f.write(f'{image_id}\n')



