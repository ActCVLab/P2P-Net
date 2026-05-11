import json
import os
import datetime
import torch
from torch.utils import data
import nrx_irst_transforms
from model import P2P_HDNet
from nrx_dataset_IRST import IRST
from train_utils import train_eval_utils as utils
from torch.utils.tensorboard import SummaryWriter

def create_model(num_joints, load_pretrain_weights=False):

    model = P2P_HDNet(base_channel=32, num_joints=num_joints)

    # 载入预训练权重
    if load_pretrain_weights:
        # 载入预训练模型权重
        weights_dict = torch.load(r"N:\Scientist\csdn_yellowflower_deep-learning-master\csdn_yellowflower_deep-learning-master\pytorch_keypoint\HRNet\save_weights\model-209.pth", map_location='cpu')

        for k in list(weights_dict.keys()):
            # 如果载入的是imagenet权重，就删除无用权重，删除了全连接层的权重
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]

            # 如果载入的是coco权重，对比下num_joints，如果不相等就删除
            if "final_layer" in k:
                if weights_dict[k].shape[0] != num_joints:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys)

    return model


def main(args):

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ######################################## part-1 数据初始化部分 ########################################

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    nw = 0
    print('Using %g dataloader workers' % nw)

    # 获取图像高宽
    fixed_size = args.fixed_size
    # 热力图尺寸 以原图为基础的高宽上分别下降四倍
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)

    ######################################## part-2 数据集预处理部分 ########################################

    # 获取数据集根目录
    data_root = args.data_path
    # 注意这里的collate_fn是自定义的，因为读取的数据包括 image和 targets，不能直接使用默认的方法合成 batch
    batch_size = args.batch_size

    # 数据流
    data_transform = {
        # 训练集数据增强
        "train": nrx_irst_transforms.Compose([
            # 仿射变换操作 包括旋转、缩放、平移和倾斜等操作 ##########
            nrx_irst_transforms.AffineTransform(scale=(0.85, 1.15), rotation=(-180, 180),fixed_size=fixed_size),
            # 生成热力图
            nrx_irst_transforms.nrxKeypointToHeatMap_oneMapwithManyPoints(heatmap_hw=heatmap_hw, gaussian_sigma=2),
            # 数据转化为Tensor格式
            nrx_irst_transforms.ToTensor(),
            # 标准化处理
            nrx_irst_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # 验证集数据增强
        "test": nrx_irst_transforms.Compose([
            nrx_irst_transforms.AffineTransform(scale=(1, 1), fixed_size=fixed_size),
            # 添加热力图生成，用于计算验证集损失
            nrx_irst_transforms.nrxKeypointToHeatMap_oneMapwithManyPoints(heatmap_hw=heatmap_hw, gaussian_sigma=2),
            nrx_irst_transforms.ToTensor(),
            nrx_irst_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    train_dataset = IRST(data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)

    # 加载验证集数据
    val_dataset = IRST(data_root, "test", transforms=data_transform["test"], fixed_size=args.fixed_size)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    # 创建模型
    model = create_model(num_joints=args.num_joints)
    # 将模型移动至GPU
    model.to(device)

    # 模型结构可视化
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_logs'))


    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    # 使用优化器
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 自适应学习率调整
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                              verbose=True)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []

    # 添加验证集损失
    val_loss =[]

    learning_rate = []
    val_map = []

    calculate_epoch = 0

    yet_tg_num = 0

    # 初始化跟踪最高F1分数的变量
    best_f1_score = -1
    best_epoch = 0

    # 初始化记录最好成绩的指标
    # 给best_metrics一个初始值，确保即使所有F1分数相同时也不会出错
    best_metrics = {
        "epoch": 0,
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "val_mean_loss": float('inf')
    }



    # 每一个 epoch内容
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # 记录训练损失到TensorBoard
        writer.add_scalar('Loss/train', mean_loss.item(), epoch)
        writer.add_scalar('Learning Rate', lr, epoch)


        if epoch % args.eval_interval == 0:
            # 在验证集上评估模型
            val_mean_loss = utils.validate_one_epoch(model, val_data_loader, device, scaler)
            val_loss.append(val_mean_loss)

            # 根据验证集损失更新学习率
            lr_scheduler.step(val_mean_loss)

            # 记录验证损失到TensorBoard
            writer.add_scalar('Loss/val', val_mean_loss, epoch)

            total_tp, total_fp, total_fn, precision, recall, f1_score = utils.nrx_evaluate(model, val_data_loader, device, scaler, args.threshold, args.value_range,
                               args.max_num_targets, args.output_dir, args.data_path, args.tp_distance)

            writer.add_scalar('Val/TP', total_tp, epoch)
            writer.add_scalar('Val/FP', total_fp, epoch)
            writer.add_scalar('Val/FN', total_fn, epoch)
            writer.add_scalar('Val/Precision', precision, epoch)
            writer.add_scalar('Val/Recall', recall, epoch)
            writer.add_scalar('Val/F1', f1_score, epoch)

            # 用于表示计算的目标总数是否正确
            tg_num = total_tp + total_fn
            if tg_num != yet_tg_num:
                yet_tg_num = tg_num
                tg_flag = False
            else:
                tg_flag = True

            # 检查是否为最佳F1分数
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_epoch = epoch
                best_metrics.update({  # 使用update方法更新字典
                    "epoch": best_epoch,
                    "total_tp": total_tp,
                    "total_fp": total_fp,
                    "total_fn": total_fn,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "val_mean_loss": val_mean_loss
                })

            with open(os.path.join(args.output_dir, 'val_result.txt'), 'a') as log_file:
                log_file.write(f"Epoch: {epoch}\n")
                log_file.write(f"TP: {total_tp} FP: {total_fp} FN: {total_fn}\n")
                log_file.write(f"TG_NUM: {tg_num} TG_FLAG: {tg_flag}\n")
                log_file.write(f"Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1_score:.4f}\n")
                log_file.write(f"Validation Mean Loss: {val_mean_loss:.4f}\n")
                log_file.write("-------------------------------------------------\n")
                # 在每轮末位输出最佳Best F1 epoch
                log_file.write(f"Best F1 Score so far: {best_f1_score:.4f} at Epoch: {best_epoch}\n")
                log_file.write("=================================================\n")

                if epoch >= args.min_save_weight_epochs:
                    # save weights
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch}
                    if args.amp:
                        save_files["scaler"] = scaler.state_dict()
                    torch.save(save_files, f"./{args.output_dir}/model-{epoch}.pth")

        calculate_epoch += 1

    print(f"Best F1 Score so far: {best_f1_score:.4f} at Epoch: {best_epoch}")
    print(f"Metrics at best F1 Score: TP: {best_metrics['total_tp']}, FP: {best_metrics['total_fp']}, FN: {best_metrics['total_fn']}, "
        f"Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"successful save weights in {args.output_dir}! ")

    # 关闭SummaryWriter
    writer.close()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path',
                        default=r'N:\Project\MyGithub\P2P-HDNet\IRSTD_Datasets\IRSTD-1k',
                        help='dataset')
    # 输入图像尺寸
    parser.add_argument('--fixed-size', default=[640, 640], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=1, type=int, help='num_joints')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 最小保存权重
    parser.add_argument('--min-save-weight-epochs', default=0, type=int, help='min_save_weight_epochs')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的batch size
    parser.add_argument('--batch-size', default=5, type=int, metavar='N',
                        help='batch size when training.')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    # 验证集测试的间隔数
    parser.add_argument('--eval-interval', default=10, type=int, help='eval_interval')

    # 文件保存地址
    parser.add_argument('--output-dir', default='20241220_p2p_hdnet_save_weight_irstd1k_batch5_epoch2000_lr0.0001auto',
                        help='path where to save')
    # # 学习率
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    # 目标置信度阈值
    parser.add_argument('--threshold', default=0.2, type=float, help='threshold')
    # 目标选择范围
    parser.add_argument('--value-range', default=0.2, type=float, help='value_range')
    # 最大识别目标数
    parser.add_argument('--max-num-targets', default=8, type=int, help='max_num_targets')
    # 设置认定为TP的欧式距离
    parser.add_argument('--tp-distance', default=5, type=int, help='tp_distance')

    args = parser.parse_args()
    print(args)


    main(args)
