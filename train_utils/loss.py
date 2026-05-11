import torch
import matplotlib.pyplot as plt

class KpLoss(object):
    def __init__(self):
        # 实例化了一个均方损失误差适实例
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])

        ########## nrx test 绘制热图 ##########
        # batch_index = 0
        # channel_index = 0
        #
        # # 提取指定批次和通道的热图数据
        # heatmap = heatmaps.clone().cpu()[batch_index, channel_index, :, :]
        # # heatmap_numpy = heatmap.numpy()
        # # 绘制热图
        # # 清除当前的图表，以确保不在旧图上绘制
        # plt.clf()
        # cax = plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        # plt.colorbar(cax, fraction=0.046, pad=0.04)  # 显示颜色条
        # # heatmap_save_path = output_dir+"\\"+img_name+"heatmap.png"
        # # plt.savefig(heatmap_save_path, format='png', dpi=500)
        # plt.show()
        #####################################

        # [num_kps] -> [B, num_kps]
        # kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        # 乘 2 放大损失值
        loss = torch.sum(loss*2)/bs
        i=1
        # loss = torch.sum(loss * kps_weights) / bs
        return loss
