import torch
import torch.nn.functional as F

def structure_loss(pred, mask):
    """
    边界感知的加权分割损失。

    用31×31均值滤波估计局部背景，差值大的区域（即边界）
    获得更高权重（最大6倍），同时对BCE和IoU施加相同的
    空间权重以强化边界监督。
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')  # reduce -> reduction
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()