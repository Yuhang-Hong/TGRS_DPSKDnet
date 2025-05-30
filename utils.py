import torch 
from torch.nn import functional as F
from sklearn.metrics import cohen_kappa_score
from utils_HSI import metrics
import numpy as np
import os 

def cross_entropy(input, target, label_smooth=0, reduction='mean'):
    """Cross entropy loss.

    Args:
        input (torch.Tensor): logit matrix with shape of (batch, num_classes).
        target (torch.LongTensor): int label matrix.
        label_smooth (float, optional): label smoothing hyper-parameter.
            Default is 0.
        reduction (str, optional): how the losses for a mini-batch
            will be aggregated. Default is 'mean'.
    """
    num_classes = input.shape[1]
    log_prob = F.log_softmax(input, dim=1)
    zeros = torch.zeros(log_prob.size())
    target = zeros.scatter_(1, target.unsqueeze(1).data.cpu(), 1)
    target = target.type_as(input)
    target = (1-label_smooth) * target + label_smooth/num_classes
    loss = (-target * log_prob).sum(1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError
    
class KDLoss(torch.nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss


def soft_KD_loss(logit, label, total_classes, temp_factor):
    average_logits = []
    # print(label.shape)
    cal_kd_loss = KDLoss(temp_factor=temp_factor)

    # 遍历所有类别，计算每个类别的平均 logits
    for cur_cls in range(total_classes):
        # 筛选出当前类别的样本 logits
        cur_cls_logits = logit[label == cur_cls]
        
        # 计算当前类别的平均 logits
        cur_average_logit = cur_cls_logits.mean(0) if cur_cls_logits.size(0) > 0 else torch.zeros_like(logit[0])

        # 保存类别平均 logits
        average_logits.append(cur_average_logit)

    # 使用索引生成软标签
    average_logits = torch.stack(average_logits)
    
    cskd_label = average_logits[label].detach()

    # 计算知识蒸馏损失
    loss = cal_kd_loss(logit, cskd_label)
    
    return loss

def pair_KD_loss(logit1, logit2, label, total_classes, temp_factor):
    average_logits = []
    cal_kd_loss = KDLoss(temp_factor=temp_factor)

    # 遍历所有类别，计算每个类别的平均 logits
    for cur_cls in range(total_classes):
        # 筛选出当前类别的样本 logits
        cur_cls_logits1 = logit1[label == cur_cls]
        cur_cls_logits2 = logit2[label == cur_cls]
        
        # 分别计算两个 logit 的当前类别平均值
        cur_average_logit1 = cur_cls_logits1.mean(0) if cur_cls_logits1.size(0) > 0 else torch.zeros_like(logit1[0])
        cur_average_logit2 = cur_cls_logits2.mean(0) if cur_cls_logits2.size(0) > 0 else torch.zeros_like(logit2[0])

        # 计算两者平均，作为最终类别的平均 logits
        cur_average_logit = (cur_average_logit1 + cur_average_logit2) / 2

        # 保存类别平均 logits
        average_logits.append(cur_average_logit)

    # 使用索引生成软标签
    average_logits = torch.stack(average_logits)
    cskd_label = average_logits[label].detach()

    # 分别计算两个 logit 的知识蒸馏损失
    loss1 = cal_kd_loss(logit1, cskd_label)
    loss2 = cal_kd_loss(logit2, cskd_label)

    # 合并损失（这里取平均）
    total_loss = (loss1 + loss2) / 2

    return total_loss


def evaluate(net, val_loader, gpu, log_dir, tgt=False):
    ps = torch.tensor([], device=gpu)
    ys = torch.tensor([], device=gpu)
    with torch.no_grad():
        for x1, y1 in val_loader:
            x1 = x1.to(gpu, non_blocking=True)
            y1 = y1.to(gpu, non_blocking=True) - 1
            p1 = net(x1).argmax(dim=1)
            ps = torch.cat([ps, p1])
            ys = torch.cat([ys, y1])
            # 结果计算保持在GPU
        acc = (ps == ys).float().mean().item() * 100

    if tgt:
        ps = ps.cpu().numpy()
        ys = ys.cpu().numpy()
        # 获取评估指标
        
        n_classes = int(ys.max() + 1)

        results = metrics(ps, ys, n_classes=n_classes)
        confusion_matrix = results['Confusion_matrix']
        TPR = np.round(results['TPR'] * 100, 2)
        OA = results['Accuracy']

        # 计算 Kappa 系数并缩放到 0~100
        kappa = cohen_kappa_score(ys, ps) * 100

        # 打印结果
        print(confusion_matrix, '\n', 'TPR:', TPR, '\n', 'OA:', OA)
        print(f"Kappa Coefficient: {kappa:.2f}")

        # 构造文件保存路径
        results_path = os.path.join(log_dir, 'evaluation_results.txt')

        # 使用 'a' 模式打开文件（追加写入）
        with open(results_path, 'a') as f:
            f.write("\n==== Evaluation Results ====\n")

            # 保存混淆矩阵
            f.write("Confusion Matrix:\n")
            for row in confusion_matrix:
                f.write(" ".join(map(str, row)) + "\n")

            # 保存 TPR
            f.write("\nTPR (True Positive Rate):\n")
            f.write(" ".join(map(str, TPR)) + "\n")

            # 保存 Overall Accuracy (OA)
            f.write(f"\nOverall Accuracy (OA): {OA:.2f}%\n")

            # 保存缩放到 0-100 的 Kappa 系数
            f.write(f"Kappa Coefficient (scaled to 100): {kappa:.2f}\n")

            f.write("\n=============================\n")

        print(f"Results have been appended to {results_path}")

    return acc


def evaluate_tgt(cls_net, gpu, loader, modelpath, log_dir):
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    teacc = evaluate(cls_net, loader, gpu, log_dir, tgt=True)
    return teacc
