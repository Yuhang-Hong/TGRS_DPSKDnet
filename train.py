from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
from utils_HSI import sample_gt, metrics, seed_worker
from datasets import get_dataset, HyperX
import os
import time
import numpy as np
import pandas as pd
import argparse
from con_losses import SupConLoss
from network import discriminator
from network import generator
from datetime import datetime
from data_augmentation import fourier_transform
from utils import pair_KD_loss
from utils import evaluate, evaluate_tgt
# from data_augmentation import random_fourier_transform
# from data_augmentation.learning_fourier_transform import FourierTransformModule
# from data_augmentation.learning_total_fourier_transform import FourierTransformModule

parser = argparse.ArgumentParser(description='PyTorch SDEnet')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str,
                    default='dataset/Pavia/')

parser.add_argument('--source_name', type=str, default='paviaU',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='paviaC',
                    help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-3,
                         help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=256,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=233,
                    help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=5,
                    help='multiple of of data augmentation')
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--d_se', type=int, default=64)
parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)

parser.add_argument('--warmup', type=int, default=5)

parser.add_argument('--lr_scheduler', type=str, default='none')

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")
args = parser.parse_args()

def experiment():
    settings = locals().copy()
    print(settings)
    hyperparams = vars(args)
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name+'to'+args.target_name)
    log_dir = os.path.join(root, str(args.lr)+'_dim'+str(args.pro_dim) +
                           '_pt'+str(args.patch_size)+'_bs'+str(args.batch_size)+'_'+ time_str)
    
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)  # 写入tensorboard
    df = pd.DataFrame([args])
    df.to_csv(os.path.join(log_dir, 'params.txt'))

    seed_worker(args.seed)
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,           # 读取数据集
                                                                                        args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                        args.data_path)
    # 计算源数据集和目标数据集中非零标签（即有意义标签）的样本数
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    # 计算一个用于训练样本的比例参数 tmp。它结合了源和目标数据集中的样本数量、给定的采样比例参数 training_sample_ratio 和重采样比例 re_ratio，用来控制从源数据集中选择的样本数相对于目标数据集的比例。
    tmp = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    #  使用了 np.pad 函数对源图像、目标图像、源标签和目标标签进行了边缘填充。
    #  填充的目的是在提取图像补丁（patch）时避免边缘像素数据不足，确保所有位置都可以获取到完整的补丁数据。
    r = int(hyperparams['patch_size']/2)+1
    # r = int(15/2)+1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')  # 进行对称填充
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)),
                    'constant', constant_values=(0, 0))  # 边缘区域采用0填充
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)),
                    'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, _, _ = sample_gt(
        gt_src, args.training_sample_ratio, mode='random')  # 从源域中采样组成训练集和验证集
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')  # 从目标域中采样组成测试集
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:  # 训练数据不满足比例要求，进行扩充
        for i in range(args.re_ratio-1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()  # pytorch随机数生成对象
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    # train_dataset = HyperX_Gaussian_patch(
    #     img_src_con, train_gt_src_con, 13, **hyperparams_train)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True,
                                   num_workers=6,
                                   persistent_workers=True,
                                   prefetch_factor=4)
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 batch_size=hyperparams['batch_size'],
                                 num_workers=6,
                                 persistent_workers=True,
                                 prefetch_factor=4)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  batch_size=hyperparams['batch_size'],
                                  num_workers=6,
                                  persistent_workers=True,
                                  prefetch_factor=4)
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    D_net = discriminator.Discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes,
                                        patch_size=hyperparams['patch_size']).to(args.gpu)  # 定义生成器
    D_opt = optim.Adam(D_net.parameters(), lr=args.lr)
    G_net = generator.Generator(
        n=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10, device=args.gpu).to(args.gpu)
    G_opt = optim.Adam(G_net.parameters(), lr=args.lr)

    cls_criterion = nn.CrossEntropyLoss()

    con_criterion = SupConLoss(device=args.gpu)  # 自定义的对比损失

    best_acc = 0
    taracc, taracc_list = 0, []


    for epoch in range(1, args.num_epoch+1):

        t1 = time.time()
        loss_list = []
        D_net.train()  # 训练判别器

        for i, (x, y) in enumerate(train_loader):
            # x, y = x.to(args.gpu), y.to(args.gpu)
            x, y = x.to(args.gpu, non_blocking=True, memory_format=torch.channels_last), y.to(args.gpu, non_blocking=True)
            # y = y - 1
            y = y - 1  # 转换索引
            with torch.no_grad():
                x_ED = G_net(x)  # G目前是随机初始化，没参与训练
            rand = torch.nn.init.uniform_(torch.empty(len(x), 1, 1, 1)).to(
                args.gpu)  # Uniform distribution
            x_ID = rand*x + (1-rand)*x_ED  # 经过扩展域生成中间域

            x_tgt = G_net(x)  # 生成器的输出是图像块
            # x2_tgt = G_net(x)
            p_SD, z_SD = D_net(x, mode='train')  # 根据不同域，得到不同的结果
            p_ED, z_ED = D_net(x_ED, mode='train')
            p_ID, z_ID = D_net(x_ID, mode='train')
            x_fft = fourier_transform.FAM(x)
            x_fft_ID = fourier_transform.FAM(x_ID)
            x_fft_ED = fourier_transform.FAM(x_ED)
            
            p_fft_SD, _ = D_net(x_fft, mode='train')
            p_fft_ID, _ = D_net(x_fft_ID, mode='train')
            p_fft_ED, _ = D_net(x_fft_ED, mode='train')
            
            # ext_cls_loss = cls_criterion(p_fft_SD, y.long()) + cls_criterion(
            #     p_fft_ID, y.long()) + cls_criterion(p_fft_ED, y.long())
            
            pair_kd_loss = torch.tensor(0.0, device=args.gpu)  # 保证返回的是张量
            
            if epoch > args.warmup:

                pair_kd_loss = pair_KD_loss(p_SD, p_fft_SD, y.long(), num_classes, 4) + \
                pair_KD_loss(p_ID, p_fft_ID, y.long(), num_classes, 4) + \
                pair_KD_loss(p_ED, p_fft_ED, y.long(), num_classes, 4) 
                
            zsrc = torch.cat(
                [z_SD.unsqueeze(1), z_ED.unsqueeze(1), z_ID.unsqueeze(1)], dim=1)

            cls_loss = cls_criterion(
                p_SD, y.long()) + cls_criterion(p_ED, y.long()) + cls_criterion(p_ID, y.long()) + cls_criterion(p_fft_SD, y.long()) + cls_criterion(
                p_fft_ID, y.long()) + cls_criterion(p_fft_ED, y.long())

            p_tgt, z_tgt = D_net(x_tgt, mode='train')  # x_tgt是生成的伪目标域图像
            tgt_cls_loss = cls_criterion(p_tgt, y.long())

            zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1)
            con_loss = con_criterion(zall, y, adv=False)  # 执行监督对比损失
            
            loss = cls_loss + args.lambda_1*con_loss + tgt_cls_loss + args.lambda_2*pair_kd_loss 

            D_opt.zero_grad()
            loss.backward(retain_graph=True)

            # num_adv = y.unique().size()
            # zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1), z_ID.unsqueeze(1)], dim=1)
            zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(
                1).detach(), z_ID.unsqueeze(1).detach()], dim=1)
            
            con_loss_adv = 0
            idx_1 = np.random.randint(0, zsrc.size(1))

            for i, id in enumerate(y.unique()):
                mask = y == y.unique()[i]
                z_SD_i, zsrc_i = z_SD[mask], zsrc_con[mask]
                y_i = torch.cat([torch.zeros(z_SD_i.shape[0]),
                                torch.ones(z_SD_i.shape[0])])
                zall = torch.cat(
                    [z_SD_i.unsqueeze(1).detach(), zsrc_i[:, idx_1:idx_1+1]], dim=0)
                if y_i.size()[0] > 2:
                    con_loss_adv += con_criterion(zall, y_i)
            con_loss_adv = con_loss_adv/y.unique().shape[0]

            loss = tgt_cls_loss + con_loss_adv
            G_opt.zero_grad()
            # rand_fft_opt.zero_grad()
            
            loss.backward()
            D_opt.step()
            G_opt.step()
            # rand_fft_opt.step()

            # if args.lr_scheduler in ['cosine']:
            #     scheduler.step()

            loss_list.append(
                [cls_loss.item(), con_loss.item(), tgt_cls_loss.item(), con_loss_adv.item(), pair_kd_loss.item()])
        
        cls_loss, tgt_cls_loss, con_loss, con_loss_adv, pair_kd_loss = np.mean(# , pse_loss = np.mean(
            loss_list, 0)

        D_net.eval()
        teacc = evaluate(D_net, val_loader, args.gpu, log_dir)
        if best_acc < teacc:
            best_acc = teacc
            torch.save({'Discriminator': D_net.state_dict()},
                       os.path.join(log_dir, f'best.pkl'))
        t2 = time.time()

        print(f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2-t1:.2f}, cls {cls_loss:.4f} tgt_cls {tgt_cls_loss:.4f} con {con_loss:.4f} con_adv {con_loss_adv:.4f} pair_KD_loss {pair_kd_loss:.4f} /// val {len(val_loader.dataset)}, teacc {teacc:2.2f}')
        writer.add_scalar('cls_loss', cls_loss, epoch)
        writer.add_scalar('tgt_cls_loss', tgt_cls_loss, epoch)
        writer.add_scalar('con_loss', con_loss, epoch)
        writer.add_scalar('con_loss_adv', con_loss_adv, epoch)
        # writer.add_scalar('ext_cls_loss', ext_cls_loss, epoch)
        writer.add_scalar('fft_loss', pair_kd_loss, epoch)
        # writer.add_scalar('pse_loss', pse_loss, epoch)
        writer.add_scalar('teacc', teacc, epoch)

        if epoch % args.log_interval == 0: 
            
            pklpath = f'{log_dir}/best.pkl'
            taracc = evaluate_tgt(D_net, args.gpu, test_loader, pklpath, log_dir)
            taracc_list.append(round(taracc, 2))
            
            model_name = f"best_{taracc:.2f}.pth"  # 使用 .pth 后缀更规范
            torch.save(
                {"Discriminator": D_net.state_dict()},
                os.path.join(log_dir, model_name)
            )
            log_message = f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}'

            print(log_message)

            results_path = os.path.join(log_dir, 'best.txt')
            with open(results_path, "a") as f:
                f.write(log_message + "\n")

    writer.close()


if __name__ == '__main__':
    experiment()
