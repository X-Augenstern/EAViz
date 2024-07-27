# import os
# import numpy as np
from torch import load
from AD.model.Loss import classLoss  # 引入损失的类函数
# from matplotlib import pyplot as plt
# from torch import optim
# from torch.utils.data import DataLoader
# from EAViz.AD.dataset.dataset import EEGDataset, loadDataPath
from AD.model.R2D import resnet34  # model 1
from AD.model.senet import SENet18  # model 2
from AD.model.densenet import DenseNet121  # model 3
from AD.model.vgg import VGG16  # model 4
from AD.model.googlenet import GoogLeNet

from AD.AE_Combine import AutoEncoder, SkipAutoEncoder, MemAutoEncoder, EstimatorAutoEncoder, VAE, Resnet_Encoder  # 导入AE模型


# class EarlyStopping:
#     def __init__(self, patience=3, verbose=True, delta=0, trace_func=print):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.trace_func = trace_func
#
#     def __call__(self, loss, model, path, epoch, optimizer):
#         score = -loss
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(loss, model, path, epoch, optimizer)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(loss, model, path, epoch, optimizer)
#             self.counter = 0
#
#     def save_checkpoint(self, loss, model, path, epoch, optimizer):
#         if self.verbose:
#             self.trace_func(
#                 f'Validation loss decreased ({self.val_loss_min:.6f} --> {loss:.6f}).  Saving model ...')
#         states = {
#             'epoch': epoch + 1,
#             'val_loss': loss,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#         }
#         torch.save(states, path)
#         self.val_loss_min = loss
#
#
# class AverageMeter(object):
#     def __init__(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# def leftBestModel(path):
#     for i in os.listdir(path):
#         if not i.endswith('.pth'):
#             os.rmdir(os.path.join(path, i))
#     weights = os.listdir(path)
#     if len(weights) == 1:
#         return
#     total = sorted(weights, key=lambda x: int(x.split('_')[-1][:-4]))
#     for i in total[:-1]:
#         os.remove(os.path.join(path, i))
#
#
# def adjust_learning_rate(cfg, optimizer, epoch):
#     lr = cfg.learning_rate
#     steps = np.sum(epoch > np.asarray(cfg.lr_decay_epochs))
#     if steps > 0:
#         lr = lr * (cfg.lr_decay_rate ** steps)
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def PrepareModel1(cfg, test=False):
    printout = 'Using Model: ' + cfg.model
    if cfg.model == 'Resnet34':
        model = resnet34(loss=classLoss())  # 修改损失
        print('***************************************** Using ResNet34 *****************************************')
    elif cfg.model == 'SENet18':
        model = SENet18(loss=classLoss())  # 修改损失
        print('***************************************** Using SENet18 *****************************************')
    elif cfg.model == 'DenseNet121':
        model = DenseNet121(loss=classLoss())
        print('***************************************** Using DenseNet121 *****************************************')
    elif cfg.model == "VGG16":
        model = VGG16(loss=classLoss())
        print('******************************************* Using VGG16 *******************************************')
    elif cfg.model == "GoogLeNet":
        model = GoogLeNet(loss=classLoss())
    else:
        raise NotImplementedError
    if test:
        state_dict = load(cfg.test_weight)['state_dict']
        model.load_state_dict(state_dict)
        print(' Weight_path: ' + cfg.test_weight)
    print(printout)
    return model.float().cuda()


def PrepareModel2(cfg, weight=None):  # AE 的模型
    assert cfg.model in ['AE', 'SkipAE', 'MemAE', 'EstimatorAutoEncoder', 'VAE', 'Resnet_Encoder']
    if cfg.model == 'AE':
        model = AutoEncoder(with_last_relu=cfg.with_last_relu)
        print('***************************************** Using AE *****************************************')
    elif cfg.model == 'SkipAE':
        model = SkipAutoEncoder(with_last_relu=cfg.with_last_relu)
        print('***************************************** Using SkipAE *****************************************')
    elif cfg.model == 'MemAE':
        model = MemAutoEncoder(with_last_relu=cfg.with_last_relu)
        print('***************************************** Using MemAE *****************************************')
    elif cfg.model == 'EstimatorAutoEncoder':
        model = EstimatorAutoEncoder(with_last_relu=cfg.with_last_relu)
    elif cfg.model == 'Resnet_Encoder':
        model = Resnet_Encoder(with_last_relu=cfg.with_last_relu)
    elif cfg.model == 'VAE':
        model = VAE(1000)
        print('***************************************** Using VAE *****************************************')
    else:
        raise NotImplementedError
    if weight:
        state_dict = load(weight)['state_dict']
        model.load_state_dict(state_dict)
        print(f'loading model {cfg.model} with weight {weight}')
    else:
        print(f'loading model {cfg.model}')
    return model.float().cuda()


#############################################################################

# def PrepareOptimizer(cfg, model):
#     parameters = model.parameters()
#     optimizer = optim.SGD(parameters,
#                           lr=cfg.learning_rate,
#                           momentum=cfg.momentum,
#                           weight_decay=cfg.wd)
#     return optimizer
#
#
# def PrepareDataset(cfg, train=True):
#     if train:
#         train_p, val_p = loadDataPath(cfg, True)
#         train_dataset = EEGDataset(cfg, train_p)
#         train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
#                                        shuffle=True, num_workers=cfg.num_workers)
#         val_dataset = EEGDataset(cfg, val_p)
#         val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
#                                      shuffle=False, num_workers=cfg.num_workers)
#     else:
#         test_p = loadDataPath(cfg, False)
#         test_dataset = EEGDataset(cfg, test_p)
#         test_data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
#                                       shuffle=False, num_workers=cfg.num_workers)
#     return (train_data_loader, val_data_loader) if train else test_data_loader
#
#
# def PrepareSaveFile(cfg):
#     i = 0
#     while True:
#         save_dir = cfg.save_root + str(i)
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#             break
#         i += 1
#     ckpt_path = os.path.join(save_dir, cfg.saveweight)
#     logp = os.path.join(save_dir, cfg.savelog)
#     if not os.path.exists(ckpt_path):
#         os.makedirs(ckpt_path)
#     if not os.path.exists(logp):
#         os.makedirs(logp)
#     return ckpt_path, logp
#
#
# def GetPerformance(y_pred, y_true):  # [X, B, 2], [X, B, 1]
#     TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))  # 10
#     FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))  # 0
#     FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))  # 6
#     TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))  # 34
#
#     # 根据上面得到的值计算A、P、R、F1
#     A = (TP + TN) / (TP + FP + FN + TN)  # y_pred与y_ture中同时为1或0
#     P = TP / (TP + FP)  # y_pred中为1的元素同时在y_true中也为1
#     R = TP / (TP + FN)  # y_true中为1的元素同时在y_pred中也为1
#     Fscore = 2 * P * R / (P + R)
#
#     print('acc , Pre , recall , F1      :', A, P, R, Fscore)
#     return A, P, R, Fscore
#     # print('acc', A)  # A=44/50
#     # print('pre', P)  # P=10/10
#     # print('recall', R)  # R=10/16
#     # print('F1', F1)
#
#
# def Transform(X, label):
#     batch_size = X.shape[0]
#     X = X.float().cuda()
#     label = label.float().cuda()
#     return X, label, batch_size
#
#
# def show_confusion_matrix(cm, path, show=True):
#     fig, ax = plt.subplots()
#     classes = ['A', 'B', 'C', 'D', 'E', 'F', 'AB', 'AC', 'AD', 'BD']
#     title = 'multi_lable classification'
#     im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]))
#     ax.set_ylabel('True label', fontsize=20)
#     ax.set_xlabel('Predicted label', fontsize=20)
#     ax.set_title(title, fontsize=20)
#     ax.set_xticklabels(classes, fontsize=20)
#     ax.set_yticklabels(classes, rotation=90, fontsize=20, va="center")
#
#     ax.set_ylim(1.5, -0.5)
#
#     fmt = 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black", fontsize=20)
#     fig.tight_layout()
#     plt.savefig(path, dpi=300)
#     if show:
#         plt.show()
#     plt.close()
