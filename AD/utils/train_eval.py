import time
import torch
from utils.tools import AverageMeter, Transform


def train_epoch(cfg, epoch, data_loader, model, optimizer, logs, writer):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()
    loss = dict()
    for loss_name in cfg.loss:
        loss[loss_name] = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()

    end_time = time.time()
    for j, data_item in enumerate(data_loader):
        path, X, label = data_item
        X, label, batch_size = Transform(X, label)
        data_time.update(time.time() - end_time)
        optimizer.zero_grad()
        pred = model(X)
        loss_dict = model.loss(pred, label)
        total_loss = loss_dict['Total_Loss']
        total_loss.backward()
        total_losses.update(total_loss.item(), batch_size)
        for k in loss.keys():
            loss[k].update(loss_dict[k].item(), batch_size)
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        iter = (epoch - 1) * len(data_loader) + (j + 1)
        writer.add_scalar('train/batch/total_loss', total_losses.val, iter)
        for k in loss.keys():
            writer.add_scalar(f'train/batch/{k}', loss[k].val, iter)
        if logs:
            print("Epoch: [{0}][{1}/{2}]".format(epoch, j + 1, len(data_loader)), end=" | ")
            print("Time {batch_time.val:.3f} ({batch_time.avg:.3f})".format(batch_time=batch_time), end=" | ")
            print("Data {data_time.val:.3f} ({data_time.avg:.3f})".format(data_time=data_time), end=" | ")
            for k in loss.keys():
                print(f"{k} {loss[k].val:.4f} ({loss[k].avg:.4f})", end=" | ")
            print("Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})".format(total_loss=total_losses))

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))
    print("Train total loss: {:.4f}".format(total_losses.avg))
    for k in loss.keys():
        print("{}: {:.4f}".format(k, loss[k].avg))
    writer.add_scalar('train/epoch/total_loss', total_losses.avg, epoch)
    for k in loss.keys():
        writer.add_scalar(f'train/epoch/{k}', loss[k].avg, epoch)


def val_epoch(cfg, epoch, data_loader, model, writer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    loss = dict()
    for loss_name in cfg.loss:
        loss[loss_name] = AverageMeter()
    end_time = time.time()
    with torch.no_grad():
        for i, data_item in enumerate(data_loader):
            path, X, label = data_item
            X, label, batch_size = Transform(X, label)
            data_time.update(time.time() - end_time)
            pred = model(X)
            loss_dict = model.loss(pred, label)
            total_loss = loss_dict['Total_Loss']
            total_losses.update(total_loss.item(), batch_size)
            for k in loss.keys():
                loss[k].update(loss_dict[k].item(), batch_size)
            batch_time.update(time.time() - end_time)
            end_time = time.time()

        writer.add_scalar('val/epoch/total_loss', total_losses.avg, epoch)
        for k in loss.keys():
            writer.add_scalar(f'val/epoch/{k}', loss[k].avg, epoch)
            print("{}: {:.4f}".format(k, loss[k].avg))
        print("Val total loss: {:.4f}".format(total_losses.avg))
    return total_losses.avg
