import logging
import os
import time
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import amp

from loss.supcontrast import SupConLoss
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def build_all_text_features(model, num_classes, batch, device, detach):
    features = []
    context = torch.no_grad() if detach else nullcontext()
    with context:
        i_ter = num_classes // batch
        left = num_classes - batch * (num_classes // batch)
        if left != 0:
            i_ter = i_ter + 1
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch).to(device)
            else:
                l_list = torch.arange(i * batch, num_classes).to(device)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True)
            features.append(text_feature.cpu() if detach else text_feature)
    text_features = torch.cat(features, 0)
    return text_features.cuda() if detach else text_features


def do_train_stage2_joint(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer_text,
             optimizer_image,
             optimizer_center,
             scheduler_text,
             scheduler_image,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start joint stage2 training')
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, dataset_name=cfg.DATASETS.NAMES)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    all_start_time = time.monotonic()
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler_text.step(epoch)
        scheduler_image.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer_text.zero_grad()
            optimizer_image.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            with amp.autocast(enabled=True):
                text_features = build_all_text_features(model, num_classes, batch, device, detach=False)
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view)
                logits = image_features @ text_features.t()
                image_loss = loss_fn(score, feat, target, target_cam, logits)
                text_features_for_target = model(label=target, get_text=True)
                image_features_for_text_loss = image_features.detach()
                text_loss_i2t = xent(image_features_for_text_loss, text_features_for_target, target, target)
                text_loss_t2i = xent(text_features_for_target, image_features_for_text_loss, target, target)
                text_loss = text_loss_i2t + text_loss_t2i
                loss = cfg.SOLVER.STAGE2.IMAGE_LOSS_WEIGHT * image_loss + cfg.SOLVER.STAGE2.TEXT_LOSS_WEIGHT * text_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer_text)
            scaler.step(optimizer_image)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.STAGE2.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Stage2 Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Image Loss: {:.3f}, Text Loss: {:.3f}, Acc: {:.3f}, Text Lr: {:.2e}, Image Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, image_loss.item(), text_loss.item(), acc_meter.avg, scheduler_text._get_lr(epoch)[0],
                                    scheduler_image.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info("Stage2 epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            save_name = cfg.MODEL.NAME + '_stage2_joint_{}.pth'.format(epoch)
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, save_name))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, save_name))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN and dist.get_rank() != 0:
                continue
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)
                    else:
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else:
                        target_view = None
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Stage2 Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage2 joint running time: {}".format(total_time))
