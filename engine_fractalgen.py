import math
import sys
import os
import time
import shutil
from typing import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import torch.nn.functional as F
from resnet.train import MultiScaleResNet152


# --- Hausdorff Dimension Calculation Functions ---

def box_count(image, box_size):
    """
    Counts the number of boxes of a given size needed to cover the non-zero pixels of a binary image.
    """
    # Threshold the image to make it binary (0 or 255)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Image dimensions
    height, width = binary_image.shape

    # Initialize box count
    count = 0

    # Slide a box of size (box_size, box_size) across the image
    for y in range(0, height, box_size):
        for x in range(0, width, box_size):
            # Extract a box from the image
            box = binary_image[y:y + box_size, x:x + box_size]
            if np.any(box):  # If any pixel in the box is non-zero, count this box
                count += 1

    return count

def hausdorff_dimension(image, max_box_size=64, step=2):
    """
    Calculates the Hausdorff dimension of an image using the box-counting method.
    """
    sizes = []
    counts = []

    # Iterate through different box sizes
    for box_size in range(1, max_box_size + 1, step):
        if box_size > 0:
            count = box_count(image, box_size)
            if count > 0:
                sizes.append(box_size)
                counts.append(count)

    if not sizes or not counts:
        return 0.0

    # Fit a line in log-log space
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)

    # Perform linear regression (fit a line)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)

    # Hausdorff dimension is the negative of the slope
    hd = -slope
    return hd


def train_one_epoch(model, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None,
                    hd_model=None, standard_hd_values=None, hd_weight_list=None, mmds=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()
    loss = None
    # 预尝试一次性加载 hd_model（如果传入的是路径或 args 提供路径）

    hd_model_loaded = None
    if isinstance(hd_model, (str, Path)):
        hd_model_loaded = load_hd_model(hd_model, device)
    elif isinstance(hd_model, torch.nn.Module):
        hd_model_loaded = hd_model.to(device)
        hd_model_loaded.eval()
    elif getattr(args, "hd_model", None):
        hd_model_loaded = load_hd_model(args.hd_model, device)


    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    print('dataloader len: {}', len(data_loader))
    # 在 train_one_epoch 开头（for 循环之前）初始化缓存
    sampled_images_cache = None
    sampled_cache_step = -1
    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Main forward and optional HD loss integration (per-batch) ---
        with torch.cuda.amp.autocast():
            loss = model(samples, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # --- compute HD loss for this batch (keep graph if hd_model exists) ---
        hd_loss = None
        hd_loss_value = 0.0
        model_without_ddp = model.module if hasattr(model, "module") else model

        # decide whether to re-sample this batch (减少频率 & 样本数)
        hd_every = getattr(args, "hd_every", 40)            # 默认每20个batch计算一次HD（可通过args修改）
        hd_sample_n = getattr(args, "hd_sample_n", 8)       # 每次只用前 hd_sample_n 张图计算HD
        reuse_cache = getattr(args, "hd_reuse_cache", True) # 是否重用缓存结果直到下一次采样

        do_sample = (data_iter_step % hd_every == 0) or (sampled_images_cache is None)
        if do_sample:
            # 生成较小数量的样本用于 HD（可通过 args 控制 num_conds/num_iter），
            # 若 model.sample 只能按 batch-size 返回，可只取前 hd_sample_n
            with torch.cuda.amp.autocast():
                class_embedding = model_without_ddp.class_emb(labels)
                sampled_images_full = model_without_ddp.sample(
                    cond_list=[class_embedding for _ in range(args.num_conds)],
                    num_iter_list=[int(num_iter) for num_iter in args.num_iter_list.split(",")],
                    cfg=1.0, cfg_schedule='constant',
                    temperature=args.temperature,
                    filter_threshold=args.filter_threshold,
                    fractal_level=0
                )
            # denormalize but keep float and computation graph
            pix_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
            pix_std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
            sampled_images_full = (sampled_images_full * pix_std + pix_mean) * 255.0
            sampled_images_full = torch.clamp(sampled_images_full, 0.0, 255.0).float()

            # cache (只保留前 hd_sample_n 用于 HD loss)
            sampled_images_cache = sampled_images_full[:hd_sample_n].detach() if reuse_cache else sampled_images_full[:hd_sample_n]
            sampled_cache_step = data_iter_step
        else:
            # 复用缓存；如果缓存不存在则跳过 HD 计算
            if sampled_images_cache is None:
                # 跳过 HD 本次计算以避免阻塞
                hd_loss = None
                hd_loss_value = 0.0
                sampled_images_cache = None

        # 如果有缓存或刚生成的样本，则使用缓存的小批量计算 HD（可微路径同前）
        if sampled_images_cache is not None:
            # 将 cached 图片恢复到需要的设备/类型并保持或剥离梯度（取决是否可微）
            imgs_for_hd = sampled_images_cache.to(device)
            # resolve hd_model for this run (use preloaded if available)
            hd_model_used = hd_model_loaded
            # 只使用在函数入口预先加载的 hd_model_loaded（不要在循环中再次 load）
            hd_model_used = hd_model_loaded

            if hd_model_used is not None:
                # freeze hd_model params
                hd_model_used.to(device)
                hd_model_used.eval()
                for p in hd_model_used.parameters():
                    p.requires_grad = False

                # 原：把图像转为 [0,1] 后又做了 ImageNet mean/std，导致与 resnet/train 中 ToTensor() 不一致
                # imgs = imgs_for_hd.float() / 255.0
                # imgs_norm = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
                # mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, -1, 1, 1)
                # std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, -1, 1, 1)
                # imgs_norm = (imgs_norm - mean) / std
                # 修正：resnet/train.py 使用 Resize + ToTensor() -> 输入应为 [0,1] 的张量 (B,C,224,224)
                imgs = imgs_for_hd.float() / 255.0                # 现在 imgs 在 [0,1]
                imgs_norm = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
                # debug（可选）：打印输入统计，便于对齐训练时预处理
                # print("HD-input min/max/mean:", imgs_norm.min().item(), imgs_norm.max().item(), imgs_norm.mean().item())

                 # 若希望反向传播 HD loss，请不要用 no_grad()
                with torch.cuda.amp.autocast():
                    preds = hd_model_used(imgs_norm)
                preds = preds.view(-1)
                print(f"HD preds: {preds.detach().cpu().numpy()}")
                target_hd = torch.full_like(preds, float(standard_hd_values))
                hd_loss = torch.nn.functional.l1_loss(preds, target_hd)
                hd_loss_value = float(hd_loss.detach().cpu().item())
                print(f"HD loss: {hd_loss_value:.4f}")
            else:
                # fallback box-count on cached images (在CPU上异步处理效果更好)
                numpy_images = imgs_for_hd.detach().cpu().numpy().transpose(0, 2, 3, 1)
                predicted_hd_list = []
                for img_np in numpy_images:
                    gray_img = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    hd_val = hausdorff_dimension(gray_img, max_box_size=max(2, args.img_size // 2))
                    predicted_hd_list.append(hd_val)
                hd_loss_value = float(np.mean(predicted_hd_list))
                hd_loss = None

        # compute hd_weight (mmds优先)
        # 不在每个 batch 更新 MMDS（每 epoch 更新一次），这里只读取当前 lambda 值供当次合并损失使用
        if mmds is not None:
            hd_weight = mmds.get_lambda()
        else:
            hd_weight = hd_weight_list[epoch]
        hd_weight = max(0.0, float(hd_weight))
        print(f"HD weight: {hd_weight:.4f}")
        

        # combine losses: only add hd_loss if it's differentiable (hd_loss is not None)
        if hd_loss is not None and hd_weight > 0.0:
            total_loss = loss + hd_weight * hd_loss
        else:
            total_loss = loss

        # Backpropagate total loss
        loss_scaler(total_loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(hd_loss=hd_loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('hd_loss', hd_loss_value, epoch_1000x)

    # # --- HD Loss Calculation (once per epoch) ---
    # hd_loss_value = 0.0
    # if standard_hd_values is not None and hd_weight_list is not None and epoch < len(hd_weight_list):
    #     print(f"--- Calculating HD Loss for epoch {epoch} ---")
    #     model_without_ddp = model.module
    #     # Generate images using the labels from the last batch
    #     with torch.no_grad():
    #         with torch.cuda.amp.autocast():
    #             # 1. 生成图像
    #             class_embedding = model_without_ddp.class_emb(labels)
    #             sampled_images = model_without_ddp.sample(
    #                 cond_list=[class_embedding for _ in range(args.num_conds)],
    #                 num_iter_list=[int(num_iter) for num_iter in args.num_iter_list.split(",")],
    #                 cfg=1.0,  # 在训练中通常不使用CFG
    #                 cfg_schedule='constant',
    #                 temperature=args.temperature,
    #                 filter_threshold=args.filter_threshold,
    #                 fractal_level=0
    #             )

    #             # 2. 图像预处理 (denormalize)
    #             pix_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
    #             pix_std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
    #             # 反归一化到 [0, 255]
    #             sampled_images = (sampled_images * pix_std + pix_mean) * 255
    #             sampled_images = torch.clamp(sampled_images, 0, 255).byte()

    #     # 3. 计算HD：优先使用传入或 args 指定的回归模型，否则使用盒计数
    #     predicted_hd_list = []

    #     # 如果外部没有传入已加载的 hd_model 且 args 指定了路径，则尝试加载
    #     hd_model_used =None
    #     if hd_model_used is None and getattr(args, "hd_model_path", None):
    #         print('Using Resnet')
    #         hd_model_used = load_hd_model(args.hd_model_path, device)

    #     if hd_model_used is not None:
    #         # 使用回归网络预测 HD
    #         # 准备输入：float [0,1] -> resize 224 -> ImageNet norm
    #         print('Using Resnet')
    #         imgs = sampled_images.float() / 255.0  # B, C, H, W
    #         imgs = imgs.to(device)
    #         imgs_resized = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
    #         mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, -1, 1, 1)
    #         std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, -1, 1, 1)
    #         imgs_norm = (imgs_resized - mean) / std
    #         with torch.no_grad():
    #             preds = hd_model_used(imgs_norm)  # 期望输出形状 [B] 或 [B,1]
    #         preds = preds.view(-1).detach().cpu().numpy()
    #         predicted_hd_list = [float(x) for x in preds]
    #     else:
    #         # fallback: 原 box-counting 实现（逐张处理）
    #         numpy_images = sampled_images.cpu().numpy().transpose(0, 2, 3, 1) # B, H, W, C
    #         for img_np in numpy_images:
    #             gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    #             hd_val = hausdorff_dimension(gray_img, max_box_size=max(2, args.img_size // 2))
    #             predicted_hd_list.append(hd_val)
        
    #     predicted_hd = torch.tensor(predicted_hd_list, dtype=torch.float32).to(device)

    #     # 4. 获取标准HD值
    #     target_hd = torch.full_like(predicted_hd, standard_hd_values)

    #     # 5. 计算hd_loss (L1 Loss)
    #     hd_loss = torch.nn.functional.l1_loss(predicted_hd, target_hd)
    #     hd_loss_value = hd_loss.item()

    #     # 6. 根据epoch计算权重


    #     if mmds is not None:
    #         print('mmds')
    #         # 尝试从 metric_logger 获取当前训练损失做为 L_val；fallback 使用 hd_loss
    #         try:
    #             L_val = metric_logger.meters['loss'].global_avg
    #         except Exception:
    #             L_val = hd_loss_value
    #         mmds.update_lambda(L_val)
    #         hd_weight = mmds.get_lambda()
    #     else:
    #         hd_weight = hd_weight_list[epoch]
    #     # 保证非负
    #     hd_weight = max(0.0, float(hd_weight))
    #     print(f'HD Loss: {hd_loss_value:.4f}, Weight: {hd_weight:.6f}')



    #     # hd_weight = hd_weight_list[epoch]
    #     # print(f'HD Loss: {hd_loss_value:.4f}, Weight: {hd_weight:.4f}')

    #     # 7. Backpropagate HD loss
    #     if hd_weight > 0:
    #         loss_scaler(loss + hd_weight * hd_loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
    #         optimizer.zero_grad()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_nll(model: torch.nn.Module, data_loader: Iterable, device: torch.device, N: int):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = ''
    print_freq = 20

    total_samples = 0
    total_bpd = 0.0

    for _, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss = 0.0
        # Average multiple forward passes for a stable NLL estimate.
        for _ in range(N):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    one_loss = model(samples, labels)
                    loss += one_loss
        loss /= N
        loss_value = loss.item()

        # convert loss to bits/dim
        bpd_value = loss_value / math.log(2)
        total_samples += samples.size(0)
        total_bpd += bpd_value * samples.size(0)

        torch.cuda.synchronize()
        metric_logger.update(bpd=bpd_value)

    print("BPD: {:.5f}".format(total_bpd / total_samples))


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # # Construct the folder name for saving generated images.
    # save_folder = os.path.join(
    #     args.output_dir,
    #     "ariter{}-temp{}-{}cfg{}-filter{}-image{}".format(
    #         args.num_iter_list, args.temperature, args.cfg_schedule,
    #         args.cfg, args.filter_threshold, args.num_images
    #     )
    # )

    # if args.evaluate_gen:
    #     save_folder += "_evaluate"
    # print("Save to:", save_folder)
    # if misc.get_rank() == 0 and not os.path.exists(save_folder):
    #     os.makedirs(save_folder)

    import datetime

    # 时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 基础名称
    base_folder_name = "ariter{}-temp{}-{}cfg{}-filter{}-image{}".format(
        args.num_iter_list, args.temperature, args.cfg_schedule,
        args.cfg, args.filter_threshold, args.num_images
    )

    # 加时间戳避免覆盖
    save_folder = os.path.join(args.output_dir, base_folder_name + "_" + timestamp)

    # 若 evaluate_gen 模式附加标记
    if args.evaluate_gen:
        save_folder += "_evaluate"

    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)


    # Ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    used_time = 0.0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                class_embedding = model_without_ddp.class_emb(labels_gen)
                if not args.cfg == 1.0:
                    # Concatenate fake latent for classifier-free guidance.
                    class_embedding = torch.cat(
                        [class_embedding, model_without_ddp.fake_latent.repeat(batch_size, 1)],
                        dim=0
                    )
                sampled_images = model_without_ddp.sample(
                    cond_list=[class_embedding for _ in range(args.num_conds)],
                    num_iter_list=[int(num_iter) for num_iter in args.num_iter_list.split(",")],
                    cfg=args.cfg, cfg_schedule=args.cfg_schedule,
                    temperature=args.temperature,
                    filter_threshold=args.filter_threshold,
                    fractal_level=0
                )

        # Measure generation speed (skip first batch).
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        if i >= 1:
            used_time += batch_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()

        # Denormalize images.
        pix_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1, -1, 1, 1)
        pix_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1, -1, 1, 1)
        sampled_images = sampled_images * pix_std + pix_mean
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    time.sleep(10)


    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 64:
            fid_statistics_file = 'fid_stats/adm_in64_stats.npz'
        elif args.img_size == 256:
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}".format(args.cfg)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        # if not args.evaluate_gen:
        #     # remove temporal saving folder for online eval
        #     shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)

def load_hd_model(hd_model_path, device):
    """
    加载 resnet/train.py 中的 MultiScaleResNet152 回归模型。
    hd_model_path 可以是 None、模型对象 或 checkpoint 路径（.pth）。
    返回：已加载并设为 eval() 的模型，或 None（加载失败或路径为空）。
    """
    if not hd_model_path:
        return None
    # 如果已经是模型实例，直接返回
    if isinstance(hd_model_path, torch.nn.Module):
        mdl = hd_model_path
        mdl.to(device)
        mdl.eval()
        return mdl
    try:
        model = MultiScaleResNet152()
        ckpt = torch.load(hd_model_path, map_location=device)
        # 支持 {'state_dict': ...} 或直接 state_dict
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # 尝试移除 module. 前缀
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state)
        model.to(device)
        model.eval()
        print(f"Loaded HD model from {hd_model_path}")
        return model
    except Exception as e:
        print(f"Failed to load HD model from {hd_model_path}: {e}")
        return None