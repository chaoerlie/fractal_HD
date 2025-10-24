# -*- coding: utf-8 -*-
# file: generation_hd_filter_sequential.py

import os
import sys
import csv
import math
import glob
import shutil
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
            ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF", ".WEBP")

# ---------- 原始 box-counting（与你最初的方法一致） ----------
def _boxcount(binary: np.ndarray, k: int) -> int:
    ny, nx = binary.shape
    pad_y = (k - ny % k) % k
    pad_x = (k - nx % k) % k
    if pad_y or pad_x:
        binary = np.pad(binary, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=False)
    S = np.add.reduceat(
        np.add.reduceat(binary.view(np.uint8), np.arange(0, binary.shape[0], k), axis=0),
        np.arange(0, binary.shape[1], k), axis=1
    )
    return np.count_nonzero(S)

def box_counting_dimension_from_image(img: Image.Image,
                                      threshold: Optional[float] = None,
                                      box_sizes: Optional[List[int]] = None) -> float:
    """灰度->阈值(默认=均值)，尺度=2的幂，从大到小；线性回归斜率"""
    arr = np.array(img)
    if arr.ndim == 3:
        arr = 0.299*arr[..., 0] + 0.587*arr[..., 1] + 0.114*arr[..., 2]

    if arr.dtype == bool:
        binary = arr
    else:
        if threshold is None:
            threshold = float(arr.mean())
        binary = arr > threshold

    h, w = binary.shape
    max_k = min(h, w)
    if max_k < 2:
        raise ValueError("image too small")

    if box_sizes is None:
        max_pow = int(np.floor(np.log2(max_k)))
        ks = [2**p for p in range(max_pow, 1, -1)]  # >=4 时更稳，但保证>=2以取得两点
    else:
        ks = sorted([int(k) for k in box_sizes if 2 <= k <= max_k], reverse=True)

    Ns, used_ks = [], []
    for k in ks:
        n = _boxcount(binary, k)
        if n > 0:
            Ns.append(n); used_ks.append(k)

    if len(used_ks) < 2:
        raise ValueError("not enough valid scales")

    x = np.log(1.0/np.array(used_ks, dtype=float))
    y = np.log(np.array(Ns, dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)

# ---------- 工具 ----------
def list_flat_images(folder: str, recursive: bool = False) -> List[str]:
    paths = set()
    if recursive:
        base = os.path.join(folder, "**/*")
        for ext in IMG_EXTS:
            paths.update(glob.glob(base + ext, recursive=True))
    else:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and os.path.splitext(p)[1] in IMG_EXTS:
                paths.add(p)
    # 无扩展名但可能是图片（可选）
    return sorted(paths)

def read_class_thresholds(stats_csv_path: str, column: str = "median") -> Dict[str, float]:
    column = column.strip().lower()
    if column not in ("median", "mean"):
        raise ValueError("column must be 'median' or 'mean'")
    out: Dict[str, float] = {}
    with open(stats_csv_path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            cname = str(row["class"])
            out[cname] = float(row[column])
    return out

def chunk_by_n(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]

def _compute_one_hd(args) -> Tuple[str, float]:
    path, fixed_binarize, box_sizes = args
    try:
        img = Image.open(path)
        D = box_counting_dimension_from_image(img, threshold=fixed_binarize, box_sizes=box_sizes)
        return path, float(D)
    except Exception:
        return path, float("nan")

# ---------- 核心：按顺序分组，再按两种模式筛 ----------
def filter_generation_by_hd_sequential(
    generation_root: str,
    stats_csv_path: str,
    images_per_class: int,                 # 每类文件数 n（按顺序分块）
    mode: str = "threshold",               # "threshold" 或 "top"
    threshold_col: str = "median",         # 当 mode="threshold" 时用 CSV 的哪一列
    top_fraction: float = 0.5,             # 当 mode="top" 时保留比例
    threshold_override: Optional[float] = None,  # 全局阈值（忽略 CSV）
    class_name_format: str = "{:05d}",     # 生成类名格式，用于对齐 CSV（例：00000, 00001, ...）
    fixed_binarize: Optional[float] = None,# HD 二值化阈值（None=均值；如 128）
    box_sizes: Optional[List[int]] = None, # 盒尺度（像素）；None=自动
    sort_key: str = "name",                # "name" 或 "mtime"：按文件名或修改时间排序
    reverse: bool = False,                 # 是否反向排序
    workers: int = 0,                      # 并行进程数；0/1=单进程
    dry_run: bool = True,                  # True=演练；False=真执行
    move_to_backup: bool = False,          # True=移动到备份；False=删除
    backup_root: Optional[str] = None,     # 备份根目录；None= generation_root/_deleted_backup
    log_csv: str = "generation_filter_log.csv",
) -> int:
    generation_root = os.path.abspath(generation_root)
    if not os.path.isdir(generation_root):
        print(f"[fatal] generation_root is not a directory: {generation_root}")
        return 0

    # 列出平铺的图片并排序
    imgs = list_flat_images(generation_root, recursive=False)
    if not imgs:
        print(f"[fatal] no images found under: {generation_root}")
        return 0

    if sort_key == "mtime":
        imgs.sort(key=lambda p: os.path.getmtime(p), reverse=reverse)
    else:
        imgs.sort(key=lambda p: os.path.basename(p), reverse=reverse)

    total_imgs = len(imgs)
    print(f"[scan] root={generation_root} files_found={total_imgs} images_per_class={images_per_class}")

    # 按 n 切块
    groups = chunk_by_n(imgs, images_per_class)
    total_classes = len(groups)
    if len(groups[-1]) != images_per_class:
        print(f"[warn] last group has {len(groups[-1])} files (< {images_per_class}); will still process as a class.")

    # 阈值表
    thresholds: Dict[str, float] = {}
    if mode == "threshold" and threshold_override is None:
        thresholds = read_class_thresholds(stats_csv_path, column=threshold_col)

    # 备份
    if move_to_backup and not dry_run:
        backup_root = backup_root or os.path.join(generation_root, "_deleted_backup")
        os.makedirs(backup_root, exist_ok=True)

    # 日志
    log_rows: List[List[str]] = []
    def log(path: str, cls: str, hd: float, action: str, reason: str):
        log_rows.append([path, cls, f"{hd:.6f}" if np.isfinite(hd) else "nan", action, reason])

    # 逐类处理
    for cls_idx, paths in enumerate(groups):
        cls_name = class_name_format.format(cls_idx)
        print(f"[class] {cls_name} -> {len(paths)} files")

        # 计算 HD
        args_iter = [(p, fixed_binarize, box_sizes) for p in paths]
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_compute_one_hd, args_iter, chunksize=16))
        else:
            results = [_compute_one_hd(a) for a in args_iter]

        pairs = [(p, v) for (p, v) in results if (not math.isnan(v) and np.isfinite(v))]
        if not pairs:
            print(f"[warn] class={cls_name} all HD failed or empty.")
            continue

        if mode == "threshold":
            thr = threshold_override
            if thr is None:
                if cls_name not in thresholds:
                    # CSV 里没有该类名：保守，全部保留
                    for p, v in pairs: log(p, cls_name, v, "keep", "no-threshold-found")
                    continue
                thr = thresholds[cls_name]

            for p, v in pairs:
                if v >= thr:
                    log(p, cls_name, v, "keep", f"hd>=thr({thr:.6f})")
                else:
                    log(p, cls_name, v, "delete", f"hd<thr({thr:.6f})")
                    if not dry_run:
                        try:
                            if move_to_backup:
                                dst_dir = os.path.join(backup_root, cls_name)
                                os.makedirs(dst_dir, exist_ok=True)
                                shutil.move(p, os.path.join(dst_dir, os.path.basename(p)))
                            else:
                                os.remove(p)
                        except Exception as e:
                            log(p, cls_name, v, "error", f"delete/move-failed:{e}")

        elif mode == "top":
            frac = float(top_fraction)
            frac = min(max(frac, 0.0), 1.0)
            k = max(1, int(math.ceil(len(pairs) * frac)))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
            keep_set = set(p for p, _ in pairs_sorted[:k])

            for p, v in pairs:
                if p in keep_set:
                    log(p, cls_name, v, "keep", f"top{int(frac*100)}%")
                else:
                    log(p, cls_name, v, "delete", f"bottom{int((1-frac)*100)}%")
                    if not dry_run:
                        try:
                            if move_to_backup:
                                dst_dir = os.path.join(backup_root, cls_name)
                                os.makedirs(dst_dir, exist_ok=True)
                                shutil.move(p, os.path.join(dst_dir, os.path.basename(p)))
                            else:
                                os.remove(p)
                        except Exception as e:
                            log(p, cls_name, v, "error", f"delete/move-failed:{e}")
        else:
            raise ValueError("mode must be 'threshold' or 'top'")

    # 写日志
    with open(log_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "hd", "action", "reason"])
        w.writerows(log_rows)

    print(f"[summary] classes={total_classes} files={total_imgs} decisions={len(log_rows)}")
    return len(log_rows)

# ---------- CLI ----------
def main_cli():
    """
    使用示例：

    # 1) Top-50%（平铺按顺序分组，每类50张）
    python generation_hd_filter_sequential.py "D:\\gen" "D:\\val_stats.csv" top 0.5 --n 50 --dry

    # 2) 阈值法：与 CSV 的 median 对齐（类名按 00000,00001... 生成）
    python generation_hd_filter_sequential.py "D:\\gen" "D:\\val_stats.csv" threshold median --n 50 --apply

    # 3) 阈值法：使用全局阈值（忽略 CSV）
    python generation_hd_filter_sequential.py "D:\\gen" "D:\\val_stats.csv" threshold median --n 50 --thr_override 1.45 --apply

    常用复刻原始 HD 的参数：
      --fixed_bin 128 --sizes 256 128 64 32 16 8 4
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_root", help="平铺图片所在目录")
    ap.add_argument("stats_csv", help="基准CSV（含列 class,median,mean），threshold模式需要；top模式可随便给")
    ap.add_argument("mode", choices=["threshold", "top"], help="筛选模式")
    ap.add_argument("arg", help="threshold模式: median/mean；top模式: 比例(0~1)，如 0.5")

    ap.add_argument("--n", type=int, default=50, help="每类图片数量（按顺序切块）")
    ap.add_argument("--class_name_format", default="{:05d}",
                    help="生成类名格式以对齐CSV（默认 5 位零填充），例如 '{:05d}' 或 'class_{:04d}'")

    ap.add_argument("--thr_override", type=float, default=None, help="统一阈值（忽略CSV）")
    ap.add_argument("--fixed_bin", type=float, default=None, help="HD二值化阈值；None=均值，如 128")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="盒尺度（像素），如 256 128 64 32 16 8 4")

    ap.add_argument("--sort_key", choices=["name", "mtime"], default="name", help="按文件名或修改时间排序")
    ap.add_argument("--reverse", action="store_true", help="反向排序")

    ap.add_argument("--workers", type=int, default=0, help="并行进程数；0/1=单进程")
    ap.add_argument("--apply", action="store_true", help="执行删除/移动；默认演练")
    ap.add_argument("--move", action="store_true", help="不删除而是移动到备份")
    ap.add_argument("--backup_root", default=None, help="备份根目录（默认 generation/_deleted_backup）")
    ap.add_argument("--log_csv", default="generation_filter_log.csv", help="日志CSV文件名")

    args = ap.parse_args()

    dry_run = True
    if args.apply:
        dry_run = False

    mode = args.mode
    if mode == "threshold":
        threshold_col = args.arg.strip().lower()
        if threshold_col not in ("median", "mean"):
            print("ERROR: threshold mode requires arg to be 'median' or 'mean'")
            sys.exit(2)
        top_fraction = 0.5
    else:
        threshold_col = "median"
        try:
            top_fraction = float(args.arg)
        except Exception:
            print("ERROR: top mode requires a float in (0,1], e.g., 0.5")
            sys.exit(2)

    decisions = filter_generation_by_hd_sequential(
        generation_root=args.generation_root,
        stats_csv_path=args.stats_csv,
        images_per_class=args.n,
        mode=mode,
        threshold_col=threshold_col,
        top_fraction=top_fraction,
        threshold_override=args.thr_override,
        class_name_format=args.class_name_format,
        fixed_binarize=args.fixed_bin,
        box_sizes=args.sizes,
        sort_key=args.sort_key,
        reverse=args.reverse,
        workers=args.workers,
        dry_run=dry_run,
        move_to_backup=args.move,
        backup_root=args.backup_root,
        log_csv=args.log_csv,
    )
    print(f"Done. Logged {decisions} decisions to {args.log_csv}")

if __name__ == "__main__":
    main_cli()
