# file: hd_batch_folder_stats.py
import os
import csv
import math
import sys
import numpy as np
from PIL import Image, ImageFilter
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------- 盒维数（稳健版） --------------------
def _otsu_threshold(arr):
    a = arr.astype(np.float64)
    a -= a.min()
    rng = a.max() if a.max() > 0 else 1.0
    a = (a / rng * 255.0).astype(np.uint8)
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = (omega * (1 - omega))
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    t = int(np.nanargmax(sigma_b2))
    return (t / 255.0) * rng + arr.min()

def _boxcount_multi_offset(binary, k, n_offsets=4):
    ny, nx = binary.shape
    counts = []
    B = binary.view(np.uint8)
    shifts_y = np.linspace(0, k-1, min(n_offsets, k), dtype=int)
    shifts_x = np.linspace(0, k-1, min(n_offsets, k), dtype=int)
    for sy in shifts_y:
        for sx in shifts_x:
            y0 = (-ny - sy) % k
            x0 = (-nx - sx) % k
            M = np.pad(B, ((sy, y0), (sx, x0)), mode="constant", constant_values=0)
            S = np.add.reduceat(
                np.add.reduceat(M, np.arange(0, M.shape[0], k), axis=0),
                np.arange(0, M.shape[1], k), axis=1
            )
            counts.append(np.count_nonzero(S))
    return float(np.mean(counts))

def box_counting_dimension_from_image_robust(
    img,
    use_edges=False,
    gaussian_sigma=1.0,
    threshold="otsu",
    min_boxes=20,
    max_occupancy=0.95,
    min_occupancy=0.01,
    n_offsets=4,
    box_sizes=None,
):
    # 读/转灰度并模糊
    if isinstance(img, Image.Image):
        gimg = img.convert("L")
        if gaussian_sigma and gaussian_sigma > 0:
            gimg = gimg.filter(ImageFilter.GaussianBlur(radius=float(gaussian_sigma)))
        arr = np.array(gimg, dtype=np.float64)
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])

    # 二值化（可选边缘）
    if threshold == "otsu":
        th = _otsu_threshold(arr)
        base_bin = arr > th
    elif threshold is None:
        base_bin = arr > arr.mean()
    else:
        base_bin = arr > float(threshold)

    if use_edges:
        up = np.zeros_like(base_bin); up[1:] = base_bin[:-1]
        dn = np.zeros_like(base_bin); dn[:-1] = base_bin[1:]
        lf = np.zeros_like(base_bin); lf[:,1:] = base_bin[:,:-1]
        rt = np.zeros_like(base_bin); rt[:,:-1] = base_bin[:,1:]
        binary = (base_bin != up) | (base_bin != dn) | (base_bin != lf) | (base_bin != rt)
    else:
        binary = base_bin

    h, w = binary.shape
    max_k = min(h, w)
    if max_k < 4:
        raise ValueError("image too small")

    # 几何序列尺度
    if box_sizes is None:
        ks = []
        k = 2**int(np.floor(np.log2(max_k//2 if max_k>=4 else max_k)))
        while k >= 2:
            ks.append(k); k //= 2
    else:
        ks = sorted([int(k) for k in box_sizes if 2 <= k <= max_k], reverse=True)

    Ns, used_ks = [], []
    total = binary.size
    for k in ks:
        n = _boxcount_multi_offset(binary, k, n_offsets=n_offsets)
        occ = (n * (k*k)) / total
        if n >= min_boxes and (min_occupancy <= occ <= max_occupancy):
            Ns.append(n); used_ks.append(k)

    if len(used_ks) < 3:
        Ns, used_ks = [], []
        tmp = ks[1:-1] if len(ks) > 2 else ks
        for k in tmp:
            n = _boxcount_multi_offset(binary, k, n_offsets=n_offsets)
            if n >= min_boxes:
                Ns.append(n); used_ks.append(k)

    if len(used_ks) < 2:
        raise ValueError("not enough valid scales")

    x_all = np.log(1.0/np.array(used_ks, dtype=float))
    y_all = np.log(np.array(Ns, dtype=float))
    idx = np.argsort(x_all)
    x_sorted, y_sorted = x_all[idx], y_all[idx]
    n = len(x_sorted)
    lo = int(np.floor(0.15*n))
    hi = int(np.ceil(0.85*n))
    x, y = (x_sorted[lo:hi], y_sorted[lo:hi]) if (hi-lo)>=2 else (x_all, y_all)

    slope, intercept = np.polyfit(x, y, 1)
    return float(slope)

# -------------------- 单图与子文件夹处理 --------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def compute_image_hd(path, params):
    try:
        img = Image.open(path)
        D = box_counting_dimension_from_image_robust(
            img,
            use_edges=params["use_edges"],
            gaussian_sigma=params["gaussian_sigma"],
            threshold=params["threshold"],
            min_boxes=params["min_boxes"],
            max_occupancy=params["max_occupancy"],
            min_occupancy=params["min_occupancy"],
            n_offsets=params["n_offsets"],
            box_sizes=None
        )
        return D
    except Exception as e:
        # 返回 NaN 以便后续过滤
        return float("nan")

def process_class_folder(class_dir, params, workers=0):
    """返回 (class_name, median, mean)；若无有效图片则返回 None"""
    class_name = os.path.basename(class_dir.rstrip("\\/"))
    img_paths = [os.path.join(class_dir, f)
                 for f in os.listdir(class_dir)
                 if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(IMG_EXTS)]
    if not img_paths:
        return None

    if workers and workers > 1:
        vals = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(compute_image_hd, p, params): p for p in img_paths}
            for _ in as_completed(futs):
                vals.append(_.result())
    else:
        vals = [compute_image_hd(p, params) for p in img_paths]

    vals = np.array([v for v in vals if (not math.isnan(v) and np.isfinite(v))], dtype=float)
    if vals.size == 0:
        return None
    median = float(np.median(vals))
    mean = float(np.mean(vals))
    return class_name, median, mean

# -------------------- 主流程 --------------------
def main(
    root_val_folder,
    output_csv="hd_stats.csv",
    use_edges=True,
    gaussian_sigma=1.0,
    threshold="otsu",
    n_offsets=6,
    min_boxes=30,
    max_occupancy=0.90,
    min_occupancy=0.01,
    workers=0
):
    params = dict(
        use_edges=use_edges,
        gaussian_sigma=gaussian_sigma,
        threshold=threshold,
        n_offsets=n_offsets,
        min_boxes=min_boxes,
        max_occupancy=max_occupancy,
        min_occupancy=min_occupancy,
    )

    # 找到一级子文件夹（类）
    class_dirs = [os.path.join(root_val_folder, d)
                  for d in os.listdir(root_val_folder)
                  if os.path.isdir(os.path.join(root_val_folder, d))]
    class_dirs.sort()

    rows = []
    total = len(class_dirs)
    print(f"Found {total} class folders under: {root_val_folder}")
    for i, cdir in enumerate(class_dirs, 1):
        res = process_class_folder(cdir, params, workers=workers)
        if res is None:
            print(f"[{i}/{total}] {os.path.basename(cdir)} -> no valid images or all failed")
            continue
        cname, median, mean = res
        print(f"[{i}/{total}] {cname} -> median={median:.4f}, mean={mean:.4f}")
        rows.append((cname, f"{median:.6f}", f"{mean:.6f}"))

    if not rows:
        print("No results produced. Check your folder structure and image formats.")
        return

    # 用 UTF-8-SIG 方便 Excel 打开
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "median", "mean"])
        writer.writerows(rows)
    print(f"Done. Saved to: {output_csv}")

if __name__ == "__main__":
    # 简单的命令行参数解析（也可换 argparse）
    if len(sys.argv) < 2:
        print("Usage:\n  python hd_batch_folder_stats.py <VAL_ROOT> [OUTPUT_CSV]")
        print('Example:\n  python hd_batch_folder_stats.py "C:\\Users\\you\\Desktop\\val" val_stats.csv')
        sys.exit(1)
    val_root = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) >= 3 else "hd_stats.csv"

    # 你可以根据数据特点调整以下默认参数（减少偏大）：
    main(
        root_val_folder=val_root,
        output_csv=out_csv,
        use_edges=False,        # 线条/结构主导的图像更稳（不想用边缘就改为 False）
        gaussian_sigma=0.0,    # 轻度平滑，压噪
        threshold="otsu",      # 自动阈值
        n_offsets=6,           # 多偏移平均，减少网格对齐偏差
        min_boxes=30,          # 严一点，剔除小样本盒数的尺度
        max_occupancy=0.90,    # 占空比过高（太粗尺度）剔除
        min_occupancy=0.01,    # 占空比过低（噪点主导）剔除
        workers=0              # 多进程并行：如 8（按 CPU 调整）；0/1 为单进程
    )
