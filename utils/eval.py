import os
import cv2
import numba
import numpy as np
from tqdm.contrib.concurrent import process_map
from utils.metrics import (
    Smeasure, WeightedFmeasure, Fmeasure, Emeasure,
    _cal_mae, _prepare_data, _prepare_data_safe
)

SM = Smeasure()
WFM = WeightedFmeasure()


@numba.jit(nopython=True)
def generate_parts_numel_combinations(fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel, gt_fg_numel, gt_size):
    bg_fg_numel = gt_fg_numel - fg_fg_numel
    bg_bg_numel = pred_bg_numel - bg_fg_numel

    parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

    mean_pred_value = pred_fg_numel / gt_size
    mean_gt_value = gt_fg_numel / gt_size

    demeaned_pred_fg_value = 1 - mean_pred_value
    demeaned_pred_bg_value = 0 - mean_pred_value
    demeaned_gt_fg_value = 1 - mean_gt_value
    demeaned_gt_bg_value = 0 - mean_gt_value

    combinations = [
        (demeaned_pred_fg_value, demeaned_gt_fg_value),
        (demeaned_pred_fg_value, demeaned_gt_bg_value),
        (demeaned_pred_bg_value, demeaned_gt_fg_value),
        (demeaned_pred_bg_value, demeaned_gt_bg_value),
    ]
    return parts_numel, combinations


def cal_em_with_cumsumhistogram(pred: np.ndarray, gt: np.ndarray, gt_fg_numel, gt_size) -> np.ndarray:
    pred = (pred * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
    fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
    fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
    fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

    fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
    bg___numel_w_thrs = gt_size - fg___numel_w_thrs

    if gt_fg_numel == 0:
        enhanced_matrix_sum = bg___numel_w_thrs
    elif gt_fg_numel == gt_size:
        enhanced_matrix_sum = fg___numel_w_thrs
    else:
        parts_numel_w_thrs, combinations = generate_parts_numel_combinations(
            fg_fg_numel=fg_fg_numel_w_thrs,
            fg_bg_numel=fg_bg_numel_w_thrs,
            pred_fg_numel=fg___numel_w_thrs,
            pred_bg_numel=bg___numel_w_thrs,
            gt_fg_numel=gt_fg_numel,
            gt_size=gt_size,
        )
        results_parts = np.empty(shape=(4, 256), dtype=np.float64)
        for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
            align_matrix_value = (
                2 * (combination[0] * combination[1])
                / (combination[0] ** 2 + combination[1] ** 2 + np.spacing(1))
            )
            enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
            results_parts[i] = enhanced_matrix_value * part_numel
        enhanced_matrix_sum = results_parts.sum(axis=0)

    return enhanced_matrix_sum / (gt_size - 1 + np.spacing(1))


def _load_pair(mask_root, pred_root, mask_name):
    """读取GT和预测图像，尺寸不一致时自动resize预测图。"""
    gt = cv2.imread(os.path.join(mask_root, mask_name), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"GT not found: {mask_name}")

    pred_path = os.path.join(pred_root, mask_name)
    if not os.path.exists(pred_path):
        base = os.path.splitext(mask_name)[0]
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(pred_root, base + ext)
            if os.path.exists(candidate):
                pred_path = candidate
                break

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise FileNotFoundError(f"Prediction not found: {mask_name}")

    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    return pred, gt


def measure_mea(mask_name):
    try:
        pred, gt = _load_pair(mask_root, pred_root, mask_name)

        try:
            pred, gt = _prepare_data(pred=pred, gt=gt)
        except Exception:
            pred, gt = _prepare_data_safe(pred=pred, gt=gt)

        # S-measure
        sm = SM.cal_sm(pred, gt)

        # E-measure (cumsum histogram版本，避免多进程冲突)
        gt_fg_numel = np.count_nonzero(gt)
        gt_size = gt.shape[0] * gt.shape[1]
        changeable_em = cal_em_with_cumsumhistogram(pred, gt, gt_fg_numel, gt_size)

        # Weighted F-measure
        wfm = 0 if np.all(~gt) else WFM.cal_wfm(pred, gt)

        # MAE
        mae = _cal_mae(pred, gt)

        # F-measure，各进程独立实例化避免状态共享
        pred_raw = (pred * 255).astype(np.uint8)
        gt_raw  = (gt  * 255).astype(np.uint8)

        fm_calc = Fmeasure()
        fm_calc.step(pred_raw, gt_raw)
        cfms = fm_calc.changeable_fms[-1] if fm_calc.changeable_fms else []
        fm = float(np.nanmax(cfms)) if len(cfms) and not np.all(np.isnan(cfms)) else 0.0

        em_calc = Emeasure()
        em_calc.step(pred_raw, gt_raw)
        cems = em_calc.changeable_ems[-1] if em_calc.changeable_ems else []
        em = float(np.nanmax(cems)) if len(cems) and not np.all(np.isnan(cems)) else 0.0

        return sm, changeable_em, wfm, mae, fm, em

    except Exception as e:
        print(f"Skipping {mask_name}: {e}")
        return None


def eval(mask_path, pred_path, dataset_name):
    global mask_root, pred_root
    mask_root = mask_path
    pred_root = os.path.join(pred_path, dataset_name)

    if not os.path.exists(mask_root):
        print(f"GT path not found: {mask_root}")
        return None
    if not os.path.exists(pred_root):
        print(f"Prediction path not found: {pred_root}")
        return None

    mask_names = sorted(os.listdir(mask_root))
    results = process_map(measure_mea, mask_names, max_workers=4, chunksize=8)
    results = [r for r in results if r is not None]

    if not results:
        print(f"No valid results for {dataset_name}")
        return None

    sms, ems, wfms, maes, fms, em2018s = zip(*results)

    scores = {
        "Smeasure":  np.mean(sms),
        "wFmeasure": np.mean(wfms),
        "MAE":       np.mean(maes),
        "meanEm":    np.mean(np.array(ems, dtype=np.float64), axis=0).mean(),
        "maxEm":     np.mean(np.array(ems, dtype=np.float64), axis=0).max(),
        "Fmeasure":  np.mean(fms),
        "Emeasure":  np.mean(em2018s),
    }
    print(f"{dataset_name}: { {k: f'{v:.4f}' for k, v in scores.items()} }")
    return scores