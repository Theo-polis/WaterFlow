import torch
from torch import nn
import torch.nn.functional as F

def normalize_to_01(x):
    # Check for NaN or Inf values
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("Warning: NaN or Inf detected in normalize_to_01 input")
        return torch.zeros_like(x)
    
    x_min, x_max = x.min(), x.max()
    range_val = x_max - x_min
    
    if range_val < 1e-8:  # Nearly constant values
        return torch.zeros_like(x)
    else:
        result = (x - x_min) / range_val
        # Final check
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN or Inf detected after normalize_to_01")
            return torch.zeros_like(x)
        return result


def simple_train_val_forward(model: nn.Module, gt=None, image=None, **kwargs):
    if model.training:
        assert gt is not None and image is not None
        return model(gt, image, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs)
        
        # Check model output for NaN/Inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("Warning: NaN or Inf detected in model sample output")
            pred = torch.zeros_like(pred)
        
        if time_ensemble:
            preds = torch.concat(model.history, dim=1).detach().cpu()
            
            # Check history for NaN/Inf
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print("Warning: NaN or Inf detected in model history")
                preds = torch.zeros_like(preds)
            
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }


def train_val_forward(model: nn.Module, gt=None, image=None, seg=None, depth=None, **kwargs):
    if model.training:
        assert gt is not None and image is not None and seg is not None
        # 训练时传入深度图
        return model(gt, image, seg=seg, depth_map=depth, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, depth_map=depth, **kwargs).detach().cpu()
        if time_ensemble:
            """ Here is the function 3, Uncertainty based"""
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }