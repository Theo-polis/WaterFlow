import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from tqdm import tqdm
from rectified_flow_pytorch import RectifiedFlow
from denoising_diffusion_pytorch.simple_diffusion import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    default
)


class ModelAdapter(nn.Module):
    def __init__(self, user_model, pred_objective='x0'):
        super().__init__()
        self.user_model = user_model
        self.current_cond_img = None
        self.current_extra_cond = None
        self.current_depth_map = None
        self.pred_objective = pred_objective

    def set_conditions(self, cond_img, extra_cond=None, depth_map=None):
        self.current_cond_img = cond_img
        self.current_extra_cond = extra_cond
        self.current_depth_map = depth_map

    def forward(self, x, times):
        if hasattr(times, 'dim') and times.dim() == 0:
            times = times.repeat(x.shape[0])

        log_snr = self._time_to_log_snr(times)
        model_output = self.user_model(x, log_snr, self.current_cond_img, self.current_depth_map)
        if self.pred_objective == 'x0':
            predicted_x0_logits = model_output
            predicted_x0 = torch.sigmoid(predicted_x0_logits) * 2.0 - 1.0
            velocity = predicted_x0 - x
            return velocity

        elif self.pred_objective == 'v':
            return model_output

        elif self.pred_objective == 'eps':
            predicted_noise = model_output
            return -predicted_noise

        else:
            raise ValueError(f"Unsupported pred_objective: {self.pred_objective}")

    def _time_to_log_snr(self, t):
        """不进行转换"""
        import math
        s = 0.008
        ddmp_time = t
        # return -2 * torch.log(torch.tan((ddmp_time + s) / (1 + s) * math.pi / 2))
        return t


class CondRectifiedFlow(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size: int,
            channels: int = 1,
            extra_channels: int = 0,
            cond_channels: int = 3,
            loss_type: str = 'l2',
            num_sample_steps: int = 20,
            solver_method: str = "euler",
            path_type: str = "cond_ot",
            sigma_min: float = 1e-4,
            clip_sample_denoised: bool = True,
            pred_objective: str = 'x0',
            odeint_kwargs: dict = None,
            predict: str = 'flow',  # 'flow' 或 'noise'
            rf_loss_fn: str = 'mse',
            noise_schedule=None,
            clip_during_sampling: bool = False,
            clip_values: tuple = (-1., 1.),
            **kwargs
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.extra_channels = extra_channels
        self.cond_channels = cond_channels
        self.num_sample_steps = num_sample_steps
        self.solver_method = solver_method
        self.clip_sample_denoised = clip_sample_denoised
        self.sigma_min = sigma_min
        self.history = []
        self.model_adapter = ModelAdapter(model, pred_objective)

        if odeint_kwargs is None:
            odeint_kwargs = dict(
                atol=1e-5,
                rtol=1e-5,
                method='midpoint' if solver_method == 'midpoint' else 'euler'
            )

        self.rf = RectifiedFlow(
            model=self.model_adapter,
            predict=predict,
            loss_fn=rf_loss_fn,
            noise_schedule=noise_schedule or (lambda t: t),
            odeint_kwargs=odeint_kwargs,
            data_normalize_fn=normalize_to_neg_one_to_one,
            data_unnormalize_fn=unnormalize_to_zero_to_one,
            clip_during_sampling=clip_during_sampling,
            clip_values=clip_values,
            **kwargs
        )

        if loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            try:
                from utils.import_utils import get_obj_from_str
                self.external_loss_fn = get_obj_from_str(loss_type)
                self.use_external_loss = True
            except Exception as e:
                print(f"Warning: Could not import custom loss function {loss_type}, using RF's loss. Error: {e}")
                self.use_external_loss = False
        else:
            self.use_external_loss = False
            self.simple_loss_type = loss_type

    @property
    def device(self):
        return next(self.model_adapter.user_model.parameters()).device

    def forward(self, img, cond_img, seg=None, extra_cond=None, depth_map=None, *args, **kwargs):
        b, channels, h, w = img.shape
        cond_channels = cond_img.shape[1]

        assert channels == self.channels
        assert h == w == self.image_size
        assert cond_channels == self.cond_channels

        if extra_cond is None:
            extra_cond = torch.zeros((b, self.extra_channels, h, w), device=self.device)

        self.model_adapter.set_conditions(cond_img, extra_cond, depth_map)

        if seg is not None:
            target_data = normalize_to_neg_one_to_one(seg)
        else:
            target_data = img

        if self.use_external_loss:
            model_prediction = self.get_model_prediction(img, cond_img, seg, extra_cond, depth_map)
            target = (target_data + 1) / 2
            return self.external_loss_fn(model_prediction, target)

        elif hasattr(self, 'simple_loss_type'):
            model_prediction = self.get_model_prediction(img, cond_img, seg, extra_cond, depth_map)
            target = (target_data + 1) / 2

            if self.simple_loss_type == 'l2':
                return F.mse_loss(model_prediction, target)
            elif self.simple_loss_type == 'l1':
                return F.l1_loss(model_prediction, target)
            elif self.simple_loss_type == 'l1+l2':
                return F.mse_loss(model_prediction, target) + F.l1_loss(model_prediction, target)
            elif self.simple_loss_type == 'mean(l1, l2)':
                return (F.mse_loss(model_prediction, target) + F.l1_loss(model_prediction, target)) / 2
        else:
            rf_loss = self.rf(target_data)
            return rf_loss

    def get_model_prediction(self, img, cond_img, seg=None, extra_cond=None, depth_map=None):
        b, channels, h, w = img.shape
        if extra_cond is None:
            extra_cond = torch.zeros((b, self.extra_channels, h, w), device=self.device)
        self.model_adapter.set_conditions(cond_img, extra_cond, depth_map)
        target_data = normalize_to_neg_one_to_one(seg if seg is not None else img)
        times = torch.rand(b, device=self.device)
        noise = torch.randn_like(target_data)
        noised_data = noise.lerp(target_data, times.view(-1, 1, 1, 1))
        log_snr = self.model_adapter._time_to_log_snr(times)
        model_output = self.model_adapter.user_model(noised_data, log_snr, cond_img, depth_map)

        return model_output

    @torch.no_grad()
    def sample(self, cond_img, extra_cond=None, depth_map=None, verbose=True):
        b, c, h, w = cond_img.shape
        if extra_cond is None:
            extra_cond = torch.zeros((b, self.extra_channels, h, w), device=self.device)
        self.model_adapter.set_conditions(cond_img, extra_cond, depth_map)
        self.history = []
        result = self._sample_with_history(b, cond_img, extra_cond, depth_map, verbose)
        if self.clip_sample_denoised:
            result = result.clamp(0., 1.)
        return result

    def _sample_with_history(self, batch_size, cond_img, extra_cond, depth_map, verbose=True):
        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.device)
        dt = 1.0 / self.num_sample_steps
        steps = tqdm(range(self.num_sample_steps), desc='Rectified Flow Sampling (Fixed)') if verbose else range(
            self.num_sample_steps)
        for i in steps:
            t = i * dt
            t_tensor = torch.full((batch_size,), t, device=self.device)
            velocity = self._get_velocity_at_time(x, t_tensor)
            x = x + velocity * dt
            self.history.append(x.clone())
            if self.clip_sample_denoised and i % 5 == 0:
                x.clamp_(-1.5, 1.5)
        if self.clip_sample_denoised:
            x.clamp_(-1., 1.)
        result = unnormalize_to_zero_to_one(x)
        result = torch.sigmoid((result - 0.5) * 6.0)

        return result

    def _get_velocity_at_time(self, x, times):
        log_snr = self.model_adapter._time_to_log_snr(times)
        model_output = self.model_adapter.user_model(x, log_snr, self.model_adapter.current_cond_img, self.model_adapter.current_depth_map)
        if self.model_adapter.pred_objective == 'x0':
            predicted_x0_logits = model_output
            predicted_x0 = torch.sigmoid(predicted_x0_logits) * 2.0 - 1.0
            velocity = predicted_x0 - x
            return velocity
        elif self.model_adapter.pred_objective == 'v':
            return model_output
        elif self.model_adapter.pred_objective == 'eps':
            return -model_output
        else:
            raise ValueError(f"Unsupported pred_objective: {self.model_adapter.pred_objective}")

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_img, extra_cond, depth_map=None, verbose=True):
        return self.sample(cond_img, extra_cond, depth_map, verbose)

CondGaussianDiffusion = CondRectifiedFlow