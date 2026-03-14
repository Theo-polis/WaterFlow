import torch
import torch.nn as nn
import torch.nn.functional as F

class SeaThruPhysicsPrior:
    """
    基于SeaThru模型的水下物理先验计算。

    SeaThru将水下成像分解为直接信号和后向散射两个分量，
    并对二者使用不同的衰减系数(βD ≠ βB)，这是与传统
    暗通道先验方法的核心区别。

    参考: Akkaynak & Treibitz, "Sea-Thru: A Method for Removing
          Water From Underwater Images", CVPR 2019.
    """

    def __init__(self, device='cuda'):
        self.device = device

    def compute_physics_priors(self, rgb_image, depth_map):
        """
        计算SeaThru物理先验特征。

        Args:
            rgb_image: [B, 3, H, W] 水下RGB图像，值域[0,1]
            depth_map: [B, 1, H, W] 场景深度图

        Returns:
            dict: 物理先验特征，包含后向散射、传输图、恢复场景等
        """
        return self._compute_fast_approximation(rgb_image, depth_map)

    def _compute_fast_approximation(self, rgb_image, depth_map):
        """
        GPU友好的SeaThru近似实现。

        保持βD ≠ βB的核心约束，用经验参数代替原论文中的
        逐图像非线性拟合，以适配batch训练场景。
        """
        B, C, H, W = rgb_image.shape

        background_light = self._gpu_background_estimation(rgb_image, depth_map)
        beta_B = self._estimate_beta_B_gpu(rgb_image, depth_map, background_light)
        beta_D = self._estimate_beta_D_gpu(rgb_image, depth_map, background_light, beta_B)

        z_map = depth_map[:, 0:1]  # [B, 1, H, W]
        T_D = torch.exp(-beta_D * z_map)  # 直接分量传输图 [B, 3, H, W]
        T_B = torch.exp(-beta_B * z_map)  # 后向散射传输图 [B, 3, H, W]

        # Bc = B_inf * (1 - T_B)
        seathru_backscatter = background_light * (1 - T_B)

        # Dc = clamp(Ic - Bc)
        seathru_direct = torch.clamp(rgb_image - seathru_backscatter, 0.01, 1.0)

        # Jc = Dc / T_D
        recovered_scene = torch.clamp(
            seathru_direct / torch.clamp(T_D, 0.01, 1.0),
            0, 2.0
        )

        advanced_features = self._compute_advanced_features(
            seathru_backscatter, T_D, beta_D, depth_map
        )

        return {
            'background_light': background_light,  # [B, 3, 1, 1]
            'transmission_m': T_D[:, :2].mean(dim=1, keepdim=True),  # [B, 1, H, W]
            'transmission_b': T_D[:, 2:3],  # [B, 1, H, W]
            'restored_scene': recovered_scene,  # [B, 3, H, W]
            'color_distortion': torch.abs(seathru_direct - recovered_scene),  # [B, 3, H, W]
            'depth': depth_map,  # [B, 1, H, W]
            'seathru_backscatter': seathru_backscatter,  # [B, 3, H, W]
            'seathru_direct': seathru_direct,  # [B, 3, H, W]
            'range_dependent_beta': beta_D,  # [B, 3, H, W]
            'seathru_transmission': T_D,  # [B, 3, H, W]
            'beta_B_map': beta_B.expand(-1, C, -1, -1),  # [B, 3, H, W]
            'T_B_map': T_B.expand(-1, C, -1, -1),  # [B, 3, H, W]
            **advanced_features
        }

    def _gpu_background_estimation(self, image, depth):
        """
        背景光估计。

        将深度范围划分为10个等间距区间，在每个区间内
        选取亮度最低的1%像素，取其均值作为该深度处的
        背景光估计，与SeaThru原文的depth clustering策略一致。
        """
        B, C, H, W = image.shape
        depth_flat = depth.view(B, -1)
        z_min = depth_flat.min(dim=1)[0]
        z_max = depth_flat.max(dim=1)[0]

        background_lights = []
        for b in range(B):
            img_b = image[b]
            depth_b = depth[b, 0]
            z_bins = torch.linspace(z_min[b], z_max[b], 11, device=image.device)
            dark_pixels = []

            for i in range(10):
                if i == 9:
                    mask = (depth_b >= z_bins[i]) & (depth_b <= z_bins[i + 1])
                else:
                    mask = (depth_b >= z_bins[i]) & (depth_b < z_bins[i + 1])

                if mask.sum() == 0:
                    continue

                masked_pixels = img_b[:, mask]
                brightness = masked_pixels.mean(dim=0)
                num_dark = max(1, int(0.01 * brightness.shape[0]))
                _, dark_indices = torch.topk(brightness, num_dark, largest=False)
                dark_pixels.append(masked_pixels[:, dark_indices])

            if dark_pixels:
                bg_light = torch.cat(dark_pixels, dim=1).mean(dim=1, keepdim=True)
            else:
                # 区间内无有效像素时，退化为全图最暗5%
                brightness = img_b.mean(dim=0).view(-1)
                num_dark = max(1, H * W // 20)
                _, dark_indices = torch.topk(brightness, num_dark, largest=False)
                bg_light = img_b.view(3, -1)[:, dark_indices].mean(dim=1, keepdim=True)

            background_lights.append(bg_light)

        background_light = torch.stack(background_lights, dim=0).unsqueeze(-1)  # [B, 3, 1, 1]
        return torch.clamp(background_light, 0.1, 0.9)

    def _estimate_beta_B_gpu(self, image, depth, background_light):
        """
        估计后向散射系数βB。

        以SeaThru论文中典型海水参数[0.15, 0.12, 0.08](R/G/B)为基准，
        根据图像相对于背景光的暗度进行自适应调整。
        """
        B, C, H, W = image.shape
        beta_B_base = torch.tensor([0.15, 0.12, 0.08], device=image.device)

        img_mean = image.mean(dim=[2, 3])
        bg_mean = background_light.squeeze(-1).squeeze(-1)
        darkness_factor = 1.0 + 0.5 * (1.0 - img_mean / torch.clamp(bg_mean, 0.1, 1.0))

        beta_B = torch.clamp(beta_B_base[None, :] * darkness_factor, 0.05, 0.5)
        return beta_B.view(B, 3, 1, 1)

    def _estimate_beta_D_gpu(self, image, depth, background_light, beta_B):
        """
        估计直接衰减系数βD。

        用两项指数之和近似SeaThru的距离相关衰减模型：
            βD(z) ≈ 0.7·exp(-0.15·z̃) + 0.3·exp(-0.05·z̃)
        其中z̃为归一化深度。强制βD > βB以满足SeaThru的物理约束。
        """
        B, C, H, W = image.shape
        z = depth[:, 0]

        base_beta_D = torch.tensor([0.8, 0.4, 0.2], device=image.device).view(1, 3, 1, 1)
        z_norm = z / torch.clamp(
            z.max(dim=-1, keepdim=True)[0].max(dim=-1, keepdim=True)[0],
            1.0, 100.0
        )
        distance_modulation = 0.7 * torch.exp(-0.15 * z_norm) + 0.3 * torch.exp(-0.05 * z_norm)
        beta_D = base_beta_D * distance_modulation.unsqueeze(1)

        # 物理约束：直接分量衰减应快于后向散射分量
        beta_D = torch.maximum(beta_D, beta_B + 0.1)
        beta_D = torch.minimum(beta_D, torch.tensor(2.0, device=beta_D.device))
        return beta_D

    def _compute_advanced_features(self, backscatter, transmission, beta_D, depth):
        """计算用于PhysicsAwareEncoder各stage的辅助特征。"""
        return {
            'depth_gradient': self._compute_depth_gradient(depth),
            'transmission_channel_variance': torch.var(transmission, dim=1, keepdim=True),
            'attenuation_channel_variance': torch.var(beta_D, dim=1, keepdim=True),
            # 蓝/红、蓝/绿传输比反映水体对不同波长的选择性衰减
            'transmission_ratio_br': transmission[:, 2:3] / torch.clamp(transmission[:, 0:1], 0.01, 1.0),
            'transmission_ratio_bg': transmission[:, 2:3] / torch.clamp(transmission[:, 1:2], 0.01, 1.0),
            'attenuation_intensity': 1 - transmission.mean(dim=1, keepdim=True),
            'backscatter_intensity': backscatter.mean(dim=1, keepdim=True),
        }

    def _compute_depth_gradient(self, depth):
        """Sobel算子计算深度梯度，用于捕捉物体边界处的深度不连续性。"""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=depth.dtype, device=depth.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=depth.dtype, device=depth.device
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2)


class PhysicsFeatureFusion(nn.Module):
    """
    物理先验特征与backbone特征的跨模态融合模块。

    用可学习的注意力权重控制物理先验对视觉特征的影响程度，
    残差连接保证在物理先验无效时模型退化为原始backbone。
    """

    def __init__(self, backbone_dim, physics_dim):
        super().__init__()
        self.physics_proj = nn.Sequential(
            nn.Conv2d(physics_dim, backbone_dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(backbone_dim // 2, backbone_dim, 1)
        )
        self.cross_attention = nn.Sequential(
            nn.Conv2d(backbone_dim * 2, backbone_dim, 1),
            nn.ReLU(),
            nn.Conv2d(backbone_dim, backbone_dim, 1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(backbone_dim * 2, backbone_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(backbone_dim, backbone_dim, 1)
        )
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, backbone_feat, physics_feat):
        """
        Args:
            backbone_feat: [B, backbone_dim, H, W]
            physics_feat:  [B, physics_dim,  H, W]

        Returns:
            [B, backbone_dim, H, W]
        """
        physics_projected = self.physics_proj(physics_feat)
        attention_weight = self.cross_attention(
            torch.cat([backbone_feat, physics_projected], dim=1)
        )
        fused_feat = self.fusion_conv(
            torch.cat([backbone_feat, attention_weight * physics_projected], dim=1)
        )
        return backbone_feat + self.residual_weight * fused_feat


class PhysicsAwareEncoder(nn.Module):
    """
    分层物理先验编码器。

    将SeaThru物理量按语义分组，分别编码后对应注入PVT的4个stage：
      Stage 1 — 后向散射 + 深度梯度（局部边界信息）
      Stage 2 — 距离相关衰减系数 + 通道传输比（水体光学特性）
      Stage 3 — 传输图 + 通道方差（颜色失真分布）
      Stage 4 — 恢复场景 + 衰减强度（全局语义）
    """

    def __init__(self, output_dim=64):
        super().__init__()

        def _encoder(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, output_dim, 3, 1, 1),
                nn.ReLU()
            )

        self.stage1_encoder = _encoder(4)  # backscatter(3) + depth_gradient(1)
        self.stage2_encoder = _encoder(5)  # range_dependent_beta(3) + ratio_br(1) + ratio_bg(1)
        self.stage3_encoder = _encoder(5)  # transmission(3) + trans_var(1) + atten_var(1)
        self.stage4_encoder = _encoder(5)  # restored_scene(3) + backscatter_intensity(1) + attenuation(1)

    def forward(self, physics_priors):
        """
        Args:
            physics_priors: compute_physics_priors()返回的特征字典

        Returns:
            dict: {'stage1'~'stage4': [B, output_dim, H, W]}
        """
        s1 = self.stage1_encoder(torch.cat([
            physics_priors['seathru_backscatter'],
            physics_priors['depth_gradient'],
        ], dim=1))

        s2 = self.stage2_encoder(torch.cat([
            physics_priors['range_dependent_beta'],
            physics_priors['transmission_ratio_br'],
            physics_priors['transmission_ratio_bg'],
        ], dim=1))

        s3 = self.stage3_encoder(torch.cat([
            physics_priors['seathru_transmission'],
            physics_priors['transmission_channel_variance'],
            physics_priors['attenuation_channel_variance'],
        ], dim=1))

        s4 = self.stage4_encoder(torch.cat([
            physics_priors['restored_scene'],
            physics_priors['backscatter_intensity'],
            physics_priors['attenuation_map'],
        ], dim=1))

        return {'stage1': s1, 'stage2': s2, 'stage3': s3, 'stage4': s4}