import os
import warnings
from functools import partial

import math
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_

class SimpleResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, *, time_emb_dim=None, groups=8):
        super().__init__()
        dim_out = dim_out or dim
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if time_emb_dim else None
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )
        
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def forward(self, x, time_emb=None):
        h = self.block1(x)
        
        if time_emb is not None and self.time_mlp is not None:
            time_out = self.time_mlp(time_emb)
            h = h + time_out[..., None, None]
        
        h = self.block2(h)
        return h + self.res_conv(x)


from .net import (
    Mlp, Attention, Block, OverlapPatchEmbed, timestep_embedding,
    PyramidVisionTransformerImpr, DWConv, pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, 
    pvt_v2_b3, pvt_v2_b4, pvt_v2_b5, DropPath, resize, MLP, conv, 
    Downsample, Upsample, EmptyObject
)

from .seathru_physics import SeaThruPhysicsPrior, PhysicsFeatureFusion, PhysicsAwareEncoder

import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d

class UnderwaterDecoder(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(UnderwaterDecoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            SimpleResnetBlock(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim, groups=8),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )

    def forward(self, inputs, timesteps, x):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        c1, c2, c3, c4 = inputs

        _x = [x]
        for blk in self.down:
            if isinstance(blk, SimpleResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)

        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        L34 = self.linear_fuse34(torch.cat([_c4, _c3], dim=1))
        L2 = self.linear_fuse2(torch.cat([L34, _c2], dim=1))
        _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))

        x = torch.cat([_c, x], dim=1)
        for blk in self.up:
            if isinstance(blk, SimpleResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        x = self.pred(x)
        return x, c1, c2, c3, c4


class UnderwaterPVT(PyramidVisionTransformerImpr):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], 
                 sr_ratios=[8, 4, 2, 1], mask_chans=1, 
                 use_physics=True, physics_dim=64):
        
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dims, 
                        num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate, 
                        attn_drop_rate, drop_path_rate, norm_layer, depths, 
                        sr_ratios, mask_chans)
        
        self.use_physics = use_physics
        
        if use_physics:
            self.physics_computer = SeaThruPhysicsPrior()

            self.physics_encoder = PhysicsAwareEncoder(output_dim=physics_dim)

            self.physics_fusion_layers = nn.ModuleList([
                PhysicsFeatureFusion(embed_dims[0], physics_dim),
                PhysicsFeatureFusion(embed_dims[1], physics_dim),
                PhysicsFeatureFusion(embed_dims[2], physics_dim),
                PhysicsFeatureFusion(embed_dims[3], physics_dim),
            ])

    def forward_features(self, x, timesteps, cond_img, depth_map=None):
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0]))
        time_token = time_token.unsqueeze(dim=1)

        B = x.shape[0]
        outs = []

        physics_features = None
        if self.training and self.use_physics and depth_map is not None:
            physics_priors = self.physics_computer.compute_physics_priors(cond_img, depth_map)
            physics_features = self.physics_encoder(physics_priors)

        # stage 1
        x, H, W = self.patch_embed1(cond_img, x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if physics_features is not None:
            target_size = x.shape[2:]
            stage1_physics = F.interpolate(physics_features['stage1'], size=target_size, mode='bilinear')
            x = self.physics_fusion_layers[0](x, stage1_physics)
        
        outs.append(x)

        # stage 2
        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))
        time_token = time_token.unsqueeze(dim=1)
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if physics_features is not None:
            target_size = x.shape[2:]
            stage2_physics = F.interpolate(physics_features['stage2'], size=target_size, mode='bilinear')
            x = self.physics_fusion_layers[1](x, stage2_physics)
            
        outs.append(x)

        # stage 3
        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))
        time_token = time_token.unsqueeze(dim=1)
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if physics_features is not None:
            target_size = x.shape[2:]
            stage3_physics = F.interpolate(physics_features['stage3'], size=target_size, mode='bilinear')
            x = self.physics_fusion_layers[2](x, stage3_physics)
            
        outs.append(x)

        # stage 4
        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))
        time_token = time_token.unsqueeze(dim=1)
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if physics_features is not None:
            target_size = x.shape[2:]
            stage4_physics = F.interpolate(physics_features['stage4'], size=target_size, mode='bilinear')
            x = self.physics_fusion_layers[3](x, stage4_physics)
            
        outs.append(x)

        return outs

    def forward(self, x, timesteps, cond_img, depth_map=None):
        x = self.forward_features(x, timesteps, cond_img, depth_map)
        return x


class underwater_pvt_v2_b4_m(UnderwaterPVT):
    def __init__(self, **kwargs):
        super(underwater_pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], 
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], 
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class UnderwaterNet(nn.Module):
    def __init__(self, class_num=1, mask_chans=1, use_physics=True, 
                 physics_dim=64, **kwargs):
        super(UnderwaterNet, self).__init__()
        self.class_num = class_num
        self.use_physics = use_physics

        self.backbone = underwater_pvt_v2_b4_m(
            in_chans=3, 
            mask_chans=mask_chans, 
            use_physics=use_physics,
            physics_dim=physics_dim
        )

        self.decode_head = UnderwaterDecoder(
            dims=[64, 128, 320, 512], 
            dim=256, 
            class_num=class_num, 
            mask_chans=mask_chans
        )

        self._init_weights()

    def forward(self, x, timesteps, cond_img, depth_map=None):
        features = self.backbone(x, timesteps, cond_img, depth_map)
        features, layer1, layer2, layer3, layer4 = self.decode_head(features, timesteps, x)
        return features

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3',
            'pvt_v2_b4', 'pvt_v2_b4_m', 'pvt_v2_b5',
        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'
        from huggingface_hub import hf_hub_download
        return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        try:
            pretrained_dict = torch.load(self._download_weights('pvt_v2_b4_m'))
            model_dict = self.backbone.state_dict()
            
            # 只加载匹配的权重
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict, strict=False)
            
            print(f"Loaded {len(pretrained_dict)} pretrained weights for backbone")
            print(f"Added {len([k for k in model_dict.keys() if 'physics' in k])} physics-aware parameters")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img, depth_map=None):
        return self.forward(x, timesteps, cond_img, depth_map)

    def extract_features(self, cond_img, depth_map=None):
        return cond_img

# 向后兼容
class net(UnderwaterNet):
    def __init__(self, class_num=1, mask_chans=1, **kwargs):
        super().__init__(class_num=class_num, mask_chans=mask_chans, 
                        use_physics=False, **kwargs)