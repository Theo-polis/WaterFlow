import os
import albumentations
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch
from .data_val import (
    cv_random_flip, randomCrop, randomRotation, colorEnhance, 
    randomGaussian, randomPeper, random_modified
)
from dataset.dataset_utils import boundary_modification


class UnderwaterPolypObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, depth_root=None, 
                 mean=None, std=None, randomPeper=True, boundary_modification=False, 
                 boundary_args={}, use_depth=True, depth_estimation_method="midas"):
        
        self.trainsize = trainsize
        self.use_depth = use_depth
        self.depth_estimation_method = depth_estimation_method

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        if depth_root is not None and os.path.exists(depth_root):
            self.depths = [depth_root + f for f in os.listdir(depth_root) 
                          if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.npy')]
            self.use_precomputed_depth = True
        else:
            self.depths = None
            self.use_precomputed_depth = False
            if use_depth:
                self.depth_estimator = self._load_depth_estimator()

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if self.depths is not None:
            self.depths = sorted(self.depths)
        self.filter_files()
        self.aug_transform = albumentations.Compose([
            albumentations.RandomScale(scale_limit=0.25, p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=15, p=0.5),
            albumentations.RandomRotate90(p=0.5),
        ])

        self.img_transform = self.get_transform(mean, std)
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

        self.size = len(self.images)
        self.do_randomPeper = randomPeper
        self.do_boundary_modification = boundary_modification
        self.boundary_args = boundary_args

    def _load_depth_estimator(self):
        """加载深度估计模型"""
        try:
            if self.depth_estimation_method == "midas":
                import torch
                model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
                model.eval()
                return model
            else:
                print(f"Depth estimation method '{self.depth_estimation_method}' not supported")
                return None
        except Exception as e:
            print(f"Could not load depth estimator: {e}")
            return None

    def _estimate_depth(self, image):
        """从RGB图像估计深度"""
        if not hasattr(self, 'depth_estimator') or self.depth_estimator is None:
            return self._heuristic_depth_estimation(image)
        
        try:
            with torch.no_grad():
                input_image = np.array(image.resize((384, 384)))
                input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
                input_tensor = input_tensor.unsqueeze(0)
                depth = self.depth_estimator(input_tensor)
                depth = depth.squeeze().cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = Image.fromarray((depth * 255).astype(np.uint8))
                return depth_img
                
        except Exception as e:
            print(f"Depth estimation failed: {e}, using heuristic method")
            return self._heuristic_depth_estimation(image)

    def _heuristic_depth_estimation(self, image):
        img_array = np.array(image).astype(np.float32) / 255.0
        if len(img_array.shape) == 3:
            blue_dominance = img_array[:, :, 2] - (img_array[:, :, 0] + img_array[:, :, 1]) / 2
            brightness = img_array.mean(axis=2)
            depth_estimate = np.clip(blue_dominance + (1 - brightness), 0, 1)
        else:
            depth_estimate = 1 - img_array
        depth_img = Image.fromarray((depth_estimate * 255).astype(np.uint8))
        return depth_img

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform

    def __getitem__(self, index):
        data = {}
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.use_depth:
            if self.use_precomputed_depth:
                depth_path = self.depths[index]
                if depth_path.endswith('.npy'):
                    depth_array = np.load(depth_path)
                    depth = Image.fromarray((depth_array * 255).astype(np.uint8))
                else:
                    depth = self.binary_loader(depth_path)
            else:
                depth = self._estimate_depth(image)
        else:
            depth = None
        
        image_size = image.size

        if depth is not None:
            augmented = self.aug_transform(
                image=np.asarray(image), 
                mask=np.asarray(gt),
                mask2=np.asarray(depth)
            )
            image, gt, depth = augmented['image'], augmented['mask'], augmented['mask2']
        else:
            augmented = self.aug_transform(image=np.asarray(image), mask=np.asarray(gt))
            image, gt = augmented['image'], augmented['mask']

        if depth is not None:
            image, gt, depth = albumentations.PadIfNeeded(*image_size[::-1], border_mode=0)(
                image=image, mask=gt, mask2=depth
            ).values()
            image, gt, depth = albumentations.RandomCrop(*image_size[::-1])(
                image=image, mask=gt, mask2=depth
            ).values()
        else:
            image, gt = albumentations.PadIfNeeded(*image_size[::-1], border_mode=0)(
                image=image, mask=gt
            ).values()
            image, gt = albumentations.RandomCrop(*image_size[::-1])(
                image=image, mask=gt
            ).values()

        if self.do_boundary_modification:
            seg = random_modified(gt, **self.boundary_args)
            seg = self.gt_transform(Image.fromarray(seg))
            data['seg'] = seg

        image = colorEnhance(Image.fromarray(image))
        gt = randomPeper(gt) if self.do_randomPeper else Image.fromarray(gt)
        if depth is not None:
            depth = Image.fromarray(depth)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if depth is not None:
            depth = self.depth_transform(depth)
            data['depth'] = depth

        data['image'] = image
        data['gt'] = gt

        return data

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        if self.depths is not None:
            assert len(self.images) == len(self.depths)
            
        images = []
        gts = []
        depths = [] if self.depths is not None else None
        
        for i, (img_path, gt_path) in enumerate(zip(self.images, self.gts)):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                if depths is not None:
                    depths.append(self.depths[i])
                    
        self.images = images
        self.gts = gts
        if depths is not None:
            self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_underwater_loader(image_root, gt_root, batchsize, trainsize,
                         depth_root=None, shuffle=True, num_workers=12, 
                         pin_memory=True, use_depth=True, **kwargs):
    dataset = UnderwaterPolypObjDataset(
        image_root=image_root, 
        gt_root=gt_root, 
        trainsize=trainsize,
        depth_root=depth_root,
        use_depth=use_depth,
        **kwargs
    )
    
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return data_loader


class UnderwaterTestDataset(data.Dataset):
    def __init__(self, image_root, gt_root, testsize, depth_root=None, 
                 mean=None, std=None, use_depth=False):
        super(UnderwaterTestDataset, self).__init__()
        self.testsize = testsize
        self.use_depth = use_depth
        
        self.images = [image_root + f for f in os.listdir(image_root) 
                      if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) 
                   if f.endswith('.tif') or f.endswith('.png')]
        
        if depth_root is not None and os.path.exists(depth_root):
            self.depths = [depth_root + f for f in os.listdir(depth_root) 
                          if f.endswith('.png') or f.endswith('.npy')]
        else:
            self.depths = None
            
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if self.depths:
            self.depths = sorted(self.depths)

        self.transform = self.get_transform(mean, std)
        self.gt_transform = transforms.ToTensor()
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])
        
        self.size = len(self.images)
        self.index = 0

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]

        depth = None
        if self.use_depth and self.depths is not None and self.index < len(self.depths):
            depth_path = self.depths[self.index]
            if depth_path.endswith('.npy'):
                depth_array = np.load(depth_path)
                depth = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            else:
                depth_img = self.binary_loader(depth_path)
                depth = self.depth_transform(depth_img).unsqueeze(0)

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, image_for_post, depth

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size