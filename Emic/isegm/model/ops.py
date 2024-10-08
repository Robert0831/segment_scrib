import torch
from torch import nn as nn
import numpy as np
import isegm.model.initializer as initializer


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale

        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=scale,
            padding=1,
            groups=groups,
            bias=False)

        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            from isegm.utils.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols, out_dist_map=False):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2  #有正負樣本吧
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)  # 切成 兩個不同行數的tensor

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0   #torch.max 會有最大值 跟　index, 回傳一條true  false
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array) #列出所有x對y組合 但分別存在兩個矩陣相同的位置
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1) #複製成跟所有點的數量一樣 points.size(0),2,h,w

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy) #x-x y-y 
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale) # 除5
            coords.mul_(coords) #x^2 y^2

            coords[:, 0] += coords[:, 1] # x=x+y
            coords = coords[:, :1]  #取出x

            coords[invalid_points, :, :, :] = 1e6 #超大

            coords = coords.view(-1, num_points, 1, rows, cols)  #-1=bs *num_masks * 2  (正負樣本)    感覺應該沒有num_masks,一次應該一張
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w        把我所點的地方看做一個集合,看其他的點離我這些點的最近距離  
            coords = coords.view(-1, 2, rows, cols) #分正負樣本,算出所有點與我那些點的距離

        coord_dist_map = coords.clone()
        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()  #轉0,1
        else:
            coords.sqrt_().mul_(2).tanh_()
        if out_dist_map:
            return coords, coord_dist_map
        return coords

    def forward(self, x, coords, out_dist_map=False):
        if isinstance(x, (list, tuple)):
            b, h, w = x
        else:
            b, h, w = x.shape[0], x.shape[2], x.shape[3]
        return self.get_coord_features(coords, b, h, w, out_dist_map)

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)      #### 令一個 init_value / lr_mult 的數
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class BatchImageNormalize:
    def __init__(self, mean, std, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]

    def __call__(self, tensor):
        tensor = tensor.clone()

        tensor.sub_(self.mean.to(tensor.device)).div_(self.std.to(tensor.device))
        return tensor
